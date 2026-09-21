//! `CogneeServices` — the single place where the 6 raw engines from
//! `ComponentManager` and all derived services are built and cached.
//!
//! This is the keystone facade for the SDK bindings: every `sdk_*` function
//! obtains a `CogneeServices` via `HandleState::services()` and calls a
//! `cognee` API with the bundled `Arc<dyn …>` handles, so the wiring lives
//! in exactly one place (mirroring the CLI command builders, which are the
//! authoritative reference).

use std::sync::Arc;

use uuid::Uuid;

use cognee::ComponentManager;
use cognee::PipelineContext;
use cognee::add::AddPipeline;
use cognee::api::get_or_create_default_user;
use cognee::cognify::{ChunkStrategy, CognifyConfig};
use cognee::core::{CpuPool, RayonThreadPool};
use cognee::database::{
    CheckpointStore, DatabaseConnection, DeleteDb, IngestDb, PipelineRunRepository,
    SeaOrmCheckpointStore, SeaOrmPipelineRunRepository, SearchHistoryDb,
};
use cognee::delete::DeleteService;
use cognee::embedding::EmbeddingEngine;
use cognee::graph::GraphDBTrait;
use cognee::llm::Llm;
use cognee::ontology::{NoOpOntologyResolver, OntologyResolver, RdfLibOntologyResolver};
use cognee::search::{
    SeaOrmSessionStore, SearchBuilder, SearchOrchestrator, SessionManager, SessionStore,
};
use cognee::storage::StorageTrait;
use cognee::vector::VectorDB;

use crate::SdkError;

/// A fully-wired bundle of engines + derived services.
///
/// Built once per config version by [`CogneeServices::build`] and cached by the
/// handle. All fields are `Arc`-shared so `sdk_*` functions can cheaply clone a
/// handle into a `cognee` API call.
// Most fields are consumed by the SDK ops added in later phases; they are part
// of the facade contract now so the wiring lives in one place.
#[allow(dead_code)]
pub struct CogneeServices {
    // 6 raw engines from `ComponentManager` (the `PipelineContext` surface).
    pub storage: Arc<dyn StorageTrait>,
    /// Concrete SeaORM connection. `DatabaseConnection` implements every DB
    /// trait, so derived services coerce it via `Arc::clone(&database) as Arc<dyn …>`.
    pub database: Arc<DatabaseConnection>,
    pub graph_db: Arc<dyn GraphDBTrait>,
    pub vector_db: Arc<dyn VectorDB>,
    pub embedding_engine: Arc<dyn EmbeddingEngine>,
    pub llm: Arc<dyn Llm>,

    // Derived services (built here; see the §4 facade table in the plan).
    pub thread_pool: Arc<RayonThreadPool>,
    pub pipeline_run_repo: Arc<dyn PipelineRunRepository>,
    pub add_pipeline: Arc<AddPipeline>,
    pub delete_service: Arc<DeleteService>,
    pub search_orchestrator: Arc<SearchOrchestrator>,
    pub session_store: Arc<dyn SessionStore>,
    pub session_manager: Arc<SessionManager>,
    pub ontology_resolver: Arc<dyn OntologyResolver>,
    pub cognify_config: CognifyConfig,
    pub checkpoint_store: Arc<dyn CheckpointStore>,
}

/// Relational databases this process has already considered for a startup
/// recovery sweep, keyed by resolved URL.
///
/// Per database, not a single process-wide flag: one process can build
/// services against more than one relational URL (a test harness, an embedder
/// switching tenants), and a bare flag would recover the first and silently
/// skip every other. Per *process* and not per handle or per config version,
/// because `CogneeServices::build` runs again on every config-version change,
/// and by then a cognify started through an earlier handle may hold a claim
/// that is very much alive. Sweeping then would drop it and admit a second run
/// into the same dataset — the exact failure the claim prevents. The only
/// moment at which every leftover in a database is provably dead is the first
/// time this process touches it.
static SWEPT_DATABASES: std::sync::Mutex<std::collections::BTreeSet<String>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// Record `relational_db_url` as considered, reporting whether this call is
/// the first to do so.
///
/// Deliberately called *before* the single-process assertion is consulted.
/// Marking only on the sweeping path would leave a database unmarked when the
/// first build resolves `false`, so a later build that resolves `true` — after
/// a config change flipped the flag or repointed the URL — would sweep with
/// this process's own runs already in flight.
fn claim_first_touch(relational_db_url: &str) -> bool {
    let mut swept = match SWEPT_DATABASES.lock() {
        Ok(guard) => guard,
        // A panic in another holder says nothing about this set's contents:
        // the only mutation is one `insert`, so the worst a poisoned lock can
        // mean is that the insert did or did not happen. Recovering keeps the
        // "at most one sweep" guarantee; propagating would turn it into a
        // panic on every later build.
        Err(poisoned) => poisoned.into_inner(),
    };
    swept.insert(relational_db_url.to_string())
}

/// Clear what a killed run left behind on this database — once per database
/// per process, and only where the deployment asserts one process per
/// relational database.
///
/// A run killed mid-flight (SIGKILL, OOM, an Android process kill) wedges its
/// dataset behind **two** independent gates, and clearing one alone is worth
/// nothing:
///
/// 1. The `pipeline_runs` row left at `Initiated`/`Started`.
///    `check_pipeline_run_qualification` reads it *first*, before any claim is
///    consulted, and returns `AlreadyRunning`. It never expires. Until this
///    landed, the only sweep that retired it ran at HTTP-server startup, so an
///    embedded consumer never reached it at all.
/// 2. The exclusive-run claim. Released only by its holder
///    (`release_pipeline_run_claim` filters on the `claim_id` that died with
///    it), so liveness is inferred from age against a day-long window.
///
/// `cognee-cli pipeline-unblock` clears both for the same reason, and says so
/// in its own module docs. An embedded consumer has neither that CLI nor an
/// HTTP server.
///
/// Safety: both clears are unscoped, so both are sound **only** where no peer
/// process can hold what they drop. With one process per database, every
/// leftover present the first time this process touches it was written by a
/// dead incarnation of this same process. Anything else — a shared Postgres,
/// or the *file-backed* SQLite that is the shipped default — derives `false`
/// and is left entirely alone. `COGNEE_SINGLE_PROCESS` is how a deployment
/// that does own its file (one process per device, say) opts in; the SDK
/// cannot infer that from the URL.
///
/// The two steps are attempted independently: a transient failure on one must
/// not suppress the other, since the dataset stays wedged unless both go.
/// Both are best-effort — a failure is logged and ignored, because refusing to
/// build the SDK over a recovery convenience would turn a recoverable wedge
/// into a hard startup failure.
async fn sweep_killed_run_leftovers(
    cm: &ComponentManager,
    pipeline_run_repo: &Arc<dyn PipelineRunRepository>,
) {
    // Snapshot under the read guard and drop it before the `.await`:
    // `RwLockReadGuard` is `!Send`, and this future is awaited from PyO3
    // bindings that require `Send`.
    let (relational_db_url, single_process) = {
        let settings = cm.settings();
        (
            settings.resolved_relational_db_url(),
            settings.resolved_single_process(),
        )
    };

    // Marked first, assertion checked second — see `claim_first_touch`.
    if !claim_first_touch(&relational_db_url) {
        return;
    }
    if !single_process {
        return;
    }

    match pipeline_run_repo
        .reset_orphans("sdk_startup_orphan_single_process")
        .await
    {
        Ok(0) => {}
        Ok(reset) => tracing::warn!(
            reset,
            "retired pipeline-run rows left in flight by a previous process; the datasets \
             they blocked are runnable again"
        ),
        Err(e) => tracing::warn!(
            "startup reset of orphaned pipeline runs failed (non-fatal); a dataset wedged by \
             a killed run stays wedged, and this gate never expires: {e}"
        ),
    }

    match pipeline_run_repo
        .release_all_pipeline_run_claims("sdk_startup_sweep_single_process")
        .await
    {
        Ok(0) => {}
        Ok(released) => tracing::warn!(
            released,
            "released pipeline-run claims left behind by a previous process"
        ),
        Err(e) => tracing::warn!(
            "startup sweep of pipeline-run claims failed (non-fatal); a dataset wedged by a \
             killed run stays wedged until its claim ages out: {e}"
        ),
    }
}

impl CogneeServices {
    /// Build the full bundle from a `ComponentManager`, returning the bundle and
    /// the resolved owner id.
    ///
    /// Owner id is the OSS default user materialised by
    /// `get_or_create_default_user(&settings)`: it is the parsed
    /// `settings.default_user_id` UUID. The closed cloud build replaces this
    /// helper with a DB-backed equivalent that upserts a row in the `users`
    /// table; the call shape is identical, so this assembly path is unchanged.
    ///
    /// The LLM is resolved **strictly** here (the simplest correct v1 per the
    /// plan): callers that need keyless warm must set a non-empty dummy
    /// `llm_api_key` — `OpenAIAdapter::new` performs no network I/O at
    /// construction, so this never reaches the network.
    pub async fn build(cm: &ComponentManager) -> Result<(Self, Uuid), SdkError> {
        // --- 1. Raw engines (errors map to ComponentError → SdkError). ---
        let storage = cm.storage().await?;
        let database = cm.database().await?;
        let graph_db = cm.graph_db().await?;
        let vector_db = cm.vector_db().await?;
        let embedding_engine = cm.embedding_engine().await?;
        let llm = cm.llm().await?;

        // --- 2. Resolve owner id (Python default-user semantics). ---
        // Snapshot the email under the read guard, then drop the guard
        // before the `.await` — `RwLockReadGuard` from `std::sync` is
        // `!Send`, and `CogneeServices::build` is awaited from PyO3
        // bindings that require `Send` futures.
        //
        // owner_id = uuid5(NAMESPACE_OID, email) — must match Python.
        let default_user_email = {
            let settings = cm.settings();
            settings.default_user_email.clone()
        };
        let user = get_or_create_default_user(&default_user_email)
            .await
            .map_err(|e| SdkError::UserBootstrap(e.to_string()))?;
        let owner_id = user.id;

        // --- 3. Derived services (mirrors the CLI command builders). ---
        let thread_pool = Arc::new(
            RayonThreadPool::with_default_threads()
                .map_err(|e| SdkError::ServiceBuild(format!("thread pool: {e}")))?,
        );

        let pipeline_run_repo: Arc<dyn PipelineRunRepository> =
            Arc::new(SeaOrmPipelineRunRepository::new(Arc::clone(&database)));

        sweep_killed_run_leftovers(cm, &pipeline_run_repo).await;

        let add_pipeline = Arc::new(
            AddPipeline::new(
                Arc::clone(&storage),
                Arc::clone(&database) as Arc<dyn IngestDb>,
            )
            .with_thread_pool(Arc::clone(&thread_pool) as Arc<dyn CpuPool>)
            .with_graph_db(Arc::clone(&graph_db))
            .with_vector_db(Arc::clone(&vector_db))
            .with_database(Arc::clone(&database))
            .with_pipeline_run_repo(Arc::clone(&pipeline_run_repo)),
        );

        // Unauthorized DeleteService; the ACL-enforcing wrapper is a later-phase
        // concern.
        let delete_service = Arc::new(
            DeleteService::new(
                Arc::clone(&storage),
                Arc::clone(&database) as Arc<dyn DeleteDb>,
            )
            .with_graph_db(Arc::clone(&graph_db))
            .with_vector_db(Arc::clone(&vector_db))
            .with_pipeline_run_repo(Arc::clone(&pipeline_run_repo)),
        );

        // Session: v1 is always SeaOrmSessionStore (fs/redis features are not
        // built into the default binding configurations).
        let session_store_concrete = SeaOrmSessionStore::new(Arc::clone(&database))
            .await
            .map_err(|e| SdkError::ServiceBuild(format!("session store: {e}")))?;
        let session_store: Arc<dyn SessionStore> = Arc::new(session_store_concrete);
        let session_manager = Arc::new(SessionManager::new(Arc::clone(&session_store)));

        let search_orchestrator = Arc::new(
            SearchBuilder::new(
                Arc::clone(&vector_db),
                Arc::clone(&embedding_engine),
                Arc::clone(&graph_db),
                Arc::clone(&llm),
                Arc::clone(&database) as Arc<dyn SearchHistoryDb>,
            )
            .with_session_manager(Arc::clone(&session_manager))
            .with_dataset_resolver(Arc::clone(&database) as Arc<dyn IngestDb>)
            .build(),
        );

        // Ontology: RdfLib when a path is configured, else NoOp.
        let ontology_resolver: Arc<dyn OntologyResolver> = {
            let path = cm.settings().ontology_file_path.clone();
            if path.trim().is_empty() {
                Arc::new(NoOpOntologyResolver::new())
            } else {
                Arc::new(
                    RdfLibOntologyResolver::new(path.as_str())
                        .map_err(|e| SdkError::ServiceBuild(format!("ontology resolver: {e}")))?,
                )
            }
        };

        // CognifyConfig from Settings. `with_temporal_cognify` is a per-call
        // flag (not a Settings field) and is left at default here.
        let cognify_config = {
            let s = cm.settings();
            let chunk_strategy = match s.chunk_strategy.to_uppercase().as_str() {
                "RECURSIVE" => ChunkStrategy::Recursive,
                _ => ChunkStrategy::Paragraph,
            };
            CognifyConfig::default()
                .with_chunk_size_opt(s.chunk_size.map(|n| n as usize))
                .with_chunk_overlap(s.chunk_overlap as usize)
                .with_chunk_strategy(chunk_strategy)
                .with_max_parallel_extractions(s.llm_max_parallel_requests.max(1) as usize)
        };

        let checkpoint_store: Arc<dyn CheckpointStore> =
            Arc::new(SeaOrmCheckpointStore::new(Arc::clone(&database)));

        let services = CogneeServices {
            storage,
            database,
            graph_db,
            vector_db,
            embedding_engine,
            llm,
            thread_pool,
            pipeline_run_repo,
            add_pipeline,
            delete_service,
            search_orchestrator,
            session_store,
            session_manager,
            ontology_resolver,
            cognify_config,
            checkpoint_store,
        };

        Ok((services, owner_id))
    }

    /// The thread pool as the `dyn CpuPool` some APIs (e.g. cognify) require.
    #[allow(dead_code)] // used by cognify in later phases
    pub fn cpu_pool(&self) -> Arc<dyn CpuPool> {
        Arc::clone(&self.thread_pool) as Arc<dyn CpuPool>
    }
}
