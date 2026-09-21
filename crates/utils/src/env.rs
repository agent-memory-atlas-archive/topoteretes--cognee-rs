//! Environment-variable parsing helpers shared across the workspace.

/// Parse a truthy env-var value: `true | 1 | yes | on` (trimmed, case-insensitive).
/// Everything else (incl. empty) is `false`. Matches the Python SDK's permissive
/// truthy parsing and the previously-private `http-server` helper.
pub fn parse_env_bool(v: &str) -> bool {
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "true" | "1" | "yes" | "on"
    )
}

/// Env var asserting (or denying) that exactly one process uses the relational
/// database. Truthy per [`parse_env_bool`]; any other non-empty value is read
/// as an explicit `false`, so an operator can switch the assertion off as well
/// as on. Unset means "derive it" — see [`single_process_default`].
pub const SINGLE_PROCESS_ENV: &str = "COGNEE_SINGLE_PROCESS";

/// Whether `url` names a SQLite relational database.
///
/// Matches the `starts_with("sqlite")` test `cognee_database::connect` uses to
/// pick a backend, so this answers the same question the connection layer does.
pub fn url_is_sqlite(url: &str) -> bool {
    url.trim().starts_with("sqlite")
}

/// The single-process assertion derived from the relational DB URL alone.
///
/// SQLite is a file (or an in-memory handle) opened by one process; a second
/// cognee process pointed at the same file is not a supported deployment, and
/// a second one pointed at `:memory:` is not even sharing a database. Anything
/// else — Postgres above all — is reachable by many processes at once and must
/// be assumed to be.
pub fn single_process_default(relational_db_url: &str) -> bool {
    url_is_sqlite(relational_db_url)
}

/// Resolve the single-process assertion: an explicit `configured` value if the
/// operator set one, else [`single_process_default`].
pub fn resolve_single_process(relational_db_url: &str, configured: Option<bool>) -> bool {
    configured.unwrap_or_else(|| single_process_default(relational_db_url))
}

/// Read [`SINGLE_PROCESS_ENV`] as an explicit override, if it is set to
/// anything but the empty string.
pub fn single_process_override_from_env() -> Option<bool> {
    let raw = std::env::var(SINGLE_PROCESS_ENV).ok()?;
    if raw.trim().is_empty() {
        return None;
    }
    Some(parse_env_bool(&raw))
}

/// [`resolve_single_process`] with the override taken straight from the
/// environment. For callers that carry no settings object of their own.
pub fn single_process_from_env(relational_db_url: &str) -> bool {
    resolve_single_process(relational_db_url, single_process_override_from_env())
}

#[cfg(test)]
mod tests {
    use super::{parse_env_bool, resolve_single_process, single_process_default, url_is_sqlite};

    #[test]
    fn truthy_and_falsy() {
        for t in ["true", "TRUE", " 1 ", "Yes", "on", "ON"] {
            assert!(parse_env_bool(t), "{t:?} should be truthy");
        }
        for f in ["false", "0", "no", "off", "", "  ", "maybe"] {
            assert!(!parse_env_bool(f), "{f:?} should be falsy");
        }
    }

    #[test]
    fn sqlite_urls_assert_single_process_and_postgres_urls_do_not() {
        for sqlite in [
            "sqlite::memory:",
            "sqlite:///home/u/.cognee_system/cognee.db",
            "sqlite://cognee.db?mode=rwc",
        ] {
            assert!(url_is_sqlite(sqlite), "{sqlite:?} is a sqlite url");
            assert!(
                single_process_default(sqlite),
                "{sqlite:?} must assert single-process by default"
            );
        }
        for shared in [
            "postgres://u:p@host:5432/cognee",
            "postgresql://u:p@host/cognee?sslmode=require",
            "mysql://u:p@host/cognee",
            "",
        ] {
            assert!(
                !single_process_default(shared),
                "{shared:?} must NOT assert single-process by default"
            );
        }
    }

    #[test]
    fn an_explicit_setting_wins_over_the_derived_default_in_both_directions() {
        // The escape hatch that matters: two processes sharing one SQLite file
        // is unsupported but reachable, and an operator who does it must be
        // able to keep the claim doing its job.
        assert!(!resolve_single_process("sqlite::memory:", Some(false)));
        // And the reverse: a single-process deployment that happens to use
        // Postgres can still opt into the startup sweep.
        assert!(resolve_single_process(
            "postgres://u:p@host/cognee",
            Some(true)
        ));
    }
}
