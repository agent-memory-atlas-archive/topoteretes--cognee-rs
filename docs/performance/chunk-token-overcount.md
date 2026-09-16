# The chunk token over-count (SDK-632)

Measurement record for the per-word token over-count in `chunk_by_sentence`,
the fix that made span counting the default, and the claims that did and did
not survive checking.

Reproduce everything here with:

```bash
cargo run -p cognee-chunking --features tiktoken \
    --example measure_token_overcount -- [EXTRA_CORPUS_PATH...]
```

The four corpora under `crates/chunking/src/test_data/` are always measured;
paths on the command line are added to the run. The Alice figures below come
from `https://www.gutenberg.org/files/11/11-0.txt` (151 KB), which is not
committed.

## The defect

`chunk_by_sentence` asked the tokenizer for the size of every word **in
isolation** and summed the results. Sub-word tokenizers are not additive over a
partition: cl100k encodes `" the"` as one token but `"the"` and `" "` as two,
and `chunk_by_word` hands back words with their trailing space attached. Every
merge across a word boundary was therefore lost, and the sum over-stated the
span the words compose.

Nothing was ever *over* the embedder's limit — chunks came in under it — so this
never surfaced as an error. It surfaced as cost: the chunker cut roughly twice
as often as configured, and each chunk became one LLM call.

## Phase 1 — MEASURED, with the exact splitter

`whole` is the tokenizer's count for the corpus. `per-word` is the sum over
`chunk_by_word` pieces counted in isolation — the **exact production splitter**,
not a regex approximation. Chunk counts and fills are at the default 8191-token
budget.

| corpus | whole | per-word | ratio | chunks (legacy) | chunks (fixed) | fullest chunk, legacy | fullest chunk, fixed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `english_text` (Paradise Lost) | 1,201 | 2,072 | 1.725 | 1 | 1 | 1,201 | 1,201 |
| `english_lists` | 277 | 447 | 1.614 | 1 | 1 | 277 | 277 |
| `python_code` | 509 | 842 | 1.654 | 1 | 1 | 509 | 509 |
| `chinese_text` | 494 | 494 | **1.000** | 1 | 1 | 494 | 494 |
| Alice in Wonderland | 36,958 | 65,797 | **1.780** | **9** | **5** | **4,669** | **8,188** |
| pooled | 39,439 | 69,652 | 1.766 | | | | |

Reading the Alice row: a chunk the configuration called 8191 tokens held
**4,669** real ones, and the document cost **9** LLM calls where **5** were
paid for. With span counting the fullest chunk holds **8,188 of 8,191**. The
ticket's estimate of "~4,600 real tokens" and "roughly twice the LLM calls" is
therefore **confirmed**, to within a few tokens.

`chinese_text` at exactly 1.000 is the informative control: continuous CJK has
no spaces, so `chunk_by_word` yields whole sentence-ending segments and there
are no word-boundary merges to lose. The over-count is a property of
space-delimited text, not of the tokenizer.

### Corrections to the numbers the ticket carried

The ticket reported two mutually inconsistent figures for Alice and said so.
Neither was produced by the real code path; both are superseded.

| | whole | per-word | ratio |
|---|---:|---:|---:|
| ticket, original write-up | 37,165 | 65,478 | 1.76 |
| ticket, revised (regex approximation) | 37,165 | 64,483 | 1.73 |
| **this measurement (exact splitter)** | **36,958** | **65,797** | **1.780** |

The `whole` difference is explained: 37,165 is the sum over the nine emitted
chunk *texts*, which loses the merges at the eight chunk boundaries; 36,958 is
the document counted once.

## Phase 2 — the fix

`TokenCountMode::Span` (the default) counts the accumulated span once, at emit.
`TokenCountMode::PerWord` keeps the old arithmetic and is opt-in, off, via
`COGNEE_LEGACY_PER_WORD_TOKEN_COUNT`. See
[configuration.md](../configuration.md#cognee_legacy_per_word_token_count--a-deliberate-divergence-from-python).

Three layers changed, each the same shape — and `chunk_by_row` already worked
this way before the change, which is where the shape comes from:

- **`chunk_by_sentence`** reports the count of the emitted slice.
- **`chunk_by_paragraph`** reports the count of the emitted slice.
- **`chunk_text`** reports the count of the emitted text, including the `" "`
  it inserts between batched paragraphs.

### Why the accumulator is still a sum

A span count cannot be maintained incrementally, and the overflow check needs a
*prospective* size — "would this chunk still fit if I took the next piece?".
Re-counting the whole accumulated span once per piece would be quadratic.

So the cheap per-word (or per-sentence) sum is retained as an **upper bound**
that decides when to look, and only an exact count of the accumulated span may
decide to **cut**. On a false alarm the accumulator is re-baselined to the exact
value, so the bound does not trip again on every subsequent piece. Each
re-baseline raises the exact floor, which makes the number of exact re-counts
logarithmic in the budget rather than linear in pieces.

The cost of that is one extra tokenizer pass: `Span` counts each emitted slice
*in addition to* the per-word counts that still drive the bound, roughly
doubling the tokenizer work. That is a few milliseconds per document, against
the LLM call per extra chunk the over-count was buying.

The first cut of the fix left the *paragraph* accumulator as a plain sum. That
is safe — it errs under the limit — but it cost ~1 token per sentence boundary,
which on short prose sentences is ~9% of the budget: the fullest Alice chunk
came to 7,889 rather than 8,188. The two-tier check was applied at that layer
too, which is the difference between those two numbers.

## Consequences the ticket flagged

1. **Rust diverges from Python by default.** VERIFIED against a fresh clone:
   `cognee/tasks/chunks/chunk_by_sentence.py` still calls
   `get_word_size(word)` → `embedding_engine.tokenizer.count_tokens(word)`
   once per word. The divergence is deliberate and reversible with the env
   flag. **The matching Python change still needs an owner.**

2. **Cross-SDK tolerances do not breach.** MEASURED rather than assumed: the
   three `e2e-cross-sdk/test_data/` fixtures are 166, 883 and 942 cl100k tokens,
   and the harness sets no chunk-size override, so each produces **exactly one
   chunk under either mode** at the 8191 default. The corpora never reach a
   chunk boundary, so `COUNT_TOLERANCE` (0.5) and the node-type Jaccard floor
   (0.3) cannot be moved by this change. No re-baselining was needed.

3. **The no-tokenizer fallback differs, and is left alone.** Python returns `1`
   per word when no tokenizer is configured; Rust falls back to `WordCounter`,
   which counts whitespace-delimited words. They agree on ordinary prose. They
   differ on a piece holding no non-whitespace character — a run of consecutive
   spaces — which Python counts as `1` and Rust as `0`.

## What this does not measure

The end-to-end latency claim. Fewer chunks means fewer LLM calls, but the
[issue 212 baseline](https://github.com/topoteretes/cognee-rs/issues/212) runs
are `n=1` each and the extraction runaway is probabilistic (~5.5% of calls), so
a single faster run would not be evidence either way. Nothing here was timed
against a live provider.
