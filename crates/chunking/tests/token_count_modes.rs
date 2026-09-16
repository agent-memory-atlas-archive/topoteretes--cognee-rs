//! Pins the two [`TokenCountMode`] states against real cl100k counts (SDK-632).
//!
//! The defect these tests exist for: `chunk_by_sentence` used to ask the
//! tokenizer for the size of every word *in isolation* and sum the results.
//! Sub-word tokenizers are not additive over a partition — cl100k encodes
//! `" the"` as one token but `"the"` and `" "` as two — so the sum over-states
//! the span the words compose, MEASURED at 1.6-2.1x across the committed
//! corpora and 1.78x over Alice in Wonderland. A chunk the configuration sized
//! at 8191 tokens therefore held ~4,600, and a run made roughly twice the LLM
//! calls it was configured for.
//!
//! Every expected value below is the tokenizer's own verdict, hard-coded, so a
//! regression in either direction is visible rather than merely self-consistent.

#![cfg(feature = "tiktoken")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]

use cognee_chunking::chunk_by_paragraph::chunk_by_paragraph;
use cognee_chunking::chunk_by_sentence::chunk_by_sentence;
use cognee_chunking::text_chunker::chunk_text;
use cognee_chunking::token_counter::{TikTokenCounter, TokenCountMode, TokenCounter};
use uuid::Uuid;

/// One ordinary English sentence. cl100k encodes it as 10 tokens; counted word
/// by word through `chunk_by_word` it comes to 19 — the leading-space merges
/// that `" quick"`, `" brown"`, … would have got are all lost at the cuts. That
/// the second number is the larger one is the whole defect, in one line.
const PANGRAM: &str = "The quick brown fox jumps over the lazy dog.";
const PANGRAM_SPAN_TOKENS: usize = 10;
const PANGRAM_PER_WORD_TOKENS: usize = 19;

fn counter() -> TikTokenCounter {
    TikTokenCounter::cl100k_base().expect("cl100k_base ships with tiktoken-rs")
}

#[test]
fn span_mode_reports_the_tokenizers_own_count() {
    let counter = counter();
    assert_eq!(
        counter.count_tokens(PANGRAM),
        PANGRAM_SPAN_TOKENS,
        "the pinned expectation must be what cl100k actually says",
    );

    let chunks = chunk_by_sentence(PANGRAM, None, &counter, TokenCountMode::Span);
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0].size, PANGRAM_SPAN_TOKENS,
        "Span mode must report the count of the emitted slice itself",
    );
}

#[test]
fn per_word_mode_reproduces_the_legacy_over_count() {
    let counter = counter();
    let chunks = chunk_by_sentence(PANGRAM, None, &counter, TokenCountMode::PerWord);
    assert_eq!(chunks.len(), 1);
    assert_eq!(
        chunks[0].size, PANGRAM_PER_WORD_TOKENS,
        "PerWord mode must still sum the words in isolation — it is what Python does",
    );
}

/// The headline claim: a chunk the configuration calls N tokens must hold
/// close to N real tokens, not ~57% of N.
#[test]
fn a_budget_buys_the_tokens_it_promises() {
    let counter = counter();
    // ~46 repetitions of a 10-token sentence: enough to cross a 128-token
    // budget several times under either mode.
    let text = PANGRAM.to_string() + &format!(" {PANGRAM}").repeat(45);
    let budget = 128;
    let doc = Uuid::nil();

    let span = chunk_text(doc, &text, budget, &counter, TokenCountMode::Span);
    let legacy = chunk_text(doc, &text, budget, &counter, TokenCountMode::PerWord);

    // Measured on the fullest chunk, not the mean: the last chunk of any run is
    // a remainder, and with only a handful of chunks it would dominate a mean.
    let fullest = |chunks: &[cognee_models::DocumentChunk]| -> usize {
        chunks
            .iter()
            .map(|c| counter.count_tokens(&c.text))
            .max()
            .unwrap_or(0)
    };

    // Span leaves less than one sentence of the budget unspent — sentences are
    // indivisible here, so that is the whole of the achievable headroom. (On
    // real prose, whose sentences are far shorter than 1/12th of the budget,
    // this lands within 3 tokens of 8191; see the `measure_token_overcount`
    // example.) The legacy sum leaves ~40% of the budget unused.
    assert!(
        budget - fullest(&span) < PANGRAM_SPAN_TOKENS,
        "Span left {} of {budget} tokens unspent, more than the one indivisible \
         sentence of headroom ({} chunks)",
        budget - fullest(&span),
        span.len(),
    );
    assert!(
        (fullest(&legacy) as f64) < 0.70 * budget as f64,
        "PerWord should under-fill the budget, fullest chunk held {} of {budget} ({} chunks)",
        fullest(&legacy),
        legacy.len(),
    );
    assert!(
        legacy.len() > span.len(),
        "the over-count must cost extra chunks: {} legacy vs {} span",
        legacy.len(),
        span.len(),
    );
}

/// The budget is a ceiling, and under `Span` it is a ceiling on *real* tokens.
///
/// This is the invariant the two-tier overflow check has to preserve: the cheap
/// per-word upper bound decides when to look, but only an exact count of the
/// accumulated span may decide to cut.
#[test]
fn span_mode_never_emits_a_chunk_over_the_budget() {
    let counter = counter();
    let corpora: [(&str, String); 3] = [
        (
            "prose",
            PANGRAM.to_string() + &format!(" {PANGRAM}").repeat(80),
        ),
        // No sentence-ending punctuation at all: one unbroken run, so every cut
        // goes through the overflow branch rather than a sentence boundary.
        (
            "unbroken run",
            "alpha beta gamma delta epsilon ".repeat(120),
        ),
        // Dense punctuation, where words tokenize to several tokens each.
        ("dense", "a=1;b=2;c=3;d=4;e=5;".repeat(120)),
    ];

    for (name, text) in &corpora {
        for budget in [16_usize, 64, 512] {
            for chunk in chunk_by_paragraph(text, budget, true, &counter, TokenCountMode::Span) {
                let real = counter.count_tokens(chunk.text);
                assert_eq!(
                    chunk.chunk_size, real,
                    "('{name}', budget={budget}) reported {} for a slice of {real} tokens",
                    chunk.chunk_size,
                );
                assert!(
                    real <= budget,
                    "('{name}', budget={budget}) emitted a chunk of {real} real tokens",
                );
            }
        }
    }
}

/// Chunking must stay isomorphic: the mode changes only the arithmetic, never
/// where the text is cut apart at the sentence level.
#[test]
fn both_modes_preserve_the_input_text() {
    let counter = counter();
    let text = "First paragraph.\nSecond one is longer, with a clause; and a question? Yes!\n\nTrailing run with no terminator";
    for mode in [TokenCountMode::Span, TokenCountMode::PerWord] {
        for max in [None, Some(4_usize), Some(32)] {
            let reconstructed: String = chunk_by_sentence(text, max, &counter, mode)
                .iter()
                .map(|c| c.text)
                .collect();
            assert_eq!(
                reconstructed, text,
                "isomorphism failed for {mode:?}/{max:?}"
            );
        }
    }
}

/// A word that is on its own larger than the budget cannot be split by this
/// layer, so it is emitted oversized rather than dropped or looped on.
#[test]
fn a_single_oversized_word_is_emitted_not_dropped() {
    let counter = counter();
    let text = format!("short. {} short.", "z".repeat(400));
    let chunks = chunk_by_sentence(&text, Some(8), &counter, TokenCountMode::Span);
    let reconstructed: String = chunks.iter().map(|c| c.text).collect();
    assert_eq!(reconstructed, text);
    assert!(
        chunks.iter().any(|c| c.size > 8),
        "the oversized run must survive as an oversized chunk",
    );
}
