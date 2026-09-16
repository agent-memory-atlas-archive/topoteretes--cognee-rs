//! Measures the per-word token over-count that SDK-632 is about.
//!
//! `chunk_by_sentence` historically asked the tokenizer for the size of every
//! word *in isolation*, which destroys BPE's leading-space merges: cl100k
//! encodes `" the"` as one token but `"the"` + `" "` as two. The sum over words
//! is therefore an over-estimate of the span they compose, and every chunk the
//! configuration sizes at N tokens holds materially fewer than N.
//!
//! This example reproduces that with the **exact** splitter the production path
//! uses (`chunk_by_word`), not a regex approximation, so the numbers can be
//! checked rather than taken on trust.
//!
//! ```text
//! cargo run -p cognee-chunking --features tiktoken \
//!     --example measure_token_overcount -- [EXTRA_CORPUS_PATH...]
//! ```
//!
//! The four committed corpora under `src/test_data/` are always measured; any
//! paths given on the command line are measured too (that is how a large prose
//! corpus such as Alice in Wonderland is fed in without committing it).

use cognee_chunking::chunk_by_word::chunk_by_word;
use cognee_chunking::text_chunker::chunk_text;
use cognee_chunking::token_counter::{TikTokenCounter, TokenCountMode, TokenCounter};
use uuid::Uuid;

/// The chunk budget an OpenAI-family embedder gets by default.
const BUDGET: usize = 8191;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let counter = TikTokenCounter::cl100k_base()?;

    let mut corpora: Vec<(String, String)> = Vec::new();
    for name in [
        "english_text",
        "english_lists",
        "python_code",
        "chinese_text",
    ] {
        let path =
            concat!(env!("CARGO_MANIFEST_DIR"), "/src/test_data/").to_string() + name + ".txt";
        corpora.push((name.to_string(), std::fs::read_to_string(&path)?));
    }
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path)?;
        corpora.push((path, text));
    }

    println!(
        "{:<28} {:>9} {:>9} {:>7} {:>8} {:>8} {:>9} {:>9}",
        "corpus", "whole", "per-word", "ratio", "n(word)", "n(span)", "max tok/w", "max tok/s"
    );
    println!("{}", "-".repeat(96));

    let mut pooled_whole = 0usize;
    let mut pooled_per_word = 0usize;

    for (name, text) in &corpora {
        // (1) The truth: the tokenizer's count for the whole corpus.
        let whole = counter.count_tokens(text);

        // (2) What the legacy path accumulates: every word, in isolation,
        //     through the *exact* production splitter.
        let per_word: usize = chunk_by_word(text)
            .iter()
            .map(|w| counter.count_tokens(w.text))
            .sum();

        pooled_whole += whole;
        pooled_per_word += per_word;

        // (3) What that costs end to end: how much real text a chunk the
        //     configuration calls BUDGET tokens actually holds.
        let doc = Uuid::nil();
        let legacy = chunk_text(doc, text, BUDGET, &counter, TokenCountMode::PerWord);
        let fixed = chunk_text(doc, text, BUDGET, &counter, TokenCountMode::Span);
        let max_real = |chunks: &[cognee_models::DocumentChunk]| {
            chunks
                .iter()
                .map(|c| counter.count_tokens(&c.text))
                .max()
                .unwrap_or(0)
        };

        println!(
            "{:<28} {:>9} {:>9} {:>7.3} {:>8} {:>8} {:>9} {:>9}",
            truncate(name, 28),
            whole,
            per_word,
            per_word as f64 / whole.max(1) as f64,
            legacy.len(),
            fixed.len(),
            max_real(&legacy),
            max_real(&fixed),
        );
    }

    println!("{}", "-".repeat(96));
    println!(
        "{:<28} {:>9} {:>9} {:>7.3}",
        "POOLED",
        pooled_whole,
        pooled_per_word,
        pooled_per_word as f64 / pooled_whole.max(1) as f64,
    );
    println!(
        "\nwhole      = cl100k tokens in the corpus, counted once\n\
         per-word   = sum of cl100k counts of each chunk_by_word piece, in isolation\n\
         ratio      = per-word / whole; 1.000 means the legacy accumulation was already exact\n\
         n(word)    = chunks produced at a {BUDGET}-token budget under TokenCountMode::PerWord\n\
         n(span)    = the same under TokenCountMode::Span\n\
         max tok/*  = real cl100k tokens in the largest chunk each mode produced"
    );

    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let tail: String = s.chars().skip(s.chars().count() - (n - 1)).collect();
    format!("…{tail}")
}
