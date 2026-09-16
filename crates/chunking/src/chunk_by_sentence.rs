//! Sentence-level text chunker.
//!
//! Aggregates word-level chunks into sentences, tracking paragraph boundaries
//! and token counts.
//!
//! Port of Python `cognee.tasks.chunks.chunk_by_sentence`.

use uuid::Uuid;

use crate::chunk_by_word::{WordType, chunk_by_word};
use crate::cut_type::CutType;
use crate::token_counter::{TokenCountMode, TokenCounter};

/// A sentence-level chunk with metadata. Borrows text from the input.
#[derive(Debug, Clone)]
pub struct SentenceChunk<'a> {
    /// Unique paragraph identifier. Changes on paragraph boundaries.
    pub paragraph_id: Uuid,
    /// The sentence text, borrowed from the input.
    pub text: &'a str,
    /// Token count of the sentence (via TokenCounter).
    pub size: usize,
    /// How the sentence boundary was determined.
    pub cut_type: CutType,
}

fn word_type_to_cut_type(wt: WordType) -> CutType {
    match wt {
        WordType::ParagraphEnd => CutType::ParagraphEnd,
        WordType::SentenceEnd => CutType::SentenceEnd,
        WordType::Word => CutType::Word,
    }
}

/// Computes the byte offset of a `&str` slice relative to the start of `base`.
fn offset_in(base: &str, slice: &str) -> usize {
    slice.as_ptr() as usize - base.as_ptr() as usize
}

/// Splits text into sentences based on word-level tokenization.
///
/// - `data`: the input text
/// - `maximum_size`: optional max token count per sentence. If a sentence would
///   exceed this, it is yielded early and the overflowing word starts a new one.
/// - `counter`: token counter implementation
/// - `mode`: how a sentence's size is measured — see [`TokenCountMode`]. Under
///   the default [`TokenCountMode::Span`] the reported `size` is the counter's
///   verdict on the emitted slice itself, so it is exact by construction.
///
/// # Cost
///
/// The overflow check needs a *prospective* size — "would this sentence still
/// fit if I took the next word?" — which a span count cannot give
/// incrementally, and re-counting the accumulated span once per word would be
/// quadratic. So the per-word counts are still taken, as a cheap upper bound
/// that decides when to *look*; only an exact re-count of the accumulated span
/// may decide to *cut*, and a false alarm re-baselines the bound to that exact
/// value. Each re-baseline raises the exact floor, which makes the number of
/// re-counts per oversized sentence logarithmic in `maximum_size` rather than
/// linear in words.
///
/// [`TokenCountMode::Span`] therefore tokenizes each emitted sentence *in
/// addition to* each word, roughly doubling the tokenizer work of the per-word
/// path it replaces. That is a few milliseconds per document, against the LLM
/// call per extra chunk that the over-count was buying.
#[allow(
    clippy::expect_used,
    reason = "sentence_start invariants are upheld by the is_some() guard and the explicit set above each emit branch"
)]
pub fn chunk_by_sentence<'a, C: TokenCounter>(
    data: &'a str,
    maximum_size: Option<usize>,
    counter: &C,
    mode: TokenCountMode,
) -> Vec<SentenceChunk<'a>> {
    let words = chunk_by_word(data);
    let mut result = Vec::new();
    let mut paragraph_id = Uuid::new_v4();
    // Under `PerWord` this is the reported size. Under `Span` it is only an
    // upper bound used to decide when to spend an exact re-count; the reported
    // size is always counted from the emitted slice.
    let mut sentence_size: usize = 0;
    let mut word_type_state = WordType::Word;
    // Track the byte range of the current sentence in `data`.
    let mut sentence_start: Option<usize> = None;
    let mut sentence_end: usize = 0;

    // Size of the slice about to be emitted, in whichever unit `mode` reports.
    let emitted_size = |start: usize, end: usize, accumulated: usize| match mode {
        TokenCountMode::PerWord => accumulated,
        TokenCountMode::Span => counter.count_tokens(&data[start..end]),
    };

    for word_chunk in &words {
        let word = word_chunk.text;
        let word_type = word_chunk.word_type;
        let word_size = counter.count_tokens(word);

        let word_start_byte = offset_in(data, word);
        let word_end_byte = word_start_byte + word.len();

        // Update word_type_state: for sentence/paragraph ends, take directly.
        // For words, only update if the word contains alphabetic characters.
        match word_type {
            WordType::ParagraphEnd | WordType::SentenceEnd => {
                word_type_state = word_type;
            }
            WordType::Word => {
                if word.chars().any(|c| c.is_alphabetic()) {
                    word_type_state = word_type;
                }
            }
        }

        // What the accumulator would read once this word is taken in.
        let mut next_size = sentence_size + word_size;

        // Check overflow. Summed per-word counts over-estimate the span they
        // compose, so under `Span` a hit here is only a suspicion: confirm it
        // against the real span before cutting, and re-baseline the accumulator
        // on a false alarm so the bound does not trip again on every
        // subsequent word.
        let mut overflows = maximum_size.is_some_and(|max| next_size > max);
        if overflows
            && mode == TokenCountMode::Span
            && let Some(start) = sentence_start
            && let Some(max) = maximum_size
        {
            let exact_with_word = counter.count_tokens(&data[start..word_end_byte]);
            if exact_with_word <= max {
                overflows = false;
                next_size = exact_with_word;
            }
        }

        if overflows && sentence_start.is_some() {
            let start = sentence_start.expect("sentence_start is Some because the guard sentence_start.is_some() was checked before this branch");
            result.push(SentenceChunk {
                paragraph_id,
                text: &data[start..sentence_end],
                size: emitted_size(start, sentence_end, sentence_size),
                cut_type: word_type_to_cut_type(word_type_state),
            });
            sentence_start = Some(word_start_byte);
            sentence_end = word_end_byte;
            sentence_size = word_size;
            continue;
        }

        if matches!(word_type, WordType::ParagraphEnd | WordType::SentenceEnd) {
            if sentence_start.is_none() {
                sentence_start = Some(word_start_byte);
            }
            sentence_end = word_end_byte;
            sentence_size = next_size;

            if word_type == WordType::ParagraphEnd {
                paragraph_id = Uuid::new_v4();
            }

            let start = sentence_start
                .expect("sentence_start is Some because it was just set above if it was None");
            result.push(SentenceChunk {
                paragraph_id,
                text: &data[start..sentence_end],
                size: emitted_size(start, sentence_end, sentence_size),
                cut_type: word_type_to_cut_type(word_type_state),
            });
            sentence_start = None;
            sentence_size = 0;
        } else {
            if sentence_start.is_none() {
                sentence_start = Some(word_start_byte);
            }
            sentence_end = word_end_byte;
            sentence_size = next_size;
        }
    }

    if let Some(start) = sentence_start {
        let cut_type = if word_type_state == WordType::Word {
            CutType::SentenceCut
        } else {
            word_type_to_cut_type(word_type_state)
        };
        result.push(SentenceChunk {
            paragraph_id,
            text: &data[start..sentence_end],
            size: emitted_size(start, sentence_end, sentence_size),
            cut_type,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_counter::WordCounter;

    #[test]
    fn empty_input() {
        let chunks = chunk_by_sentence("", None, &WordCounter, TokenCountMode::Span);
        assert!(chunks.is_empty());
    }

    #[test]
    fn single_sentence() {
        let chunks = chunk_by_sentence("Hello world.", None, &WordCounter, TokenCountMode::Span);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world.");
        assert_eq!(chunks[0].size, 2);
        assert_eq!(chunks[0].cut_type, CutType::SentenceEnd);
    }

    #[test]
    fn two_sentences_same_paragraph() {
        let chunks = chunk_by_sentence(
            "Hello world. Foo bar.",
            None,
            &WordCounter,
            TokenCountMode::Span,
        );
        assert_eq!(chunks.len(), 2);
        // Same paragraph_id for both
        assert_eq!(chunks[0].paragraph_id, chunks[1].paragraph_id);
    }

    #[test]
    fn paragraph_boundary_new_id() {
        // In Python, paragraph_id is updated on paragraph_end BEFORE yielding,
        // so the sentence with paragraph_end gets the NEW id, and subsequent
        // sentences share that id until the next paragraph_end.
        // Two separate paragraphs should have different IDs:
        let chunks = chunk_by_sentence(
            "First paragraph.\nSecond paragraph.\nThird.",
            None,
            &WordCounter,
            TokenCountMode::Span,
        );
        assert_eq!(chunks.len(), 3);
        // First paragraph_end triggers new id for chunks[0]
        // Second paragraph_end triggers another new id for chunks[1]
        // chunks[0] and chunks[1] should differ (different paragraph_ends)
        assert_ne!(chunks[0].paragraph_id, chunks[1].paragraph_id);
    }

    #[test]
    fn sentence_cut_no_punctuation() {
        let chunks = chunk_by_sentence("Hello world", None, &WordCounter, TokenCountMode::Span);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].cut_type, CutType::SentenceCut);
    }

    #[test]
    fn maximum_size_overflow() {
        // max 2 words per sentence
        let chunks = chunk_by_sentence(
            "one two three four",
            Some(2),
            &WordCounter,
            TokenCountMode::Span,
        );
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, "one two ");
        assert_eq!(chunks[0].size, 2);
        assert_eq!(chunks[1].text, "three four");
        assert_eq!(chunks[1].size, 2);
    }

    #[test]
    fn token_counting_matches_word_count() {
        let chunks = chunk_by_sentence(
            "This is a test sentence.",
            None,
            &WordCounter,
            TokenCountMode::Span,
        );
        assert_eq!(chunks[0].size, 5);
    }

    #[test]
    fn isomorphism_parametrized() {
        use crate::test_inputs::{EMPTY, ENGLISH_LISTS, ENGLISH_TEXT, PYTHON_CODE};

        let texts = [
            ("english_text", ENGLISH_TEXT),
            ("english_lists", ENGLISH_LISTS),
            ("python_code", PYTHON_CODE),
            ("empty", EMPTY),
        ];
        let max_sizes: [Option<usize>; 3] = [None, Some(16), Some(64)];
        let counter = WordCounter;

        for &(name, text) in &texts {
            for max in max_sizes {
                let chunks = chunk_by_sentence(text, max, &counter, TokenCountMode::Span);
                let reconstructed: String = chunks.iter().map(|c| c.text).collect();
                assert_eq!(
                    reconstructed, text,
                    "isomorphism failed for ('{name}', max={max:?})"
                );
            }
        }
    }

    #[test]
    fn token_count_within_max_length() {
        use crate::test_inputs::{EMPTY, ENGLISH_LISTS, ENGLISH_TEXT, PYTHON_CODE};

        let texts = [
            ("english_text", ENGLISH_TEXT),
            ("english_lists", ENGLISH_LISTS),
            ("python_code", PYTHON_CODE),
            ("empty", EMPTY),
        ];
        let counter = WordCounter;

        for &(name, text) in &texts {
            for max in [16_usize, 64] {
                let chunks = chunk_by_sentence(text, Some(max), &counter, TokenCountMode::Span);
                for (i, chunk) in chunks.iter().enumerate() {
                    assert!(
                        chunk.size <= max,
                        "chunk {i} in ('{name}', max={max}) has size {} > {max}",
                        chunk.size
                    );
                }
            }
        }
    }

    #[test]
    fn chinese_text_no_panic() {
        use crate::test_inputs::CHINESE_TEXT;

        let counter = WordCounter;
        let chunks = chunk_by_sentence(CHINESE_TEXT, Some(16), &counter, TokenCountMode::Span);
        assert!(
            !chunks.is_empty(),
            "Chinese text should produce at least one chunk"
        );
    }
}
