use std::collections::HashMap;
use std::ops::Bound;

use super::{
    DocumentEvaluation, SingleDocument, SingleDocumentEvaluationContext, SingleDocumentEvaluator,
    SingleDocumentPreparer, SingleDocumentTermInfo,
};
use crate::core::json_utils::JsonTermWriter;
use crate::fieldnorm::FieldNormReader;
use crate::query::{
    AllQuery, Bm25StatisticsProvider, BooleanQuery, BoostQuery, ConstScoreQuery, EmptyQuery,
    EnableScoring, FuzzyTermQuery, Occur, PhrasePrefixQuery, PhraseQuery, Query, RegexQuery,
    TermQuery, TermSetQuery,
};
use crate::schema::{
    Field, IndexRecordOption, OwnedValue, Schema, TextFieldIndexing, TextOptions, Type, INDEXED,
    STORED, STRING, TEXT,
};
use crate::tokenizer::{PreTokenizedString, Token, TokenizerManager, WhitespaceTokenizer};
use crate::{DocSet, Index, IndexWriter, TantivyDocument, TantivyError, Term, TERMINATED};

struct TestStatistics {
    total_docs: u64,
    total_tokens: u64,
    doc_freqs: HashMap<Term, u64>,
}

impl Bm25StatisticsProvider for TestStatistics {
    fn total_num_tokens(&self, _field: Field) -> crate::Result<u64> {
        Ok(self.total_tokens)
    }

    fn total_num_docs(&self) -> crate::Result<u64> {
        Ok(self.total_docs)
    }

    fn doc_freq(&self, term: &Term) -> crate::Result<u64> {
        Ok(*self.doc_freqs.get(term).unwrap_or(&0))
    }
}

#[derive(Default)]
struct TestDocument {
    term_freqs: HashMap<Term, u32>,
    positions: HashMap<Term, Vec<u32>>,
    fieldnorms: HashMap<Field, u8>,
}

impl SingleDocument for TestDocument {
    fn term_info(&self, term: &Term) -> Option<SingleDocumentTermInfo<'_>> {
        self.term_freqs
            .get(term)
            .copied()
            .map(|term_freq| SingleDocumentTermInfo {
                term_freq,
                positions: self.positions.get(term).map(Vec::as_slice),
            })
    }

    fn fieldnorm_id(&self, field: Field) -> Option<u8> {
        self.fieldnorms.get(&field).copied()
    }

    fn visit_terms(
        &self,
        field: Field,
        range: (Bound<&Term>, Bound<&Term>),
        visitor: &mut dyn FnMut(&Term, SingleDocumentTermInfo<'_>) -> bool,
    ) {
        let mut terms = self
            .term_freqs
            .iter()
            .filter(|(term, _)| term.field() == field)
            .collect::<Vec<_>>();
        terms.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        for (term, &term_freq) in terms {
            let after_start = match range.0 {
                Bound::Included(start) => term >= start,
                Bound::Excluded(start) => term > start,
                Bound::Unbounded => true,
            };
            if !after_start {
                continue;
            }
            let before_end = match range.1 {
                Bound::Included(end) => term <= end,
                Bound::Excluded(end) => term < end,
                Bound::Unbounded => true,
            };
            if !before_end {
                break;
            }
            if !visitor(
                term,
                SingleDocumentTermInfo {
                    term_freq,
                    positions: self.positions.get(term).map(Vec::as_slice),
                },
            ) {
                break;
            }
        }
    }
}

fn preparer_for_fields(
    schema: &Schema,
    tokenizer_manager: &TokenizerManager,
    fields: &[Field],
) -> crate::Result<SingleDocumentPreparer> {
    SingleDocumentPreparer::for_fields(schema, tokenizer_manager, fields)
}

fn text_schema() -> (Schema, Field) {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_text_field("body", TEXT);
    (schema_builder.build(), field)
}

#[test]
fn empty_boolean_matches_no_documents() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let query = BooleanQuery::new(Vec::new());

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "body"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
    let scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), TERMINATED);

    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        evaluator.evaluate(&TestDocument::default())?,
        DocumentEvaluation::NoMatch
    );

    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &searcher),
    )?;
    assert_eq!(
        evaluator.evaluate(&TestDocument::default())?,
        DocumentEvaluation::NoMatch
    );
    Ok(())
}

#[test]
fn all_and_empty_query_single_document_behavior() -> crate::Result<()> {
    let (schema, _field) = text_schema();
    let statistics = TestStatistics {
        total_docs: 1,
        total_tokens: 1,
        doc_freqs: HashMap::new(),
    };
    let document = TestDocument::default();

    for context in [
        SingleDocumentEvaluationContext::without_scoring(&schema),
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    ] {
        let mut all_evaluator = AllQuery.single_document_evaluator(context)?;
        assert_eq!(
            all_evaluator.evaluate(&document)?,
            DocumentEvaluation::Match(1.0)
        );

        let mut empty_evaluator = EmptyQuery.single_document_evaluator(context)?;
        assert_eq!(
            empty_evaluator.evaluate(&document)?,
            DocumentEvaluation::NoMatch
        );
    }
    Ok(())
}

#[test]
fn single_document_term_score_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term = Term::from_field_text(field, "rust");
    let query = TermQuery::new(term.clone(), IndexRecordOption::WithFreqs);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 40,
        doc_freqs: HashMap::from([(term.clone(), 4)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "rust rust other"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(term, 2);
    document
        .fieldnorms
        .insert(field, FieldNormReader::fieldnorm_to_id(3));
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&document)? else {
        panic!("document should match");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn basic_term_scoring_treats_term_freq_as_one() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term = Term::from_field_text(field, "rust");
    let query = TermQuery::new(term.clone(), IndexRecordOption::Basic);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 40,
        doc_freqs: HashMap::from([(term.clone(), 4)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "rust rust rust other"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(term.clone(), 3);
    document
        .fieldnorms
        .insert(field, FieldNormReader::fieldnorm_to_id(4));
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(reported_freq_score) = evaluator.evaluate(&document)? else {
        panic!("document should match");
    };

    document.term_freqs.insert(term, 1);
    let DocumentEvaluation::Match(unit_freq_score) = evaluator.evaluate(&document)? else {
        panic!("document should match");
    };
    assert_eq!(reported_freq_score, unit_freq_score);
    assert!((reported_freq_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn term_without_scoring_reports_match_and_no_match() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term = Term::from_field_text(field, "rust");
    let query = TermQuery::new(term.clone(), IndexRecordOption::WithFreqs);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;

    assert_eq!(
        evaluator.evaluate(&TestDocument::default())?,
        DocumentEvaluation::NoMatch
    );

    let mut matching_document = TestDocument::default();
    matching_document.term_freqs.insert(term, 2);
    assert_eq!(
        evaluator.evaluate(&matching_document)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn single_document_fuzzy_evaluation_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "rust search"))?;
    writer.add_document(doc!(field => "database"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();

    let mut document = TantivyDocument::new();
    document.add_text(field, "rust search");
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;
    let prepared = preparer.prepare(&document)?;

    let queries = [
        FuzzyTermQuery::new_prefix(Term::from_field_text(field, "rus"), 0, true),
        FuzzyTermQuery::new(Term::from_field_text(field, "ruse"), 1, true),
        FuzzyTermQuery::new(Term::from_field_text(field, "rsut"), 1, true),
    ];
    for query in queries {
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
        assert_eq!(scorer.doc(), 0);
        let expected_score = scorer.score();

        let mut evaluator = query.single_document_evaluator(
            SingleDocumentEvaluationContext::with_scoring(&schema, &searcher),
        )?;
        let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&prepared)? else {
            panic!("fuzzy query should match");
        };
        assert_eq!(actual_score, expected_score);
        assert_eq!(evaluator.required_fields(), Some([field].as_slice()));
    }

    let query = FuzzyTermQuery::new_prefix(Term::from_field_text(field, "sql"), 0, true);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(evaluator.evaluate(&prepared)?, DocumentEvaluation::NoMatch);
    Ok(())
}

#[test]
fn single_document_fuzzy_json_path_is_filtered() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let attributes = schema_builder.add_json_field("attributes", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let document = doc!(attributes => serde_json::json!({"a": "japan"}));
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[attributes])?;
    let prepared = preparer.prepare(&document)?;

    let json_term = |path: &str, text: &str| {
        let mut term = Term::with_type_and_field(Type::Json, attributes);
        let mut writer = JsonTermWriter::wrap(&mut term, false);
        writer.push_path_segment(path);
        writer.set_str(text);
        drop(writer);
        term
    };

    // The extra `a` in the JSON path is within the fuzzy distance, but paths must be exact.
    let wrong_path_query = FuzzyTermQuery::new(json_term("aa", "japan"), 2, true);
    let mut wrong_path_evaluator = wrong_path_query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        wrong_path_evaluator.evaluate(&prepared)?,
        DocumentEvaluation::NoMatch
    );

    let matching_query = FuzzyTermQuery::new(json_term("a", "japon"), 1, true);
    let mut matching_evaluator = matching_query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        matching_evaluator.evaluate(&prepared)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn single_document_phrase_prefix_score_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "rust database systems"))?;
    writer.add_document(doc!(field => "rust database storage"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let query = PhrasePrefixQuery::new(vec![
        Term::from_field_text(field, "rust"),
        Term::from_field_text(field, "database"),
        Term::from_field_text(field, "sys"),
    ]);

    let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut matching_document = TantivyDocument::new();
    matching_document.add_text(field, "rust database systems");
    let mut non_matching_document = TantivyDocument::new();
    non_matching_document.add_text(field, "rust database storage");
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;
    let matching_prepared = preparer.prepare(&matching_document)?;
    let non_matching_prepared = preparer.prepare(&non_matching_document)?;
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &searcher),
    )?;

    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&matching_prepared)? else {
        panic!("phrase prefix query should match");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    assert_eq!(
        evaluator.evaluate(&non_matching_prepared)?,
        DocumentEvaluation::NoMatch
    );
    assert_eq!(evaluator.required_fields(), Some([field].as_slice()));
    Ok(())
}

#[test]
fn single_document_phrase_prefix_with_non_adjacent_offsets_matches_segment_scorer(
) -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "a x b cat"))?;
    writer.add_document(doc!(field => "d x dog"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();

    let term = |text| Term::from_field_text(field, text);
    let cases = [
        // Preserve a gap between the fixed terms while keeping the prefix immediately after the
        // greatest fixed-term offset, as the segment scorer's multi-fixed-term path expects.
        (
            PhrasePrefixQuery::new_with_offset(vec![
                (0, term("a")),
                (2, term("b")),
                (3, term("c")),
            ]),
            0,
            "a x b cat",
            "a b cat",
        ),
        // The segment scorer's single-fixed-term path aligns directly to the explicit prefix
        // offset, so exercise a gap between that fixed term and the prefix as well.
        (
            PhrasePrefixQuery::new_with_offset(vec![(0, term("d")), (2, term("do"))]),
            1,
            "d x dog",
            "d dog",
        ),
    ];
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;

    for (query, expected_doc, matching_text, non_matching_text) in cases {
        let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
        let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
        assert_eq!(scorer.doc(), expected_doc);
        let expected_score = scorer.score();
        assert_eq!(scorer.advance(), TERMINATED);

        let mut matching_document = TantivyDocument::new();
        matching_document.add_text(field, matching_text);
        let matching_prepared = preparer.prepare(&matching_document)?;
        let mut non_matching_document = TantivyDocument::new();
        non_matching_document.add_text(field, non_matching_text);
        let non_matching_prepared = preparer.prepare(&non_matching_document)?;
        let mut evaluator = query.single_document_evaluator(
            SingleDocumentEvaluationContext::with_scoring(&schema, &searcher),
        )?;

        let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&matching_prepared)?
        else {
            panic!("phrase prefix query with non-adjacent offsets should match");
        };
        assert!((actual_score - expected_score).abs() <= 1e-6);
        assert_eq!(
            evaluator.evaluate(&non_matching_prepared)?,
            DocumentEvaluation::NoMatch
        );
    }
    Ok(())
}

#[test]
fn single_document_phrase_prefix_honors_document_expansion_order() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let mut document = TantivyDocument::new();
    document.add_text(field, "x cb y x ca");
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;
    let prepared = preparer.prepare(&document)?;

    let mut query = PhrasePrefixQuery::new(vec![
        Term::from_field_text(field, "x"),
        Term::from_field_text(field, "c"),
    ]);
    query.set_max_expansions(1);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        evaluator.evaluate(&prepared)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn single_document_phrase_prefix_bounds_term_visit_to_prefix() -> crate::Result<()> {
    struct RangeCheckingDocument {
        field: Field,
        prefix: Term,
        end: Term,
        matching_term: Term,
    }

    impl SingleDocument for RangeCheckingDocument {
        fn term_info(&self, _term: &Term) -> Option<SingleDocumentTermInfo<'_>> {
            None
        }

        fn fieldnorm_id(&self, _field: Field) -> Option<u8> {
            None
        }

        fn visit_terms(
            &self,
            field: Field,
            range: (Bound<&Term>, Bound<&Term>),
            visitor: &mut dyn FnMut(&Term, SingleDocumentTermInfo<'_>) -> bool,
        ) {
            assert_eq!(field, self.field);
            assert!(matches!(range.0, Bound::Included(term) if term == &self.prefix));
            assert!(matches!(range.1, Bound::Excluded(term) if term == &self.end));
            visitor(
                &self.matching_term,
                SingleDocumentTermInfo {
                    term_freq: 1,
                    positions: None,
                },
            );
        }
    }

    let (schema, field) = text_schema();
    let prefix = Term::from_field_text(field, "ca");
    let document = RangeCheckingDocument {
        field,
        prefix: prefix.clone(),
        end: Term::from_field_text(field, "cb"),
        matching_term: Term::from_field_text(field, "cable"),
    };
    let query = PhrasePrefixQuery::new(vec![prefix]);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        evaluator.evaluate(&document)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "strictly ascending order")]
fn single_document_phrase_prefix_debug_asserts_visit_terms_order() {
    struct UnsortedTermsDocument {
        terms: Vec<Term>,
    }

    impl SingleDocument for UnsortedTermsDocument {
        fn term_info(&self, _term: &Term) -> Option<SingleDocumentTermInfo<'_>> {
            None
        }

        fn fieldnorm_id(&self, _field: Field) -> Option<u8> {
            None
        }

        fn visit_terms(
            &self,
            _field: Field,
            _range: (Bound<&Term>, Bound<&Term>),
            visitor: &mut dyn FnMut(&Term, SingleDocumentTermInfo<'_>) -> bool,
        ) {
            for term in &self.terms {
                if !visitor(
                    term,
                    SingleDocumentTermInfo {
                        term_freq: 1,
                        positions: None,
                    },
                ) {
                    break;
                }
            }
        }
    }

    let (schema, field) = text_schema();
    let mut query = PhrasePrefixQuery::new(vec![Term::from_field_text(field, "c")]);
    query.set_max_expansions(1);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))
        .unwrap();
    let document = UnsortedTermsDocument {
        terms: vec![
            Term::from_field_text(field, "cb"),
            Term::from_field_text(field, "ca"),
        ],
    };

    let _ = evaluator.evaluate(&document);
}

#[test]
fn single_term_phrase_prefix_does_not_require_positions() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_text_field("body", STRING);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut document = TantivyDocument::new();
    document.add_text(field, "rust");
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;
    let prepared = preparer.prepare(&document)?;
    let query = PhrasePrefixQuery::new(vec![Term::from_field_text(field, "rus")]);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        evaluator.evaluate(&prepared)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn single_document_phrase_prefix_rejects_mismatched_term_type() {
    let (schema, field) = text_schema();
    let text_term = Term::from_field_text(field, "rust");
    let u64_term = Term::from_field_u64(field, 42);
    let queries = [
        PhrasePrefixQuery::new(vec![u64_term.clone()]),
        PhrasePrefixQuery::new(vec![u64_term.clone(), text_term.clone()]),
        PhrasePrefixQuery::new(vec![text_term, u64_term]),
    ];

    for query in queries {
        let error = query
            .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))
            .err()
            .unwrap();
        assert!(matches!(
            error,
            TantivyError::SchemaError(message)
                if message
                    == "Create a phrase prefix query of the type U64, when the field given was of \
                        type Str"
        ));
    }
}

#[test]
fn single_term_phrase_prefix_score_matches_segment_range_query() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_text_field("body", STRING);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "rust"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();

    let query = BoostQuery::new(
        Box::new(PhrasePrefixQuery::new(vec![Term::from_field_text(
            field, "rus",
        )])),
        2.5,
    );
    let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TantivyDocument::new();
    document.add_text(field, "rust");
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;
    let prepared = preparer.prepare(&document)?;
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &searcher),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&prepared)? else {
        panic!("prefix-only phrase prefix query should match");
    };
    assert_eq!(actual_score, expected_score);
    assert_eq!(actual_score, 2.5);
    Ok(())
}

#[test]
fn term_score_without_fieldnorms_matches_segment_scorer() -> crate::Result<()> {
    let text_options = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_index_option(IndexRecordOption::WithFreqsAndPositions)
            .set_fieldnorms(false),
    );
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_text_field("body", text_options);
    let schema = schema_builder.build();
    let term = Term::from_field_text(field, "rust");
    let query = TermQuery::new(term.clone(), IndexRecordOption::WithFreqs);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 40,
        doc_freqs: HashMap::from([(term.clone(), 4)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "rust rust other"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(term, 2);
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&document)? else {
        panic!("document should match");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn malformed_single_document_term_data_returns_an_error() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term_a = Term::from_field_text(field, "a");
    let term_b = Term::from_field_text(field, "b");

    let term_query = TermQuery::new(term_a.clone(), IndexRecordOption::WithFreqs);
    let mut term_evaluator = term_query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    let mut zero_frequency_document = TestDocument::default();
    zero_frequency_document.term_freqs.insert(term_a.clone(), 0);
    let error = term_evaluator
        .evaluate(&zero_frequency_document)
        .unwrap_err();
    let TantivyError::InvalidArgument(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("zero term frequency"));

    let phrase_query = PhraseQuery::new(vec![term_a.clone(), term_b.clone()]);
    let mut phrase_evaluator = phrase_query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;

    let mut missing_positions_document = TestDocument::default();
    missing_positions_document
        .term_freqs
        .insert(term_a.clone(), 1);
    missing_positions_document
        .term_freqs
        .insert(term_b.clone(), 1);
    missing_positions_document
        .positions
        .insert(term_b.clone(), vec![1]);
    let error = phrase_evaluator
        .evaluate(&missing_positions_document)
        .unwrap_err();
    let TantivyError::InvalidArgument(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("did not supply positions"));

    let mut wrong_position_count_document = TestDocument::default();
    wrong_position_count_document
        .term_freqs
        .insert(term_a.clone(), 2);
    wrong_position_count_document
        .positions
        .insert(term_a.clone(), vec![0]);
    wrong_position_count_document
        .term_freqs
        .insert(term_b.clone(), 1);
    wrong_position_count_document
        .positions
        .insert(term_b.clone(), vec![1]);
    let error = phrase_evaluator
        .evaluate(&wrong_position_count_document)
        .unwrap_err();
    let TantivyError::InvalidArgument(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("1 positions for term frequency 2"));

    let mut unsorted_positions_document = TestDocument::default();
    unsorted_positions_document
        .term_freqs
        .insert(term_a.clone(), 2);
    unsorted_positions_document
        .positions
        .insert(term_a, vec![2, 0]);
    unsorted_positions_document
        .term_freqs
        .insert(term_b.clone(), 1);
    unsorted_positions_document
        .positions
        .insert(term_b, vec![1]);
    let error = phrase_evaluator
        .evaluate(&unsorted_positions_document)
        .unwrap_err();
    let TantivyError::InvalidArgument(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("positions are not sorted"));
    Ok(())
}

#[test]
fn scoring_requires_fieldnorms_for_term_and_phrase_queries() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term_a = Term::from_field_text(field, "a");
    let term_b = Term::from_field_text(field, "b");
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 30,
        doc_freqs: HashMap::from([(term_a.clone(), 5), (term_b.clone(), 4)]),
    };

    let term_query = TermQuery::new(term_a.clone(), IndexRecordOption::WithFreqs);
    let mut term_evaluator = term_query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let mut term_document = TestDocument::default();
    term_document.term_freqs.insert(term_a.clone(), 1);
    let error = term_evaluator.evaluate(&term_document).unwrap_err();
    let TantivyError::InvalidArgument(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("did not supply a fieldnorm"));

    let phrase_query = PhraseQuery::new(vec![term_a.clone(), term_b.clone()]);
    let mut phrase_evaluator = phrase_query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let mut phrase_document = TestDocument::default();
    phrase_document.term_freqs.insert(term_a.clone(), 1);
    phrase_document.positions.insert(term_a, vec![0]);
    phrase_document.term_freqs.insert(term_b.clone(), 1);
    phrase_document.positions.insert(term_b, vec![1]);
    let error = phrase_evaluator.evaluate(&phrase_document).unwrap_err();
    let TantivyError::InvalidArgument(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("did not supply a fieldnorm"));
    Ok(())
}

#[test]
fn single_document_flat_or_score_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let rust_term = Term::from_field_text(field, "rust");
    let cloud_term = Term::from_field_text(field, "cloud");
    let query = BooleanQuery::new(vec![
        (
            Occur::Should,
            Box::new(TermQuery::new(
                rust_term.clone(),
                IndexRecordOption::WithFreqs,
            )),
        ),
        (
            Occur::Should,
            Box::new(TermQuery::new(
                rust_term.clone(),
                IndexRecordOption::WithFreqs,
            )),
        ),
        (
            Occur::Should,
            Box::new(TermQuery::new(
                cloud_term.clone(),
                IndexRecordOption::WithFreqs,
            )),
        ),
    ]);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 40,
        doc_freqs: HashMap::from([(rust_term.clone(), 4), (cloud_term, 2)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "rust rust other"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(rust_term, 2);
    document
        .fieldnorms
        .insert(field, FieldNormReader::fieldnorm_to_id(3));
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&document)? else {
        panic!("document should match");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn single_document_must_should_score_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let required = Term::from_field_text(field, "required");
    let optional = Term::from_field_text(field, "optional");
    let query = BooleanQuery::new(vec![
        (
            Occur::Must,
            Box::new(TermQuery::new(
                required.clone(),
                IndexRecordOption::WithFreqs,
            )),
        ),
        (
            Occur::Should,
            Box::new(TermQuery::new(
                optional.clone(),
                IndexRecordOption::WithFreqs,
            )),
        ),
    ]);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 40,
        doc_freqs: HashMap::from([(required.clone(), 4), (optional.clone(), 2)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "required required optional"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(required, 2);
    document.term_freqs.insert(optional, 1);
    document
        .fieldnorms
        .insert(field, FieldNormReader::fieldnorm_to_id(3));
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&document)? else {
        panic!("document should match");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn boolean_single_document_match_semantics() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let required = Term::from_field_text(field, "required");
    let optional = Term::from_field_text(field, "optional");
    let excluded = Term::from_field_text(field, "excluded");
    let term_query =
        |term| Box::new(TermQuery::new(term, IndexRecordOption::Basic)) as Box<dyn Query>;
    let query = BooleanQuery::new(vec![
        (Occur::Must, term_query(required.clone())),
        (Occur::Should, term_query(optional)),
        (Occur::MustNot, term_query(excluded.clone())),
    ]);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;

    let mut document = TestDocument::default();
    assert_eq!(evaluator.evaluate(&document)?, DocumentEvaluation::NoMatch);
    document.term_freqs.insert(required, 1);
    assert_eq!(
        evaluator.evaluate(&document)?,
        DocumentEvaluation::Match(1.0)
    );
    document.term_freqs.insert(excluded, 1);
    assert_eq!(evaluator.evaluate(&document)?, DocumentEvaluation::NoMatch);
    Ok(())
}

#[test]
fn boolean_with_only_must_not_matches_no_documents() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let excluded = Term::from_field_text(field, "excluded");
    let query = BooleanQuery::new(vec![(
        Occur::MustNot,
        Box::new(TermQuery::new(excluded.clone(), IndexRecordOption::Basic)),
    )]);
    let statistics = TestStatistics {
        total_docs: 1,
        total_tokens: 1,
        doc_freqs: HashMap::from([(excluded.clone(), 1)]),
    };

    for context in [
        SingleDocumentEvaluationContext::without_scoring(&schema),
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    ] {
        let mut evaluator = query.single_document_evaluator(context)?;
        assert_eq!(
            evaluator.evaluate(&TestDocument::default())?,
            DocumentEvaluation::NoMatch
        );

        let mut excluded_document = TestDocument::default();
        excluded_document.term_freqs.insert(excluded.clone(), 1);
        assert_eq!(
            evaluator.evaluate(&excluded_document)?,
            DocumentEvaluation::NoMatch
        );
    }
    Ok(())
}

#[test]
fn boolean_must_not_does_not_require_scoring_data() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let excluded = Term::from_field_text(field, "excluded");
    let query = BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery)),
        (
            Occur::MustNot,
            Box::new(TermQuery::new(
                excluded.clone(),
                IndexRecordOption::WithFreqs,
            )),
        ),
    ]);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 20,
        doc_freqs: HashMap::from([(excluded.clone(), 2)]),
    };
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let mut document = TestDocument::default();
    document.term_freqs.insert(excluded, 1);

    assert_eq!(evaluator.evaluate(&document)?, DocumentEvaluation::NoMatch);
    Ok(())
}

#[test]
fn boolean_must_not_score_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let required = Term::from_field_text(field, "required");
    let excluded = Term::from_field_text(field, "excluded");
    let query = BooleanQuery::new(vec![
        (
            Occur::Must,
            Box::new(TermQuery::new(
                required.clone(),
                IndexRecordOption::WithFreqs,
            )),
        ),
        (
            Occur::MustNot,
            Box::new(TermQuery::new(
                excluded.clone(),
                IndexRecordOption::WithFreqs,
            )),
        ),
    ]);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 30,
        doc_freqs: HashMap::from([(required.clone(), 4), (excluded, 2)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "required required"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(required, 2);
    document
        .fieldnorms
        .insert(field, FieldNormReader::fieldnorm_to_id(2));
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&document)? else {
        panic!("document should match");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn boolean_without_scoring_skips_should_when_must_exists() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let regex = RegexQuery::from_pattern("ru.*", field)?;
    let query = BooleanQuery::new(vec![
        (Occur::Must, Box::new(AllQuery)),
        (Occur::Should, Box::new(regex)),
    ]);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(evaluator.required_fields(), Some([].as_slice()));

    assert_eq!(
        evaluator.evaluate(&TestDocument::default())?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn boolean_without_scoring_short_circuits_matching_should() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term_a = Term::from_field_text(field, "a");
    let term_b = Term::from_field_text(field, "b");
    let query = BooleanQuery::new(vec![
        (Occur::Should, Box::new(AllQuery)),
        (
            Occur::Should,
            Box::new(PhraseQuery::new(vec![term_a.clone(), term_b.clone()])),
        ),
    ]);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    let mut document = TestDocument::default();
    document.term_freqs.insert(term_a, 1);
    document.term_freqs.insert(term_b, 1);

    assert_eq!(
        evaluator.evaluate(&document)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn wrapper_scores_match_current_segment_scorer_semantics() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "body"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let statistics = TestStatistics {
        total_docs: 1,
        total_tokens: 1,
        doc_freqs: HashMap::new(),
    };

    let queries: Vec<Box<dyn Query>> = vec![
        Box::new(BoostQuery::new(
            Box::new(BooleanQuery::union(vec![
                Box::new(AllQuery),
                Box::new(AllQuery),
            ])),
            -2.0,
        )),
        Box::new(BoostQuery::new(
            Box::new(ConstScoreQuery::new(Box::new(AllQuery), 3.0)),
            -2.0,
        )),
    ];

    for query in queries {
        let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
            &statistics,
            &searcher,
        ))?;
        let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
        assert_eq!(scorer.doc(), 0);
        let expected_score = scorer.score();

        let mut evaluator = query.single_document_evaluator(
            SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
        )?;
        assert_eq!(
            evaluator.evaluate(&TestDocument::default())?,
            DocumentEvaluation::Match(expected_score)
        );
    }
    Ok(())
}

#[test]
fn boosted_term_score_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term = Term::from_field_text(field, "rust");
    let query = BoostQuery::new(
        Box::new(TermQuery::new(term.clone(), IndexRecordOption::WithFreqs)),
        2.5,
    );
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 40,
        doc_freqs: HashMap::from([(term.clone(), 4)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "rust rust other"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(term, 2);
    document
        .fieldnorms
        .insert(field, FieldNormReader::fieldnorm_to_id(3));
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&document)? else {
        panic!("document should match");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn const_score_preserves_subquery_no_match() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term = Term::from_field_text(field, "rust");
    let query = ConstScoreQuery::new(
        Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
        3.0,
    );
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;

    assert_eq!(
        evaluator.evaluate(&TestDocument::default())?,
        DocumentEvaluation::NoMatch
    );
    Ok(())
}

#[test]
fn single_document_evaluator_validates_schema_requirements() {
    let mut stored_schema_builder = Schema::builder();
    let stored_field = stored_schema_builder.add_text_field("stored_body", STORED);
    let stored_schema = stored_schema_builder.build();
    let term_query = TermQuery::new(
        Term::from_field_text(stored_field, "rust"),
        IndexRecordOption::Basic,
    );
    let error = term_query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(
            &stored_schema,
        ))
        .err()
        .unwrap();
    let TantivyError::SchemaError(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("is not indexed"));

    let no_positions_options = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default().set_index_option(IndexRecordOption::WithFreqs),
    );
    let mut no_positions_schema_builder = Schema::builder();
    let no_positions_field =
        no_positions_schema_builder.add_text_field("body", no_positions_options);
    let no_positions_schema = no_positions_schema_builder.build();
    let phrase_query = PhraseQuery::new(vec![
        Term::from_field_text(no_positions_field, "a"),
        Term::from_field_text(no_positions_field, "b"),
    ]);
    let error = phrase_query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(
            &no_positions_schema,
        ))
        .err()
        .unwrap();
    let TantivyError::SchemaError(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("does not have positions indexed"));
}

#[test]
fn term_set_is_unsupported() {
    let mut schema_builder = Schema::builder();
    let title = schema_builder.add_text_field("title", TEXT);
    let schema = schema_builder.build();
    let query = TermSetQuery::new([Term::from_field_text(title, "rust")]);

    let error = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))
        .err()
        .unwrap();
    let TantivyError::UnsupportedQueryForSingleDocumentEvaluation(query_type) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(query_type.ends_with("TermSetQuery"));
}

#[test]
fn non_string_json_phrase_term_is_unsupported() {
    let mut schema_builder = Schema::builder();
    let json_field = schema_builder.add_json_field("json", TEXT);
    let schema = schema_builder.build();

    let mut string_term = Term::with_type_and_field(Type::Json, json_field);
    {
        let mut writer = JsonTermWriter::wrap(&mut string_term, false);
        writer.push_path_segment("value");
        writer.set_str("one");
    }
    let mut numeric_term = Term::with_type_and_field(Type::Json, json_field);
    {
        let mut writer = JsonTermWriter::wrap(&mut numeric_term, false);
        writer.push_path_segment("value");
        writer.set_fast_value(2u64);
    }

    let query = PhraseQuery::new(vec![string_term, numeric_term]);
    let error = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))
        .err()
        .unwrap();
    let TantivyError::UnsupportedQueryForSingleDocumentEvaluation(message) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(message.contains("does not have position postings"));
}

#[test]
fn string_json_phrase_matches_without_scoring() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let json_field = schema_builder.add_json_field("json", TEXT);
    let schema = schema_builder.build();

    let mut term_a = Term::with_type_and_field(Type::Json, json_field);
    {
        let mut writer = JsonTermWriter::wrap(&mut term_a, false);
        writer.push_path_segment("value");
        writer.set_str("a");
    }
    let mut term_b = Term::with_type_and_field(Type::Json, json_field);
    {
        let mut writer = JsonTermWriter::wrap(&mut term_b, false);
        writer.push_path_segment("value");
        writer.set_str("b");
    }

    let query = PhraseQuery::new(vec![term_a.clone(), term_b.clone()]);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    let mut document = TestDocument::default();
    document.term_freqs.insert(term_a.clone(), 1);
    document.positions.insert(term_a, vec![0]);
    document.term_freqs.insert(term_b.clone(), 1);
    document.positions.insert(term_b, vec![1]);

    assert_eq!(
        evaluator.evaluate(&document)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn phrase_without_scoring_is_reusable_for_match_and_no_match() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term_a = Term::from_field_text(field, "a");
    let term_b = Term::from_field_text(field, "b");
    let query = PhraseQuery::new(vec![term_a.clone(), term_b.clone()]);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;

    let mut matching_document = TestDocument::default();
    matching_document.term_freqs.insert(term_a.clone(), 2);
    matching_document
        .positions
        .insert(term_a.clone(), vec![0, 10]);
    matching_document.term_freqs.insert(term_b.clone(), 2);
    matching_document
        .positions
        .insert(term_b.clone(), vec![1, 11]);
    assert_eq!(
        evaluator.evaluate(&matching_document)?,
        DocumentEvaluation::Match(1.0)
    );

    let mut non_matching_document = TestDocument::default();
    non_matching_document.term_freqs.insert(term_a.clone(), 1);
    non_matching_document
        .positions
        .insert(term_a.clone(), vec![0]);
    non_matching_document.term_freqs.insert(term_b.clone(), 1);
    non_matching_document
        .positions
        .insert(term_b.clone(), vec![2]);
    assert_eq!(
        evaluator.evaluate(&non_matching_document)?,
        DocumentEvaluation::NoMatch
    );
    assert_eq!(
        evaluator.evaluate(&matching_document)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn phrase_with_explicit_offsets_matches_expected_positions() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term_a = Term::from_field_text(field, "a");
    let term_b = Term::from_field_text(field, "b");
    let query = PhraseQuery::new_with_offset(vec![(0, term_a.clone()), (2, term_b.clone())]);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;

    let mut document = TestDocument::default();
    document.term_freqs.insert(term_a.clone(), 1);
    document.positions.insert(term_a, vec![0]);
    document.term_freqs.insert(term_b.clone(), 1);
    document.positions.insert(term_b.clone(), vec![2]);
    assert_eq!(
        evaluator.evaluate(&document)?,
        DocumentEvaluation::Match(1.0)
    );

    document.positions.insert(term_b, vec![1]);
    assert_eq!(evaluator.evaluate(&document)?, DocumentEvaluation::NoMatch);
    Ok(())
}

#[test]
fn single_document_phrase_score_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term_a = Term::from_field_text(field, "a");
    let term_b = Term::from_field_text(field, "b");
    let query = PhraseQuery::new(vec![term_a.clone(), term_b.clone()]);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 50,
        doc_freqs: HashMap::from([(term_a.clone(), 5), (term_b.clone(), 4)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "a x b a b"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(term_a.clone(), 2);
    document.positions.insert(term_a, vec![0, 3]);
    document.term_freqs.insert(term_b.clone(), 2);
    document.positions.insert(term_b, vec![2, 4]);
    document
        .fieldnorms
        .insert(field, FieldNormReader::fieldnorm_to_id(5));
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&document)? else {
        panic!("document should match the phrase");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn single_document_phrase_with_slop_score_matches_segment_scorer() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let term_a = Term::from_field_text(field, "a");
    let term_b = Term::from_field_text(field, "b");
    let mut query = PhraseQuery::new(vec![term_a.clone(), term_b.clone()]);
    query.set_slop(1);
    let statistics = TestStatistics {
        total_docs: 10,
        total_tokens: 30,
        doc_freqs: HashMap::from([(term_a.clone(), 5), (term_b.clone(), 4)]),
    };

    let index = Index::create_in_ram(schema.clone());
    let mut writer: IndexWriter = index.writer_for_tests()?;
    writer.add_document(doc!(field => "a x b"))?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        &statistics,
        &searcher,
    ))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    assert_eq!(scorer.doc(), 0);
    let expected_score = scorer.score();

    let mut document = TestDocument::default();
    document.term_freqs.insert(term_a.clone(), 1);
    document.positions.insert(term_a, vec![0]);
    document.term_freqs.insert(term_b.clone(), 1);
    document.positions.insert(term_b.clone(), vec![2]);
    document
        .fieldnorms
        .insert(field, FieldNormReader::fieldnorm_to_id(3));
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &statistics),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&document)? else {
        panic!("document should match the phrase with slop");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);

    document.positions.insert(term_b, vec![3]);
    assert_eq!(evaluator.evaluate(&document)?, DocumentEvaluation::NoMatch);
    Ok(())
}

#[test]
fn unsupported_query_includes_query_path() {
    let (schema, field) = text_schema();
    let regex = RegexQuery::from_pattern("ru.*", field).unwrap();
    let query = BooleanQuery::new(vec![(Occur::Should, Box::new(regex))]);
    let error = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))
        .err()
        .unwrap();
    let TantivyError::UnsupportedQueryForSingleDocumentEvaluation(path) = error else {
        panic!("unexpected error: {error:?}");
    };
    assert!(path.contains("BooleanQuery.Should[0]"));
    assert!(path.contains("RegexQuery"));
}

#[test]
fn prepared_document_matches_text_indexing_semantics() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let mut document = TantivyDocument::new();
    document.add_text(field, "Rust rust");
    document.add_text(field, "search");

    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;
    let prepared = preparer.prepare(&document)?;
    let rust_term = Term::from_field_text(field, "rust");
    let search_term = Term::from_field_text(field, "search");
    let rust_info = prepared.term_info(&rust_term).unwrap();
    assert_eq!(rust_info.term_freq, 2);
    assert_eq!(rust_info.positions, Some([0, 1].as_slice()));
    let search_info = prepared.term_info(&search_term).unwrap();
    assert_eq!(search_info.positions, Some([3].as_slice()));
    assert_eq!(
        prepared.fieldnorm_id(field),
        Some(FieldNormReader::fieldnorm_to_id(3))
    );
    assert_eq!(prepared.total_num_tokens(field), 3);

    let phrase = PhraseQuery::new(vec![rust_term.clone(), search_term]);
    let mut phrase_evaluator = phrase
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        phrase_evaluator.evaluate(&prepared)?,
        DocumentEvaluation::NoMatch
    );

    let query = TermQuery::new(rust_term, IndexRecordOption::WithFreqs);
    let mut writer = index.writer_for_tests()?;
    writer.add_document(document)?;
    writer.commit()?;
    let searcher = index.reader()?.searcher();
    let weight = query.weight(EnableScoring::enabled_from_searcher(&searcher))?;
    let mut scorer = weight.scorer(searcher.segment_reader(0), 1.0)?;
    let expected_score = scorer.score();
    let mut evaluator = query.single_document_evaluator(
        SingleDocumentEvaluationContext::with_scoring(&schema, &searcher),
    )?;
    let DocumentEvaluation::Match(actual_score) = evaluator.evaluate(&prepared)? else {
        panic!("prepared document should match");
    };
    assert!((actual_score - expected_score).abs() <= 1e-6);
    Ok(())
}

#[test]
fn prepared_document_visit_terms_honors_range() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let document = doc!(field => "aa ab ac ba");
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;
    let prepared = preparer.prepare(&document)?;
    let lower = Term::from_field_text(field, "ab");
    let upper = Term::from_field_text(field, "ba");
    let mut visited = Vec::new();
    prepared.visit_terms(
        field,
        (Bound::Excluded(&lower), Bound::Included(&upper)),
        &mut |term, _| {
            visited.push(term.clone());
            true
        },
    );
    assert_eq!(
        visited,
        vec![
            Term::from_field_text(field, "ac"),
            Term::from_field_text(field, "ba")
        ]
    );
    Ok(())
}

#[test]
fn evaluator_evaluates_a_prepared_regular_document() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let mut document = TantivyDocument::new();
    document.add_text(field, "Rust search");
    let query = TermQuery::new(
        Term::from_field_text(field, "rust"),
        IndexRecordOption::Basic,
    );
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    let mut preparer = SingleDocumentPreparer::for_fields(&schema, index.tokenizers(), &[field])?;
    let prepared = preparer.prepare(&document)?;

    assert_eq!(
        evaluator.evaluate(&prepared)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn prepared_document_preserves_pre_tokenized_positions() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let tokenized = PreTokenizedString {
        text: "first second".to_string(),
        tokens: vec![
            Token {
                offset_from: 0,
                offset_to: 5,
                position: 4,
                text: "first".to_string(),
                position_length: 1,
            },
            Token {
                offset_from: 6,
                offset_to: 12,
                position: 7,
                text: "second".to_string(),
                position_length: 1,
            },
        ],
    };
    let mut document = TantivyDocument::new();
    document.add_pre_tokenized_text(field, tokenized);

    let mut preparer = preparer_for_fields(
        &schema,
        &crate::tokenizer::TokenizerManager::default(),
        &[field],
    )?;
    let prepared = preparer.prepare(&document)?;
    let first = prepared
        .term_info(&Term::from_field_text(field, "first"))
        .unwrap();
    let second = prepared
        .term_info(&Term::from_field_text(field, "second"))
        .unwrap();
    assert_eq!(first.positions, Some([4].as_slice()));
    assert_eq!(second.positions, Some([7].as_slice()));
    assert_eq!(
        prepared.fieldnorm_id(field),
        Some(FieldNormReader::fieldnorm_to_id(2))
    );
    Ok(())
}

#[test]
fn prepared_document_uses_the_supplied_tokenizer_manager() -> crate::Result<()> {
    let text_options = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default().set_tokenizer("case_sensitive_whitespace"),
    );
    let mut schema_builder = Schema::builder();
    let field = schema_builder.add_text_field("body", text_options);
    let schema = schema_builder.build();
    let tokenizer_manager = TokenizerManager::new();
    tokenizer_manager.register("case_sensitive_whitespace", WhitespaceTokenizer::default());
    let mut document = TantivyDocument::new();
    document.add_text(field, "Rust Search");

    let mut preparer = preparer_for_fields(&schema, &tokenizer_manager, &[field])?;
    let prepared = preparer.prepare(&document)?;
    assert!(prepared
        .term_info(&Term::from_field_text(field, "Rust"))
        .is_some());
    assert!(prepared
        .term_info(&Term::from_field_text(field, "rust"))
        .is_none());

    let error = preparer_for_fields(&schema, &TokenizerManager::default(), &[field])
        .err()
        .unwrap();
    assert!(matches!(error, TantivyError::SchemaError(_)));
    Ok(())
}

#[test]
fn single_document_preparer_normalizes_and_validates_fields() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let title = schema_builder.add_text_field("title", TEXT);
    let body = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let tokenizer_manager = TokenizerManager::default();

    let mut preparer =
        SingleDocumentPreparer::for_fields(&schema, &tokenizer_manager, &[body, title, body])?;
    let document = doc!(title => "Rust", body => "Search");
    let prepared = preparer.prepare(&document)?;
    assert!(prepared
        .term_info(&Term::from_field_text(title, "rust"))
        .is_some());
    assert!(prepared
        .term_info(&Term::from_field_text(body, "search"))
        .is_some());

    let unknown_field = Field::from_field_id(schema.num_fields() as u32);
    let error = SingleDocumentPreparer::for_fields(&schema, &tokenizer_manager, &[unknown_field])
        .err()
        .unwrap();
    assert!(matches!(error, TantivyError::SchemaError(_)));
    Ok(())
}

#[test]
fn prepared_document_encodes_numeric_and_json_terms() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let number = schema_builder.add_u64_field("number", INDEXED);
    let json = schema_builder.add_json_field("json", TEXT);
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());
    let document = doc!(
        number => 7u64,
        json => serde_json::json!({"title": "Rust search", "count": 2}),
    );

    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[number, json])?;
    let prepared = preparer.prepare(&document)?;
    assert_eq!(prepared.total_num_tokens(number), 1);
    assert_eq!(prepared.total_num_tokens(json), 3);
    assert_eq!(
        prepared
            .term_info(&Term::from_field_u64(number, 7u64))
            .unwrap()
            .term_freq,
        1
    );

    let mut json_text_term = Term::with_type_and_field(Type::Json, json);
    {
        let mut term_writer = JsonTermWriter::wrap(&mut json_text_term, false);
        term_writer.push_path_segment("title");
        term_writer.set_str("rust");
    }
    assert_eq!(
        prepared.term_info(&json_text_term).unwrap().positions,
        Some([0].as_slice())
    );

    let mut json_number_term = Term::with_type_and_field(Type::Json, json);
    {
        let mut term_writer = JsonTermWriter::wrap(&mut json_number_term, false);
        term_writer.push_path_segment("count");
        term_writer.set_fast_value(2i64);
    }
    assert_eq!(prepared.term_info(&json_number_term).unwrap().term_freq, 1);
    Ok(())
}

#[test]
fn single_document_preparer_can_be_reused() -> crate::Result<()> {
    let (schema, field) = text_schema();
    let index = Index::create_in_ram(schema.clone());
    let mut preparer = preparer_for_fields(&schema, index.tokenizers(), &[field])?;

    let mut first_document = TantivyDocument::new();
    first_document.add_text(field, "rust rust");
    let first_prepared = preparer.prepare(&first_document)?;

    let mut second_document = TantivyDocument::new();
    second_document.add_text(field, "search");
    let second_prepared = preparer.prepare(&second_document)?;

    let rust = Term::from_field_text(field, "rust");
    let search = Term::from_field_text(field, "search");
    assert_eq!(first_prepared.term_info(&rust).unwrap().term_freq, 2);
    assert!(first_prepared.term_info(&search).is_none());
    assert!(second_prepared.term_info(&rust).is_none());
    assert_eq!(
        second_prepared.term_info(&search).unwrap().positions,
        Some([0].as_slice())
    );
    assert_eq!(first_prepared.total_num_tokens(field), 2);
    assert_eq!(second_prepared.total_num_tokens(field), 1);
    Ok(())
}

#[test]
fn single_document_preparer_only_indexes_configured_fields() -> crate::Result<()> {
    let ignored_options = TextOptions::default()
        .set_indexing_options(TextFieldIndexing::default().set_tokenizer("missing_tokenizer"));
    let mut schema_builder = Schema::builder();
    let queried_field = schema_builder.add_text_field("queried", TEXT);
    let ignored_field = schema_builder.add_text_field("ignored", ignored_options);
    let schema = schema_builder.build();
    let tokenizer_manager = TokenizerManager::default();
    let query = TermQuery::new(
        Term::from_field_text(queried_field, "rust"),
        IndexRecordOption::Basic,
    );
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        evaluator.required_fields(),
        Some([queried_field].as_slice())
    );
    let mut preparer =
        SingleDocumentPreparer::for_fields(&schema, &tokenizer_manager, &[queried_field])?;
    let mut document = TantivyDocument::new();
    document.add_text(queried_field, "Rust search");
    document.add_text(ignored_field, "ignored text");

    let prepared = preparer.prepare(&document)?;
    assert_eq!(
        evaluator.evaluate(&prepared)?,
        DocumentEvaluation::Match(1.0)
    );
    assert_eq!(prepared.total_num_tokens(queried_field), 2);
    assert_eq!(prepared.total_num_tokens(ignored_field), 0);
    assert!(prepared
        .term_info(&Term::from_field_text(ignored_field, "ignored"))
        .is_none());

    Ok(())
}

#[test]
fn query_aware_preparer_checks_presence_and_skips_top_level_null() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let number = schema_builder.add_u64_field("number", INDEXED);
    let schema = schema_builder.build();
    let query = TermQuery::new(Term::from_field_u64(number, 42), IndexRecordOption::Basic);
    let mut evaluator = query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    let mut preparer =
        SingleDocumentPreparer::for_fields(&schema, &TokenizerManager::default(), &[number])?;

    let missing_error = preparer.prepare(&TantivyDocument::new()).err().unwrap();
    assert!(matches!(
        missing_error,
        TantivyError::InvalidArgument(message)
            if message.contains("explicitly supplied") && message.contains("number")
    ));

    let mut duplicate = TantivyDocument::new();
    duplicate.add_u64(number, 42);
    duplicate.add_u64(number, 42);
    let prepared_duplicate = preparer.prepare(&duplicate)?;
    assert_eq!(
        evaluator.evaluate(&prepared_duplicate)?,
        DocumentEvaluation::Match(1.0)
    );

    let mut mixed_null = TantivyDocument::new();
    mixed_null.add_u64(number, 42);
    mixed_null.add_field_value(number, OwnedValue::Null);
    let prepared_mixed_null = preparer.prepare(&mixed_null)?;
    assert_eq!(prepared_mixed_null.total_num_tokens(number), 1);
    assert_eq!(
        evaluator.evaluate(&prepared_mixed_null)?,
        DocumentEvaluation::Match(1.0)
    );

    let mut null_document = TantivyDocument::new();
    null_document.add_field_value(number, OwnedValue::Null);
    let prepared_null = preparer.prepare(&null_document)?;
    assert_eq!(prepared_null.total_num_tokens(number), 0);
    assert_eq!(
        evaluator.evaluate(&prepared_null)?,
        DocumentEvaluation::NoMatch
    );

    let mut matching_document = TantivyDocument::new();
    matching_document.add_u64(number, 42);
    let prepared_match = preparer.prepare(&matching_document)?;
    assert_eq!(
        evaluator.evaluate(&prepared_match)?,
        DocumentEvaluation::Match(1.0)
    );
    Ok(())
}

#[test]
fn query_aware_prepared_document_rejects_broader_evaluator() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let title = schema_builder.add_text_field("title", TEXT);
    let body = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let tokenizer_manager = TokenizerManager::default();

    let title_query = TermQuery::new(
        Term::from_field_text(title, "rust"),
        IndexRecordOption::Basic,
    );
    let mut title_evaluator = title_query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    let body_query = TermQuery::new(
        Term::from_field_text(body, "search"),
        IndexRecordOption::Basic,
    );
    let mut body_evaluator = body_query
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;

    let mut query_aware_preparer =
        SingleDocumentPreparer::for_fields(&schema, &tokenizer_manager, &[title])?;
    let mut document = TantivyDocument::new();
    document.add_text(title, "Rust");
    let query_aware_document = query_aware_preparer.prepare(&document)?;
    assert_eq!(
        title_evaluator.evaluate(&query_aware_document)?,
        DocumentEvaluation::Match(1.0)
    );
    assert!(matches!(
        body_evaluator.evaluate(&query_aware_document),
        Err(TantivyError::InvalidArgument(message))
            if message.contains("did not prepare required field")
    ));

    struct UnknownRequirementsEvaluator;
    impl SingleDocumentEvaluator for UnknownRequirementsEvaluator {
        fn evaluate_impl(
            &mut self,
            _document: &dyn SingleDocument,
        ) -> crate::Result<DocumentEvaluation> {
            Ok(DocumentEvaluation::NoMatch)
        }
    }
    let mut unknown_evaluator = UnknownRequirementsEvaluator;
    assert!(matches!(
        unknown_evaluator.evaluate(&query_aware_document),
        Err(TantivyError::InvalidArgument(message))
            if message.contains("required fields are unknown")
    ));

    let mut all_evaluator = AllQuery
        .single_document_evaluator(SingleDocumentEvaluationContext::without_scoring(&schema))?;
    assert_eq!(
        all_evaluator.evaluate(&query_aware_document)?,
        DocumentEvaluation::Match(1.0)
    );

    Ok(())
}
