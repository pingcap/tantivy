use std::ops::Bound;

use levenshtein_automata::{Distance, LevenshteinAutomatonBuilder, DFA};
use once_cell::sync::OnceCell;
use tantivy_fst::Automaton;

use crate::query::single_document::{effective_record_option, validate_term_info};
use crate::query::{
    AutomatonWeight, DocumentEvaluation, EnableScoring, Query, SingleDocument,
    SingleDocumentEvaluationContext, SingleDocumentEvaluator, Weight,
};
use crate::schema::{Field, IndexRecordOption, Term, Type};
use crate::Score;
use crate::TantivyError::InvalidArgument;

pub(crate) struct DfaWrapper(pub DFA);

impl Automaton for DfaWrapper {
    type State = u32;

    fn start(&self) -> Self::State {
        self.0.initial_state()
    }

    fn is_match(&self, state: &Self::State) -> bool {
        match self.0.distance(*state) {
            Distance::Exact(_) => true,
            Distance::AtLeast(_) => false,
        }
    }

    fn can_match(&self, state: &u32) -> bool {
        *state != levenshtein_automata::SINK_STATE
    }

    fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
        self.0.transition(*state, byte)
    }
}

/// A Fuzzy Query matches all of the documents
/// containing a specific term that is within
/// Levenshtein distance
/// ```rust
/// use tantivy::collector::{Count, TopDocs};
/// use tantivy::query::FuzzyTermQuery;
/// use tantivy::schema::{Schema, TEXT};
/// use tantivy::{doc, Index, IndexWriter, Term};
///
/// fn example() -> tantivy::Result<()> {
///     let mut schema_builder = Schema::builder();
///     let title = schema_builder.add_text_field("title", TEXT);
///     let schema = schema_builder.build();
///     let index = Index::create_in_ram(schema);
///     {
///         let mut index_writer: IndexWriter = index.writer(15_000_000)?;
///         index_writer.add_document(doc!(
///             title => "The Name of the Wind",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "The Diary of Muadib",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "A Dairy Cow",
///         ))?;
///         index_writer.add_document(doc!(
///             title => "The Diary of a Young Girl",
///         ))?;
///         index_writer.commit()?;
///     }
///     let reader = index.reader()?;
///     let searcher = reader.searcher();
///
///     {
///         let term = Term::from_field_text(title, "Diary");
///         let query = FuzzyTermQuery::new(term, 1, true);
///         let (top_docs, count) = searcher.search(&query, &(TopDocs::with_limit(2), Count)).unwrap();
///         assert_eq!(count, 2);
///         assert_eq!(top_docs.len(), 2);
///     }
///
///     Ok(())
/// }
/// # assert!(example().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct FuzzyTermQuery {
    /// What term are we searching
    term: Term,
    /// How many changes are we going to allow
    distance: u8,
    /// Should a transposition cost 1 or 2?
    transposition_cost_one: bool,
    /// is a starts with query
    prefix: bool,
}

impl FuzzyTermQuery {
    /// Creates a new Fuzzy Query
    pub fn new(term: Term, distance: u8, transposition_cost_one: bool) -> FuzzyTermQuery {
        FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix: false,
        }
    }

    /// Creates a new Fuzzy Query of the Term prefix
    pub fn new_prefix(term: Term, distance: u8, transposition_cost_one: bool) -> FuzzyTermQuery {
        FuzzyTermQuery {
            term,
            distance,
            transposition_cost_one,
            prefix: true,
        }
    }

    fn specialized_automaton(&self) -> crate::Result<(DfaWrapper, Option<Box<[u8]>>)> {
        static AUTOMATON_BUILDER: [[OnceCell<LevenshteinAutomatonBuilder>; 2]; 3] = [
            [OnceCell::new(), OnceCell::new()],
            [OnceCell::new(), OnceCell::new()],
            [OnceCell::new(), OnceCell::new()],
        ];

        let automaton_builder = AUTOMATON_BUILDER
            .get(self.distance as usize)
            .ok_or_else(|| {
                InvalidArgument(format!(
                    "Levenshtein distance of {} is not allowed. Choose a value less than {}",
                    self.distance,
                    AUTOMATON_BUILDER.len()
                ))
            })?
            .get(self.transposition_cost_one as usize)
            .unwrap()
            .get_or_init(|| {
                LevenshteinAutomatonBuilder::new(self.distance, self.transposition_cost_one)
            });

        let term_value = self.term.value();

        let term_text = if term_value.typ() == Type::Json {
            if let Some(json_path_type) = term_value.json_path_type() {
                if json_path_type != Type::Str {
                    return Err(InvalidArgument(format!(
                        "The fuzzy term query requires a string path type for a json term. Found \
                         {:?}",
                        json_path_type
                    )));
                }
            }

            std::str::from_utf8(self.term.serialized_value_bytes()).map_err(|_| {
                InvalidArgument(
                    "Failed to convert json term value bytes to utf8 string.".to_string(),
                )
            })?
        } else {
            term_value.as_str().ok_or_else(|| {
                InvalidArgument("The fuzzy term query requires a string term.".to_string())
            })?
        };
        let automaton = if self.prefix {
            automaton_builder.build_prefix_dfa(term_text)
        } else {
            automaton_builder.build_dfa(term_text)
        };

        let json_path = term_value
            .as_json()
            .map(|(json_path_bytes, _)| json_path_bytes.to_vec().into_boxed_slice());
        Ok((DfaWrapper(automaton), json_path))
    }

    fn specialized_weight(&self) -> crate::Result<AutomatonWeight<DfaWrapper>> {
        let (automaton, json_path) = self.specialized_automaton()?;
        if let Some(json_path_bytes) = json_path {
            Ok(AutomatonWeight::new_for_json_path(
                self.term.field(),
                automaton,
                &json_path_bytes,
            ))
        } else {
            Ok(AutomatonWeight::new(self.term.field(), automaton))
        }
    }
}

impl Query for FuzzyTermQuery {
    fn weight(&self, _enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        Ok(Box::new(self.specialized_weight()?))
    }

    fn single_document_evaluator(
        &self,
        context: SingleDocumentEvaluationContext<'_>,
    ) -> crate::Result<Box<dyn SingleDocumentEvaluator>> {
        effective_record_option(context.schema(), &self.term, IndexRecordOption::Basic)?;
        let (automaton, json_path) = self.specialized_automaton()?;
        let score = if context.is_scoring_enabled() {
            context.boost()
        } else {
            1.0
        };
        Ok(Box::new(FuzzySingleDocumentEvaluator {
            field: self.term.field(),
            automaton,
            json_path,
            score,
        }))
    }
}

struct FuzzySingleDocumentEvaluator {
    field: Field,
    automaton: DfaWrapper,
    json_path: Option<Box<[u8]>>,
    score: Score,
}

impl SingleDocumentEvaluator for FuzzySingleDocumentEvaluator {
    fn evaluate_impl(
        &mut self,
        document: &dyn SingleDocument,
    ) -> crate::Result<DocumentEvaluation> {
        let mut result = Ok(DocumentEvaluation::NoMatch);
        let mut visitor = |term: &Term, term_info: crate::query::SingleDocumentTermInfo<'_>| {
            if matches!(result, Ok(DocumentEvaluation::Match(_)) | Err(_)) {
                return false;
            }
            if let Some(expected_json_path) = &self.json_path {
                let term_value = term.value();
                let Some((candidate_json_path, _)) = term_value.as_json() else {
                    return true;
                };
                if candidate_json_path != expected_json_path.as_ref() {
                    return true;
                }
            }
            if automaton_matches(&self.automaton, term.serialized_value_bytes()) {
                result = validate_term_info(term, term_info.term_freq)
                    .map(|()| DocumentEvaluation::Match(self.score));
                return false;
            }
            true
        };
        document.visit_terms(
            self.field,
            (Bound::Unbounded, Bound::Unbounded),
            &mut visitor,
        );
        result
    }

    fn required_fields(&self) -> Option<&[Field]> {
        Some(std::slice::from_ref(&self.field))
    }
}

fn automaton_matches<A: Automaton>(automaton: &A, bytes: &[u8]) -> bool {
    let mut state = automaton.start();
    for &byte in bytes {
        state = automaton.accept(&state, byte);
        if !automaton.can_match(&state) {
            return false;
        }
    }
    automaton.is_match(&state)
}

#[cfg(test)]
mod test {
    use super::FuzzyTermQuery;
    use crate::collector::{Count, TopDocs};
    use crate::indexer::NoMergePolicy;
    use crate::query::QueryParser;
    use crate::schema::{Schema, STORED, TEXT};
    use crate::{assert_nearly_equals, Index, IndexWriter, TantivyDocument, Term};

    #[test]
    pub fn test_fuzzy_json_path() -> crate::Result<()> {
        // # Defining the schema
        let mut schema_builder = Schema::builder();
        let attributes = schema_builder.add_json_field("attributes", TEXT | STORED);
        let schema = schema_builder.build();

        // # Indexing documents
        let index = Index::create_in_ram(schema.clone());

        let mut index_writer = index.writer_for_tests()?;
        index_writer.set_merge_policy(Box::new(NoMergePolicy));
        let doc = TantivyDocument::parse_json(
            &schema,
            r#"{
            "attributes": {
                "a": "japan"
            }
        }"#,
        )?;
        index_writer.add_document(doc)?;
        let doc = TantivyDocument::parse_json(
            &schema,
            r#"{
            "attributes": {
                "aa": "japan"
            }
        }"#,
        )?;
        index_writer.add_document(doc)?;
        index_writer.commit()?;

        let reader = index.reader()?;
        let searcher = reader.searcher();

        // # Fuzzy search
        let query_parser = QueryParser::for_index(&index, vec![attributes]);

        let get_json_path_term = |query: &str| -> crate::Result<Term> {
            let query = query_parser.parse_query(query)?;
            let mut terms = Vec::new();
            query.query_terms(&mut |term, _| {
                terms.push(term.clone());
            });

            Ok(terms[0].clone())
        };

        // shall not match the first document due to json path mismatch
        {
            let term = get_json_path_term("attributes.aa:japan")?;
            let fuzzy_query = FuzzyTermQuery::new(term, 2, true);
            let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(2))?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            assert_eq!(top_docs[0].1.doc_id, 1, "Expected the second document");
        }

        // shall match the first document because Levenshtein distance is 1 (substitute 'o' with
        // 'a')
        {
            let term = get_json_path_term("attributes.a:japon")?;

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(2))?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            assert_eq!(top_docs[0].1.doc_id, 0, "Expected the first document");
        }

        // shall not match because non-prefix Levenshtein distance is more than 1 (add 'a' and 'n')
        {
            let term = get_json_path_term("attributes.a:jap")?;

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(2))?;
            assert_eq!(top_docs.len(), 0, "Expected no document");
        }

        Ok(())
    }

    #[test]
    pub fn test_fuzzy_term() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        {
            let mut index_writer: IndexWriter = index.writer_for_tests()?;
            index_writer.add_document(doc!(
                country_field => "japan",
            ))?;
            index_writer.add_document(doc!(
                country_field => "korea",
            ))?;
            index_writer.commit()?;
        }
        let reader = index.reader()?;
        let searcher = reader.searcher();

        // passes because Levenshtein distance is 1 (substitute 'o' with 'a')
        {
            let term = Term::from_field_text(country_field, "japon");
            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(2))?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            let (score, _) = top_docs[0];
            assert_nearly_equals!(1.0, score);
        }

        // fails because non-prefix Levenshtein distance is more than 1 (add 'a' and 'n')
        {
            let term = Term::from_field_text(country_field, "jap");

            let fuzzy_query = FuzzyTermQuery::new(term, 1, true);
            let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(2))?;
            assert_eq!(top_docs.len(), 0, "Expected no document");
        }

        // passes because prefix Levenshtein distance is 0
        {
            let term = Term::from_field_text(country_field, "jap");
            let fuzzy_query = FuzzyTermQuery::new_prefix(term, 1, true);
            let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(2))?;
            assert_eq!(top_docs.len(), 1, "Expected only 1 document");
            let (score, _) = top_docs[0];
            assert_nearly_equals!(1.0, score);
        }
        Ok(())
    }

    #[test]
    pub fn test_fuzzy_term_transposition_cost_one() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let country_field = schema_builder.add_text_field("country", TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(country_field => "japan"))?;
        index_writer.commit()?;
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let term_jaapn = Term::from_field_text(country_field, "jaapn");
        {
            let fuzzy_query_transposition = FuzzyTermQuery::new(term_jaapn.clone(), 1, true);
            let count = searcher.search(&fuzzy_query_transposition, &Count)?;
            assert_eq!(count, 1);
        }
        {
            let fuzzy_query_transposition = FuzzyTermQuery::new(term_jaapn, 1, false);
            let count = searcher.search(&fuzzy_query_transposition, &Count)?;
            assert_eq!(count, 0);
        }
        Ok(())
    }
}
