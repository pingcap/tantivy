use std::ops::Bound;

use super::{prefix_end, PhrasePrefixWeight};
use crate::query::bm25::Bm25Weight;
use crate::query::phrase_query::{intersection_count, PhraseMatcher};
use crate::query::single_document::{
    downgrade_record_option_for_term, effective_record_option, validate_term_info,
};
use crate::query::{
    DocumentEvaluation, EnableScoring, Query, RangeQuery, SingleDocument,
    SingleDocumentEvaluationContext, SingleDocumentEvaluator, Weight,
};
use crate::schema::{Field, IndexRecordOption, Term};
use crate::TantivyError;

const DEFAULT_MAX_EXPANSIONS: u32 = 50;

/// `PhrasePrefixQuery` matches a specific sequence of words followed by term of which only a
/// prefix is known.
///
/// For instance the phrase prefix query for `"part t"` will match
/// the sentence
///
/// **Alan just got a part time job.**
///
/// On the other hand it will not match the sentence.
///
/// **This is my favorite part of the job.**
///
/// Using a `PhrasePrefixQuery` on a field requires positions
/// to be indexed for this field.
#[derive(Clone, Debug)]
pub struct PhrasePrefixQuery {
    field: Field,
    phrase_terms: Vec<(usize, Term)>,
    prefix: (usize, Term),
    max_expansions: u32,
}

impl PhrasePrefixQuery {
    /// Creates a new `PhrasePrefixQuery` given a list of terms.
    ///
    /// There must be at least two terms, and all terms
    /// must belong to the same field.
    /// Offset for each term will be same as index in the Vector
    /// The last Term is a prefix and not a full value
    pub fn new(terms: Vec<Term>) -> PhrasePrefixQuery {
        let terms_with_offset = terms.into_iter().enumerate().collect();
        PhrasePrefixQuery::new_with_offset(terms_with_offset)
    }

    /// Creates a new `PhrasePrefixQuery` given a list of terms and their offsets.
    ///
    /// Can be used to provide custom offset for each term.
    pub fn new_with_offset(mut terms: Vec<(usize, Term)>) -> PhrasePrefixQuery {
        assert!(
            !terms.is_empty(),
            "A phrase prefix query is required to have at least one term."
        );
        terms.sort_by_key(|&(offset, _)| offset);
        let field = terms[0].1.field();
        assert!(
            terms[1..].iter().all(|term| term.1.field() == field),
            "All terms from a phrase query must belong to the same field"
        );
        PhrasePrefixQuery {
            field,
            prefix: terms.pop().unwrap(),
            phrase_terms: terms,
            max_expansions: DEFAULT_MAX_EXPANSIONS,
        }
    }

    /// Maximum number of terms to which the last provided term will expand.
    pub fn set_max_expansions(&mut self, value: u32) {
        self.max_expansions = value;
    }

    /// The [`Field`] this `PhrasePrefixQuery` is targeting.
    pub fn field(&self) -> Field {
        self.field
    }

    /// `Term`s in the phrase without the associated offsets.
    pub fn phrase_terms(&self) -> Vec<Term> {
        // TODO should we include the last term too?
        self.phrase_terms
            .iter()
            .map(|(_, term)| term.clone())
            .collect::<Vec<Term>>()
    }

    /// Returns the [`PhrasePrefixWeight`] for the given phrase query given a specific `searcher`.
    ///
    /// This function is the same as [`Query::weight()`] except it returns
    /// a specialized type [`PhraseQueryWeight`] instead of a Boxed trait.
    /// If the query was only one term long, this returns `None` wherease [`Query::weight`]
    /// returns a boxed [`RangeWeight`]
    pub(crate) fn phrase_prefix_query_weight(
        &self,
        enable_scoring: EnableScoring<'_>,
    ) -> crate::Result<Option<PhrasePrefixWeight>> {
        if self.phrase_terms.is_empty() {
            return Ok(None);
        }
        let schema = enable_scoring.schema();
        let field_entry = schema.get_field_entry(self.field);
        let has_positions = field_entry
            .field_type()
            .get_index_record_option()
            .map(IndexRecordOption::has_positions)
            .unwrap_or(false);
        if !has_positions {
            let field_name = field_entry.name();
            return Err(crate::TantivyError::SchemaError(format!(
                "Applied phrase query on field {field_name:?}, which does not have positions \
                 indexed"
            )));
        }
        let terms = self.phrase_terms();
        let bm25_weight_opt = match enable_scoring {
            EnableScoring::Enabled { searcher, .. } => {
                Some(Bm25Weight::for_terms(searcher, &terms)?)
            }
            EnableScoring::Disabled { .. } => None,
        };
        let weight = PhrasePrefixWeight::new(
            self.phrase_terms.clone(),
            self.prefix.clone(),
            bm25_weight_opt,
            self.max_expansions,
        );
        Ok(Some(weight))
    }
}

impl Query for PhrasePrefixQuery {
    /// Create the weight associated with a query.
    ///
    /// See [`Weight`].
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        if let Some(phrase_weight) = self.phrase_prefix_query_weight(enable_scoring)? {
            Ok(Box::new(phrase_weight))
        } else {
            // There are no prefix. Let's just match the suffix.
            let end_term =
                if let Some(end_value) = prefix_end(self.prefix.1.serialized_value_bytes()) {
                    let mut end_term = Term::with_capacity(end_value.len());
                    end_term.set_field_and_type(self.field, self.prefix.1.typ());
                    end_term.append_bytes(&end_value);
                    Bound::Excluded(end_term)
                } else {
                    Bound::Unbounded
                };

            let mut range_query = RangeQuery::new_term_bounds(
                enable_scoring
                    .schema()
                    .get_field_name(self.field)
                    .to_owned(),
                self.prefix.1.typ(),
                &Bound::Included(self.prefix.1.clone()),
                &end_term,
            );
            range_query.limit(self.max_expansions as u64);
            range_query.weight(enable_scoring)
        }
    }

    fn single_document_evaluator(
        &self,
        context: SingleDocumentEvaluationContext<'_>,
    ) -> crate::Result<Box<dyn SingleDocumentEvaluator>> {
        if self.field.field_id() as usize >= context.schema().num_fields() {
            return Err(TantivyError::SchemaError(format!(
                "Field id {} does not exist in the schema",
                self.field.field_id()
            )));
        }
        let field_entry = context.schema().get_field_entry(self.field);
        let field_type = field_entry.field_type();
        let value_type = field_type.value_type();
        for (_, term) in self
            .phrase_terms
            .iter()
            .chain(std::iter::once(&self.prefix))
        {
            if value_type != term.typ() {
                return Err(TantivyError::SchemaError(format!(
                    "Create a phrase prefix query of the type {:?}, when the field given was of \
                     type {value_type:?}",
                    term.typ()
                )));
            }
        }

        if self.phrase_terms.is_empty() {
            // Prefix-only evaluation only needs term presence (Basic). Validate schema
            // compatibility at compile time; the returned record option and fieldnorm flag
            // are unused because this path scores with `prefix_only_score`, not BM25.
            effective_record_option(context.schema(), &self.prefix.1, IndexRecordOption::Basic)?;
            return Ok(Box::new(PhrasePrefixSingleDocumentEvaluator {
                field: self.field,
                phrase_terms: Vec::new(),
                prefix: self.prefix.1.clone(),
                max_expansions: self.max_expansions,
                fixed_position_target: 0,
                adjusted_positions: Vec::new(),
                matcher: None,
                suffix_positions: Vec::new(),
                bm25_weight: None,
                has_fieldnorms: false,
                prefix_only_score: if context.is_scoring_enabled() {
                    context.boost()
                } else {
                    1.0
                },
            }));
        }

        let schema_record_option = field_type.index_record_option().ok_or_else(|| {
            TantivyError::SchemaError(format!("Field {:?} is not indexed.", field_entry.name()))
        })?;
        if !schema_record_option.has_positions() {
            return Err(TantivyError::SchemaError(format!(
                "Applied phrase query on field {:?}, which does not have positions indexed",
                field_entry.name()
            )));
        }
        for (_, term) in self
            .phrase_terms
            .iter()
            .chain(std::iter::once(&self.prefix))
        {
            let record_option = downgrade_record_option_for_term(
                field_type,
                term,
                IndexRecordOption::WithFreqsAndPositions,
                schema_record_option,
            );
            if !record_option.has_positions() {
                return Err(TantivyError::UnsupportedQueryForSingleDocumentEvaluation(
                    format!("PhrasePrefixQuery term {term:?} does not have position postings"),
                ));
            }
        }

        // Mirror the existing segment-index `PhrasePrefixScorer` behavior rather than define a new
        // single-document scoring policy. `PhraseScorer` requires at least two fixed-term postings
        // because its underlying `Intersection` does, so the segment path specializes the
        // one-fixed-term case: it shifts that term directly to the prefix offset and returns a
        // constant score. The multi-fixed-term branch delegates to `PhraseScorer`, which aligns
        // the fixed phrase one position after its greatest offset and BM25-scores it; the expanded
        // prefix only filters matches. This scoring split is specific to Tantivy's current
        // implementation: `PhrasePrefixScorer::score` still has a TODO to revisit prefix scoring,
        // so preserve it here for compatibility rather than treating it as general phrase-prefix
        // semantics.
        let has_multiple_fixed_terms = self.phrase_terms.len() > 1;
        let bm25_weight = if has_multiple_fixed_terms {
            context
                .statistics_provider()
                .map(|statistics| -> crate::Result<Bm25Weight> {
                    Ok(Bm25Weight::for_terms(statistics, &self.phrase_terms())?
                        .boost_by(context.boost()))
                })
                .transpose()?
        } else {
            None
        };
        // `new_with_offset` sorts by offset and removes the greatest entry as the prefix, so the
        // suffix is already at the maximum query offset and does not need an additional shift.
        let max_fixed_offset = self.phrase_terms.last().unwrap().0;
        let fixed_position_target = if has_multiple_fixed_terms {
            max_fixed_offset.checked_add(1).ok_or_else(|| {
                TantivyError::InvalidArgument(
                    "PhrasePrefixQuery position offset exceeds usize::MAX".to_string(),
                )
            })?
        } else {
            self.prefix.0
        };
        Ok(Box::new(PhrasePrefixSingleDocumentEvaluator {
            field: self.field,
            phrase_terms: self.phrase_terms.clone(),
            prefix: self.prefix.1.clone(),
            max_expansions: self.max_expansions,
            fixed_position_target,
            adjusted_positions: vec![Vec::new(); self.phrase_terms.len()],
            matcher: has_multiple_fixed_terms
                .then(|| PhraseMatcher::new(self.phrase_terms.len(), 0)),
            suffix_positions: Vec::new(),
            bm25_weight,
            has_fieldnorms: field_entry.has_fieldnorms(),
            prefix_only_score: 1.0,
        }))
    }

    fn query_terms<'a>(&'a self, visitor: &mut dyn FnMut(&'a Term, bool)) {
        for (_, term) in &self.phrase_terms {
            visitor(term, true);
        }
    }
}

struct PhrasePrefixSingleDocumentEvaluator {
    field: Field,
    phrase_terms: Vec<(usize, Term)>,
    prefix: Term,
    max_expansions: u32,
    fixed_position_target: usize,
    adjusted_positions: Vec<Vec<u32>>,
    matcher: Option<PhraseMatcher>,
    suffix_positions: Vec<u32>,
    bm25_weight: Option<Bm25Weight>,
    has_fieldnorms: bool,
    prefix_only_score: crate::Score,
}

impl PhrasePrefixSingleDocumentEvaluator {
    fn load_fixed_positions(&mut self, document: &dyn SingleDocument) -> crate::Result<bool> {
        for (term_index, (offset, term)) in self.phrase_terms.iter().enumerate() {
            let Some(term_info) = document.term_info(term) else {
                return Ok(false);
            };
            let adjusted = &mut self.adjusted_positions[term_index];
            adjusted.clear();
            validated_positions(
                term,
                term_info,
                self.fixed_position_target - offset,
                adjusted,
            )?;
        }
        Ok(true)
    }

    fn load_suffix_positions(&mut self, document: &dyn SingleDocument) -> crate::Result<bool> {
        self.suffix_positions.clear();
        let mut num_expansions = 0u32;
        let mut error = None;
        #[cfg(debug_assertions)]
        let mut previous_term = None;
        let prefix_bytes = self.prefix.serialized_value_bytes();
        let end_term = prefix_end(prefix_bytes).map(|end_value| {
            let mut end_term = Term::with_capacity(end_value.len());
            end_term.set_field_and_type(self.field, self.prefix.typ());
            end_term.append_bytes(&end_value);
            end_term
        });
        let range_end = end_term
            .as_ref()
            .map(Bound::Excluded)
            .unwrap_or(Bound::Unbounded);
        let needs_positions = !self.phrase_terms.is_empty();
        let mut visitor = |term: &Term, term_info: crate::query::SingleDocumentTermInfo<'_>| {
            #[cfg(debug_assertions)]
            debug_assert_visit_terms_order(&mut previous_term, term);
            if error.is_some() {
                return false;
            }
            if num_expansions >= self.max_expansions {
                // Keep visiting in debug builds so an ordering violation after the expansion limit
                // is not hidden by the early exit that release builds retain.
                #[cfg(debug_assertions)]
                return true;
                #[cfg(not(debug_assertions))]
                return false;
            }
            if !term.serialized_value_bytes().starts_with(prefix_bytes) {
                return true;
            }
            num_expansions += 1;
            if !needs_positions {
                if let Err(current_error) = validate_term_info(term, term_info.term_freq) {
                    error = Some(current_error);
                    return false;
                }
                return true;
            }
            if let Err(current_error) =
                validated_positions(term, term_info, 0, &mut self.suffix_positions)
            {
                error = Some(current_error);
                return false;
            }
            true
        };
        document.visit_terms(
            self.field,
            (Bound::Included(&self.prefix), range_end),
            &mut visitor,
        );
        if let Some(error) = error {
            return Err(error);
        }
        self.suffix_positions.sort_unstable();
        Ok(num_expansions > 0)
    }
}

#[cfg(debug_assertions)]
fn debug_assert_visit_terms_order(previous_term: &mut Option<Term>, term: &Term) {
    if let Some(previous_term) = previous_term.as_ref() {
        debug_assert!(
            previous_term < term,
            "SingleDocument::visit_terms must visit distinct terms in strictly ascending order; \
             previous term {previous_term:?}, current term {term:?}"
        );
    }
    *previous_term = Some(term.clone());
}

impl SingleDocumentEvaluator for PhrasePrefixSingleDocumentEvaluator {
    fn evaluate_impl(
        &mut self,
        document: &dyn SingleDocument,
    ) -> crate::Result<DocumentEvaluation> {
        if !self.load_suffix_positions(document)? {
            return Ok(DocumentEvaluation::NoMatch);
        }
        if self.phrase_terms.is_empty() {
            return Ok(DocumentEvaluation::Match(self.prefix_only_score));
        }
        if !self.load_fixed_positions(document)? {
            return Ok(DocumentEvaluation::NoMatch);
        }

        let phrase_count;
        let fixed_matches = if let Some(matcher) = &mut self.matcher {
            let adjusted_positions = &self.adjusted_positions;
            let load_positions = |index: usize, output: &mut Vec<u32>| {
                output.clear();
                output.extend_from_slice(&adjusted_positions[index]);
            };
            phrase_count = if self.bm25_weight.is_some() {
                matcher.phrase_count(load_positions)
            } else if matcher.phrase_exists(load_positions) {
                1
            } else {
                0
            };
            matcher.get_intersection()
        } else {
            phrase_count = self.adjusted_positions[0].len() as u32;
            &self.adjusted_positions[0]
        };
        if phrase_count == 0 || intersection_count(fixed_matches, &self.suffix_positions) == 0 {
            return Ok(DocumentEvaluation::NoMatch);
        }

        let Some(bm25_weight) = &self.bm25_weight else {
            return Ok(DocumentEvaluation::Match(1.0));
        };
        let fieldnorm_id = if self.has_fieldnorms {
            document.fieldnorm_id(self.field).ok_or_else(|| {
                TantivyError::InvalidArgument(format!(
                    "SingleDocument did not supply a fieldnorm for field {:?}",
                    self.field
                ))
            })?
        } else {
            1
        };
        Ok(DocumentEvaluation::Match(
            bm25_weight.score(fieldnorm_id, phrase_count),
        ))
    }

    fn required_fields(&self) -> Option<&[Field]> {
        Some(std::slice::from_ref(&self.field))
    }
}

fn validated_positions(
    term: &Term,
    term_info: crate::query::SingleDocumentTermInfo<'_>,
    position_offset: usize,
    output: &mut Vec<u32>,
) -> crate::Result<()> {
    validate_term_info(term, term_info.term_freq)?;
    let positions = term_info.positions.ok_or_else(|| {
        TantivyError::InvalidArgument(format!(
            "SingleDocument did not supply positions for {term:?}"
        ))
    })?;
    if positions.len() != term_info.term_freq as usize {
        return Err(TantivyError::InvalidArgument(format!(
            "SingleDocument supplied {} positions for term frequency {} and term {term:?}",
            positions.len(),
            term_info.term_freq
        )));
    }
    if positions.windows(2).any(|window| window[0] > window[1]) {
        return Err(TantivyError::InvalidArgument(format!(
            "SingleDocument positions are not sorted for {term:?}"
        )));
    }
    let position_offset = u32::try_from(position_offset).map_err(|_| {
        TantivyError::InvalidArgument(format!(
            "PhrasePrefixQuery position offset exceeds u32::MAX for {term:?}"
        ))
    })?;
    if let Some(position) = positions.last() {
        if position.checked_add(position_offset).is_none() {
            return Err(TantivyError::InvalidArgument(format!(
                "PhrasePrefixQuery adjusted position exceeds u32::MAX for {term:?}"
            )));
        }
    }
    output.reserve(positions.len());
    output.extend(positions.iter().map(|position| position + position_offset));
    Ok(())
}
