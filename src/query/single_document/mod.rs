//! Evaluate queries against pre-tokenized external documents.
//!
//! # Single-document evaluation
//!
//! [`Query::single_document_evaluator`](crate::query::Query::single_document_evaluator) evaluates
//! pre-tokenized documents without first adding them to an index. Its built-in query support is:
//!
//! - Directly supported: `AllQuery`, `EmptyQuery`, `FuzzyTermQuery`, and `TermQuery`.
//!   `FuzzyTermQuery` and `TermQuery` require the term's field to be configured as indexed in the
//!   schema. Although evaluation does not read an index, it uses the field's postings and fieldnorm
//!   settings to match segment scorer semantics.
//! - Supported with constraints:
//!   - `PhraseQuery` requires positions from both the field and the `SingleDocument` input.
//!   - `PhrasePrefixQuery` additionally requires [`SingleDocument::visit_terms`] to expose indexed
//!     terms in requested ranges in sorted order. Its `max_expansions` limit applies to matching
//!     terms in the evaluated document because no segment term dictionary is available. This can
//!     differ from segment evaluation when the limit is reached. For example, set `max_expansions`
//!     to `1`. A segment dictionary containing `[ca, cb]` expands `c*` to `ca`, while a document
//!     containing only `cb` expands it to `cb` during single-document evaluation. A query such as
//!     `"x c*"` can therefore match an external `"x cb"` document even though the segment scorer
//!     would not.
//! - Composite queries: `BooleanQuery` requires every evaluated child query to be supported.
//! - `BooleanQuery` compilation skips `Should` clauses when scoring is disabled and at least one
//!   `Must` clause exists, because `Should` cannot affect the match in that case. An unsupported
//!   query in a skipped `Should` clause therefore does not cause compilation to fail.
//! - Wrapper queries: `BoostQuery` and `ConstScoreQuery` require the wrapped query to be supported.
//! - Common unsupported queries: `DisjunctionMaxQuery`, `TermSetQuery`, `RegexQuery`, `RangeQuery`,
//!   `ExistsQuery`, and `MoreLikeThisQuery`.
//!
//! The unsupported list is not exhaustive. Any query implementation that does not override
//! [`Query::single_document_evaluator`](crate::query::Query::single_document_evaluator) returns
//! [`TantivyError::UnsupportedQueryForSingleDocumentEvaluation`](crate::TantivyError::UnsupportedQueryForSingleDocumentEvaluation).
//!
//! # Caller responsibilities
//!
//! Single-document evaluation does not validate every numeric input the way index search might
//! implicitly rely on a [`Searcher`](crate::Searcher). Callers must supply values that satisfy the
//! same invariants as index-side BM25 construction:
//!
//! - [`Bm25StatisticsProvider`](crate::query::Bm25StatisticsProvider): `total_num_docs > 0`,
//!   positive field token counts, and `doc_freq <= total_num_docs` for every scored term.
//! - Query scores and boosts (`BoostQuery`, `ConstScoreQuery`, and accumulated
//!   [`SingleDocumentEvaluationContext`](crate::query::SingleDocumentEvaluationContext) boosts):
//!   finite values. Non-finite inputs may produce non-finite scores or panic inside BM25 code.
//! - [`SingleDocument`](crate::query::SingleDocument) document data: see trait docs below.
//!   Malformed term frequencies, positions, or missing fieldnorms return `InvalidArgument` at
//!   evaluation time.
//!
//! Callers with regular [`Document`](crate::schema::Document) values should create a query-aware
//! [`SingleDocumentPreparer`] with [`SingleDocumentPreparer::for_fields`], prepare each document
//! with [`SingleDocumentPreparer::prepare`], and pass the result to
//! [`SingleDocumentEvaluator::evaluate`]. Query-aware preparation only indexes the explicitly
//! listed fields, which must cover every field required by the evaluator. Every required top-level
//! field must occur at least once in the input document. Use
//! [`OwnedValue::Null`](crate::schema::OwnedValue::Null) to explicitly represent a required field
//! that has no value. A top-level `Null` satisfies the presence check but emits no indexed data.

use std::ops::Bound;

mod prepared_document;

pub use self::prepared_document::{PreparedSingleDocument, SingleDocumentPreparer};
use crate::query::Bm25StatisticsProvider;
use crate::schema::{Field, FieldType, IndexRecordOption, Schema, Type};
use crate::{Score, TantivyError, Term};

/// Occurrence information for one term in one external document.
#[derive(Clone, Copy, Debug)]
pub struct SingleDocumentTermInfo<'a> {
    /// Number of occurrences written to Tantivy postings. This must be greater than zero.
    pub term_freq: u32,
    /// Tantivy postings positions for this term.
    ///
    /// Positions are required only when evaluating a [`PhraseQuery`](crate::query::PhraseQuery) or
    /// [`PhrasePrefixQuery`](crate::query::PhrasePrefixQuery). Other supported query types ignore
    /// them, so callers may use `None`. When required, positions must be sorted absolute token
    /// positions and their length must equal `term_freq`.
    pub positions: Option<&'a [u32]>,
}

/// A pre-tokenized view of one external document.
///
/// Positions may be omitted except when evaluating a [`PhraseQuery`](crate::query::PhraseQuery) or
/// [`PhrasePrefixQuery`](crate::query::PhrasePrefixQuery). Phrase positions use Tantivy postings
/// semantics: they are sorted absolute token positions including indexing-time position gaps or
/// offsets, and must not have phrase-query offsets applied in advance. Their length must equal
/// `term_freq`.
///
/// [`PhrasePrefixQuery`](crate::query::PhrasePrefixQuery) additionally requires
/// [`Self::visit_terms`] to expose every indexed term in the requested range in ascending [`Term`]
/// order.
pub trait SingleDocument {
    /// Returns occurrence information for `term`, or `None` when the term is absent.
    fn term_info(&self, term: &Term) -> Option<SingleDocumentTermInfo<'_>>;

    /// Visits every indexed term for `field` within `range` in ascending [`Term`] order.
    ///
    /// Bounds use the natural [`Term`] ordering. Each distinct term in the range must be visited
    /// exactly once unless `visitor` returns `false` to stop iteration early. The supplied term
    /// information follows the same invariants as [`Self::term_info`].
    /// [`PhrasePrefixQuery`](crate::query::PhrasePrefixQuery) single-document evaluation
    /// debug-asserts the strict ordering requirement. Automaton and prefix queries use this method
    /// because they cannot know all matching terms in advance.
    fn visit_terms(
        &self,
        field: Field,
        range: (Bound<&Term>, Bound<&Term>),
        visitor: &mut dyn FnMut(&Term, SingleDocumentTermInfo<'_>) -> bool,
    );

    /// Returns the compressed fieldnorm id for `field`.
    ///
    /// `None` means the caller did not supply a fieldnorm for this field. A fieldnorm is required
    /// only when scoring a [`TermQuery`](crate::query::TermQuery),
    /// [`PhraseQuery`](crate::query::PhraseQuery), or
    /// [`PhrasePrefixQuery`](crate::query::PhrasePrefixQuery) on a field configured with
    /// fieldnorms.
    fn fieldnorm_id(&self, field: Field) -> Option<u8>;

    /// Validates that this document contains the fields an evaluator may read.
    ///
    /// Implementations that always expose a complete document can use the default implementation.
    /// Query-aware implementations should reject required fields that were not prepared. When
    /// `required_fields` is `None`, the evaluator's requirements are unknown and only a complete
    /// document is safe.
    fn validate_required_fields(&self, _required_fields: Option<&[Field]>) -> crate::Result<()> {
        Ok(())
    }
}

/// Controls whether a single-document evaluator computes scores.
#[derive(Clone, Copy)]
pub enum SingleDocumentScoring<'a> {
    /// Compute scores using the supplied corpus statistics.
    Enabled(&'a dyn Bm25StatisticsProvider),
    /// Only compute the match state.
    Disabled,
}

/// Query compilation context for single-document evaluation.
///
/// When scoring is enabled, callers must supply a [`Bm25StatisticsProvider`] and query parameters
/// that satisfy the invariants documented in the [`single_document`](crate::query::single_document)
/// module.
#[derive(Clone, Copy)]
pub struct SingleDocumentEvaluationContext<'a> {
    schema: &'a Schema,
    scoring: SingleDocumentScoring<'a>,
    boost: Score,
}

impl<'a> SingleDocumentEvaluationContext<'a> {
    /// Creates a context that computes scores with `statistics`.
    ///
    /// `statistics` must satisfy the BM25 invariants described in the
    /// [`single_document`](crate::query::single_document) module. Invalid statistics may panic
    /// during compilation.
    pub fn with_scoring(schema: &'a Schema, statistics: &'a dyn Bm25StatisticsProvider) -> Self {
        Self {
            schema,
            scoring: SingleDocumentScoring::Enabled(statistics),
            boost: 1.0,
        }
    }

    /// Creates a context that only computes the match state.
    pub fn without_scoring(schema: &'a Schema) -> Self {
        Self {
            schema,
            scoring: SingleDocumentScoring::Disabled,
            boost: 1.0,
        }
    }

    /// Returns the schema used to interpret query terms.
    pub fn schema(&self) -> &Schema {
        self.schema
    }

    /// Returns the statistics provider when scoring is enabled.
    pub fn statistics_provider(&self) -> Option<&dyn Bm25StatisticsProvider> {
        match self.scoring {
            SingleDocumentScoring::Enabled(statistics) => Some(statistics),
            SingleDocumentScoring::Disabled => None,
        }
    }

    /// Returns whether scoring is enabled.
    pub fn is_scoring_enabled(&self) -> bool {
        matches!(self.scoring, SingleDocumentScoring::Enabled(_))
    }

    pub(crate) fn boost(&self) -> Score {
        self.boost
    }

    /// Multiplies the accumulated boost factor.
    ///
    /// `factor` must be finite. This function does not validate the input.
    pub(crate) fn with_boost(self, factor: Score) -> Self {
        Self {
            boost: self.boost * factor,
            ..self
        }
    }

    pub(crate) fn scoring_disabled(self) -> Self {
        Self {
            scoring: SingleDocumentScoring::Disabled,
            ..self
        }
    }
}

/// Match and score result for one external document.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DocumentEvaluation {
    /// The document matched. The payload is its score.
    Match(Score),
    /// The document did not match.
    NoMatch,
}

/// A query compiled for repeated evaluation of external documents.
///
/// An evaluator may keep and reuse mutable scratch buffers between calls. It can be reused
/// sequentially and moved to a worker thread, but this trait does not require `Clone` or `Sync`.
/// Compile a separate evaluator per worker when evaluating documents concurrently.
pub trait SingleDocumentEvaluator: Send {
    /// Evaluates one pre-tokenized document.
    ///
    /// This is the checked public entry point. Evaluator implementations provide their query
    /// logic in [`Self::evaluate_impl`].
    fn evaluate(&mut self, document: &dyn SingleDocument) -> crate::Result<DocumentEvaluation> {
        // This is the second of two required-field checks. Query-aware preparation first verifies
        // that its input explicitly supplies every field requested by the evaluator it was built
        // for. Evaluation checks the resulting coverage again because a prepared document can be
        // retained and accidentally passed to a different, broader evaluator. Without this check,
        // an unprepared field would be indistinguishable from a prepared field with no terms and
        // could silently turn a match into NoMatch.
        document.validate_required_fields(self.required_fields())?;
        self.evaluate_impl(document)
    }

    /// Implements the evaluator-specific match and score logic.
    ///
    /// Callers should use [`Self::evaluate`] so that prepared-field coverage is validated first.
    fn evaluate_impl(&mut self, document: &dyn SingleDocument)
        -> crate::Result<DocumentEvaluation>;

    /// Returns the fields whose indexed data may be read by this evaluator.
    ///
    /// `Some(fields)` enables query-aware document preparation. `Some(&[])` means that the
    /// evaluator does not inspect document fields. The default `None` means that the requirements
    /// are unknown, so the evaluator cannot safely evaluate documents prepared for an explicit
    /// field subset by [`SingleDocumentPreparer::for_fields`]. The returned set must remain stable
    /// for the lifetime of the evaluator.
    fn required_fields(&self) -> Option<&[Field]> {
        None
    }
}

pub(crate) fn validate_term_info(term: &Term, term_freq: u32) -> crate::Result<()> {
    if term_freq == 0 {
        return Err(TantivyError::InvalidArgument(format!(
            "SingleDocument returned a zero term frequency for {term:?}"
        )));
    }
    Ok(())
}

/// Prepends a parent query path to unsupported-query errors.
///
/// Other error variants are intentionally left unchanged to preserve their classification.
pub(crate) fn with_unsupported_query_path(
    error: TantivyError,
    path: impl AsRef<str>,
) -> TantivyError {
    match error {
        TantivyError::UnsupportedQueryForSingleDocumentEvaluation(child_path) => {
            TantivyError::UnsupportedQueryForSingleDocumentEvaluation(format!(
                "{} -> {child_path}",
                path.as_ref()
            ))
        }
        other => other,
    }
}

/// Resolves the [`IndexRecordOption`] to use when reading postings for `term`
/// during single-document evaluation.
///
/// The query may request a record level (for example frequencies for BM25 or
/// positions for phrase matching), but the schema and the term itself may only
/// expose a lower level. This function validates the field and returns the
/// effective level together with whether the field stores fieldnorms.
pub(crate) fn effective_record_option(
    schema: &Schema,
    term: &Term,
    requested: IndexRecordOption,
) -> crate::Result<(IndexRecordOption, bool)> {
    let field = term.field();
    if field.field_id() as usize >= schema.num_fields() {
        return Err(TantivyError::SchemaError(format!(
            "Field id {} does not exist in the schema",
            field.field_id()
        )));
    }
    let field_entry = schema.get_field_entry(field);
    let field_type = field_entry.field_type();
    let Some(stored) = field_type.index_record_option() else {
        return Err(TantivyError::SchemaError(format!(
            "Field {:?} is not indexed.",
            field_entry.name()
        )));
    };

    let effective = downgrade_record_option_for_term(field_type, term, requested, stored);
    Ok((effective, field_entry.has_fieldnorms()))
}

/// Computes the highest [`IndexRecordOption`] supported both by `requested`
/// and by the postings stored for this term.
///
/// For JSON fields, only string terms retain the schema's full record level;
/// non-string JSON terms are capped at [`IndexRecordOption::Basic`].
pub(crate) fn downgrade_record_option_for_term(
    field_type: &FieldType,
    term: &Term,
    requested: IndexRecordOption,
    mut stored: IndexRecordOption,
) -> IndexRecordOption {
    if matches!(field_type, FieldType::JsonObject(_)) {
        let is_json_string = term
            .value()
            .as_json_value_bytes()
            .map(|json_value| json_value.typ() == Type::Str)
            .unwrap_or(false);
        if !is_json_string {
            stored = IndexRecordOption::Basic;
        }
    }

    requested.downgrade(stored)
}

#[cfg(test)]
mod tests;
