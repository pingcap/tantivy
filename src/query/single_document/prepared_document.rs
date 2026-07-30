use std::collections::HashMap;
use std::io;
use std::ops::Bound;
use std::sync::Arc;

use common::JsonPathWriter;
use stacker::Addr;

use super::{SingleDocument, SingleDocumentTermInfo};
use crate::fieldnorm::FieldNormReader;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::indexer::document_indexer::{
    index_document, text_analyzers_for_schema, DocumentIndexingMode, DocumentIndexingOutput,
};
use crate::indexer::path_to_unordered_id::OrderedPathId;
use crate::postings::{FieldSerializer, IndexingContext, PostingsWriter};
use crate::schema::{Document, Field, Schema, Type, JSON_END_OF_PATH};
use crate::tokenizer::{TextAnalyzer, TokenizerManager};
use crate::{DocId, TantivyError, Term};

/// Prepares regular [`Document`] values for single-document evaluation.
///
/// A preparer binds a schema and its indexing tokenizers once, then reuses the resulting text
/// analyzers and temporary buffers across calls to [`Self::prepare`]. Each returned
/// [`PreparedSingleDocument`] is owned and can be evaluated independently of the preparer.
///
/// Use the same schema and tokenizer configuration that would be used to index the documents.
pub struct SingleDocumentPreparer {
    schema: Schema,
    required_fields: Arc<[Field]>,
    per_field_text_analyzers: Vec<TextAnalyzer>,
    // These are per-document scratch objects, but keeping them on the reusable preparer preserves
    // their allocated capacity across `prepare` calls. `prepare` clears all document-local state
    // before returning, including on errors.
    term_buffer: Term,
    json_path_writer: JsonPathWriter,
    indexing_context: IndexingContext,
}

impl SingleDocumentPreparer {
    /// Creates a reusable preparer that only prepares `required_fields`.
    ///
    /// `required_fields` must cover every field that an evaluator may read. Incompatible reuse is
    /// rejected by [`SingleDocumentEvaluator::evaluate`](super::SingleDocumentEvaluator::evaluate).
    ///
    /// Every required field must occur at least once as a top-level field in each input document.
    /// Supply [`crate::schema::OwnedValue::Null`] when a required field has no value. The `Null`
    /// counts as explicitly supplied but is not sent through Tantivy's type-specific indexing
    /// logic and emits no term, token, position, or fieldnorm.
    pub fn for_fields(
        schema: &Schema,
        tokenizer_manager: &TokenizerManager,
        required_fields: &[Field],
    ) -> crate::Result<Self> {
        let mut required_fields = required_fields.to_vec();
        required_fields.sort_unstable();
        required_fields.dedup();
        for &field in &required_fields {
            if field.field_id() as usize >= schema.num_fields() {
                return Err(TantivyError::SchemaError(format!(
                    "Field id {} does not exist in the schema",
                    field.field_id()
                )));
            }
        }
        Self::with_required_fields(schema, tokenizer_manager, required_fields.into())
    }

    fn with_required_fields(
        schema: &Schema,
        tokenizer_manager: &TokenizerManager,
        required_fields: Arc<[Field]>,
    ) -> crate::Result<Self> {
        let per_field_text_analyzers = text_analyzers_for_schema(
            schema,
            tokenizer_manager,
            DocumentIndexingMode::QueryAware(required_fields.as_ref()),
        )?;
        Ok(Self {
            // `Schema` is backed by an `Arc`, so this clone only increments the reference count;
            // it does not copy the schema's fields or indexing options.
            schema: schema.clone(),
            required_fields,
            per_field_text_analyzers,
            term_buffer: Term::with_capacity(16),
            json_path_writer: JsonPathWriter::default(),
            indexing_context: IndexingContext::new(16),
        })
    }

    /// Prepares one document using Tantivy's indexing-time term and position semantics.
    pub fn prepare<D: Document>(&mut self, document: &D) -> crate::Result<PreparedSingleDocument> {
        let result = self.prepare_inner(document);

        // This output implementation does not write to the indexing arenas, so they can be reused.
        // JSON path ids are document-local and must be cleared before preparing the next document.
        debug_assert!(self.indexing_context.term_index.is_empty());
        debug_assert!(self.indexing_context.arena.is_empty());
        self.indexing_context.path_to_unordered_id.clear();
        self.json_path_writer.clear();
        result
    }

    fn prepare_inner<D: Document>(
        &mut self,
        document: &D,
    ) -> crate::Result<PreparedSingleDocument> {
        let mut provided_required_fields = vec![false; self.required_fields.len()];

        for (field, _value) in document.iter_fields_and_values() {
            let Ok(required_field_index) = self.required_fields.binary_search(&field) else {
                continue;
            };
            // Presence is independent from indexed output: a top-level Null satisfies this first
            // check and is filtered later by query-aware document indexing.
            provided_required_fields[required_field_index] = true;
        }

        // This is the first of two required-field checks. It catches an incomplete caller input at
        // preparation time. The prepared document separately records these fields as its static
        // coverage, and `SingleDocumentEvaluator::evaluate` checks that coverage again in case the
        // document is later reused with a broader evaluator.
        for (&field, &was_provided) in self.required_fields.iter().zip(&provided_required_fields) {
            if !was_provided {
                return Err(TantivyError::InvalidArgument(format!(
                    "SingleDocumentPreparer requires field {:?} to be explicitly supplied; use \
                     OwnedValue::Null when it has no value",
                    self.schema.get_field_entry(field).name()
                )));
            }
        }

        // Query-aware indexing selects `required_fields` and removes top-level Null values before
        // type-specific indexing. Nested Null values remain part of their JSON/array container.
        // The output is local because its maps are moved into the returned prepared document and
        // therefore cannot retain capacity for the next call.
        let mut output = PreparedDocumentIndexingOutput::default();
        index_document(
            0,
            document,
            &self.schema,
            DocumentIndexingMode::QueryAware(self.required_fields.as_ref()),
            &mut self.per_field_text_analyzers,
            &mut self.term_buffer,
            &mut self.json_path_writer,
            &mut self.indexing_context,
            &mut output,
        )?;
        output.normalize_json_terms(&self.indexing_context)?;
        Ok(output.take_prepared_document(Arc::clone(&self.required_fields)))
    }
}

/// An owned, indexed representation of one document for single-document evaluation.
///
/// `PreparedSingleDocument` applies the same tokenization, term encoding, multi-value position
/// gaps, and fieldnorm calculation as Tantivy's indexing pipeline. Preparing a document once and
/// reusing it avoids tokenizing the document again for every query.
///
/// # Example
///
/// ```
/// use tantivy::query::{
///     DocumentEvaluation, Query, SingleDocumentEvaluationContext, SingleDocumentPreparer,
///     TermQuery,
/// };
/// use tantivy::schema::{IndexRecordOption, Schema, TEXT};
/// use tantivy::{Index, TantivyDocument, Term};
///
/// # fn main() -> tantivy::Result<()> {
/// let mut schema_builder = Schema::builder();
/// let body = schema_builder.add_text_field("body", TEXT);
/// let schema = schema_builder.build();
/// let index = Index::create_in_ram(schema.clone());
/// let mut document = TantivyDocument::new();
/// document.add_text(body, "Rust search");
///
/// let query = TermQuery::new(
///     Term::from_field_text(body, "rust"),
///     IndexRecordOption::Basic,
/// );
/// let mut evaluator = query.single_document_evaluator(
///     SingleDocumentEvaluationContext::without_scoring(&schema),
/// )?;
/// let mut preparer =
///     SingleDocumentPreparer::for_fields(&schema, index.tokenizers(), &[body])?;
/// let prepared = preparer.prepare(&document)?;
/// assert_eq!(
///     evaluator.evaluate(&prepared)?,
///     DocumentEvaluation::Match(1.0)
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default)]
pub struct PreparedSingleDocument {
    term_positions: Vec<(Term, Vec<u32>)>,
    fieldnorm_ids: HashMap<Field, u8>,
    total_num_tokens_by_field: HashMap<Field, u64>,
    prepared_fields: Arc<[Field]>,
}

impl PreparedSingleDocument {
    /// Returns the exact number of indexed tokens for `field` in this document.
    ///
    /// Unlike [`Self::fieldnorm_id`], this value is not compressed. It can be
    /// accumulated across prepared documents to build corpus-level statistics
    /// such as the average field length used by BM25.
    pub fn total_num_tokens(&self, field: Field) -> u64 {
        self.total_num_tokens_by_field
            .get(&field)
            .copied()
            .unwrap_or(0)
    }
}

impl SingleDocument for PreparedSingleDocument {
    fn term_info(&self, term: &Term) -> Option<SingleDocumentTermInfo<'_>> {
        self.term_positions
            .binary_search_by(|(candidate, _)| candidate.cmp(term))
            .ok()
            .map(|term_index| &self.term_positions[term_index].1)
            .map(|positions| SingleDocumentTermInfo {
                term_freq: positions.len() as u32,
                positions: Some(positions.as_slice()),
            })
    }

    fn visit_terms(
        &self,
        field: Field,
        range: (Bound<&Term>, Bound<&Term>),
        visitor: &mut dyn FnMut(&Term, SingleDocumentTermInfo<'_>) -> bool,
    ) {
        let field_start = self
            .term_positions
            .partition_point(|(term, _)| term.field() < field);
        let field_end = field_start
            + self.term_positions[field_start..].partition_point(|(term, _)| term.field() == field);
        let field_terms = &self.term_positions[field_start..field_end];

        let range_start = match range.0 {
            Bound::Included(start) => field_terms.partition_point(|(term, _)| term < start),
            Bound::Excluded(start) => field_terms.partition_point(|(term, _)| term <= start),
            Bound::Unbounded => 0,
        };
        let range_end = match range.1 {
            Bound::Included(end) => field_terms.partition_point(|(term, _)| term <= end),
            Bound::Excluded(end) => field_terms.partition_point(|(term, _)| term < end),
            Bound::Unbounded => field_terms.len(),
        };
        if range_start >= range_end {
            return;
        }

        for (term, positions) in &field_terms[range_start..range_end] {
            if !visitor(
                term,
                SingleDocumentTermInfo {
                    term_freq: positions.len() as u32,
                    positions: Some(positions.as_slice()),
                },
            ) {
                break;
            }
        }
    }

    fn fieldnorm_id(&self, field: Field) -> Option<u8> {
        self.fieldnorm_ids.get(&field).copied()
    }

    fn validate_required_fields(&self, required_fields: Option<&[Field]>) -> crate::Result<()> {
        let Some(required_fields) = required_fields else {
            return Err(TantivyError::InvalidArgument(
                "PreparedSingleDocument was query-aware, but the evaluator's required fields are \
                 unknown"
                    .to_string(),
            ));
        };
        if let Some(field) = required_fields
            .iter()
            .find(|field| self.prepared_fields.binary_search(field).is_err())
        {
            return Err(TantivyError::InvalidArgument(format!(
                "PreparedSingleDocument did not prepare required field {field:?}"
            )));
        }
        Ok(())
    }
}

#[derive(Default)]
struct PreparedDocumentIndexingOutput {
    postings_writer: PreparedDocumentPostingsWriter,
    fieldnorm_ids: HashMap<Field, u8>,
}

impl PreparedDocumentIndexingOutput {
    fn normalize_json_terms(&mut self, indexing_context: &IndexingContext) -> crate::Result<()> {
        let unordered_to_ordered = indexing_context
            .path_to_unordered_id
            .unordered_id_to_ordered_id();
        let ordered_paths = indexing_context.path_to_unordered_id.ordered_id_to_path();
        let raw_term_positions = std::mem::take(&mut self.postings_writer.term_positions);

        for (term, mut positions) in raw_term_positions {
            let normalized_term = if term.typ() == Type::Json {
                normalize_json_term(term, &unordered_to_ordered, &ordered_paths)?
            } else {
                term
            };
            let normalized_positions = self
                .postings_writer
                .term_positions
                .entry(normalized_term)
                .or_default();
            if normalized_positions.is_empty() {
                normalized_positions.append(&mut positions);
            } else {
                normalized_positions.append(&mut positions);
                normalized_positions.sort_unstable();
            }
        }
        Ok(())
    }

    fn take_prepared_document(&mut self, prepared_fields: Arc<[Field]>) -> PreparedSingleDocument {
        let mut term_positions = std::mem::take(&mut self.postings_writer.term_positions)
            .into_iter()
            .collect::<Vec<_>>();
        term_positions.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
        let mut total_num_tokens_by_field = HashMap::new();
        for (term, positions) in &term_positions {
            *total_num_tokens_by_field.entry(term.field()).or_default() += positions.len() as u64;
        }
        self.postings_writer.total_num_tokens = 0;

        PreparedSingleDocument {
            term_positions,
            fieldnorm_ids: std::mem::take(&mut self.fieldnorm_ids),
            total_num_tokens_by_field,
            prepared_fields,
        }
    }
}

impl DocumentIndexingOutput for PreparedDocumentIndexingOutput {
    fn postings_writer(&mut self, _field: Field) -> &mut dyn PostingsWriter {
        &mut self.postings_writer
    }

    fn record_fieldnorm(&mut self, _doc: DocId, field: Field, fieldnorm: u32) {
        self.fieldnorm_ids
            .insert(field, FieldNormReader::fieldnorm_to_id(fieldnorm));
    }
}

fn normalize_json_term(
    raw_term: Term,
    unordered_to_ordered: &[OrderedPathId],
    ordered_paths: &[&str],
) -> crate::Result<Term> {
    let raw_value = raw_term.serialized_value_bytes();
    let unordered_id_bytes: [u8; 4] = raw_value
        .get(..5)
        .and_then(|bytes| bytes.get(..4))
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| {
            TantivyError::InternalError(
                "PreparedSingleDocument encountered an invalid intermediate JSON term".to_string(),
            )
        })?;
    let unordered_id = u32::from_be_bytes(unordered_id_bytes) as usize;
    let ordered_id = unordered_to_ordered.get(unordered_id).ok_or_else(|| {
        TantivyError::InternalError(
            "PreparedSingleDocument encountered an unknown JSON path id".to_string(),
        )
    })?;
    let path = ordered_paths
        .get(ordered_id.path_id() as usize)
        .ok_or_else(|| {
            TantivyError::InternalError(
                "PreparedSingleDocument encountered an unknown ordered JSON path id".to_string(),
            )
        })?;

    let mut term = Term::with_type_and_field(Type::Json, raw_term.field());
    term.append_path(path.as_bytes());
    term.append_bytes(&[JSON_END_OF_PATH]);
    term.append_bytes(&raw_value[4..]);
    Ok(term)
}

#[derive(Default)]
struct PreparedDocumentPostingsWriter {
    term_positions: HashMap<Term, Vec<u32>>,
    total_num_tokens: u64,
}

impl PostingsWriter for PreparedDocumentPostingsWriter {
    fn subscribe(&mut self, _doc: DocId, position: u32, term: &Term, _ctx: &mut IndexingContext) {
        if let Some(positions) = self.term_positions.get_mut(term) {
            positions.push(position);
        } else {
            // `Term` owns its byte buffer. Clone it only when inserting a new distinct term,
            // instead of cloning once for every occurrence before an `entry` lookup.
            self.term_positions.insert(term.clone(), vec![position]);
        }
        self.total_num_tokens += 1;
    }

    fn serialize(
        &self,
        _term_addrs: &[(Field, OrderedPathId, &[u8], Addr)],
        _ordered_id_to_path: &[&str],
        _doc_id_map: Option<&DocIdMapping>,
        _ctx: &IndexingContext,
        _serializer: &mut FieldSerializer,
    ) -> io::Result<()> {
        unreachable!("PreparedDocumentPostingsWriter cannot serialize postings")
    }

    fn total_num_tokens(&self) -> u64 {
        self.total_num_tokens
    }
}
