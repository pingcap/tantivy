use tantivy::{
    fastfield::AliveBitSet,
    query::{EnableScoring, Query, Scorer},
    DocAddress,
    Result, // 添加了Result
    Searcher,
    SegmentOrdinal,
    TantivyDocument,
    TERMINATED,
};

pub struct SegQueryIter<'a> {
    // The ordinal (index) of the current segment within the index searcher.
    seg_ord: SegmentOrdinal,
    // Reference to the original searcher, used to retrieve documents.
    searcher: &'a Searcher,
    // A scorer that advances through matching document IDs.
    // Scorer guarantee to return document IDs in ascending order.
    scorer: Box<dyn Scorer>,
    // Set of alive DocIds.
    alive_bitset: Option<&'a AliveBitSet>,
}

impl<'a> SegQueryIter<'a> {
    pub fn new(seg_ord: SegmentOrdinal, searcher: &'a Searcher, query: &dyn Query) -> Result<Self> {
        let seg_reader = searcher.segment_reader(seg_ord);
        let alive_bitset = seg_reader.alive_bitset();
        let weight = query.weight(EnableScoring::Disabled {
            schema: seg_reader.schema(),
            searcher_opt: None,
        })?;
        let scorer = weight.scorer(seg_reader, 1.0)?;
        Ok(Self {
            seg_ord,
            searcher,
            scorer,
            alive_bitset,
        })
    }

    pub fn next_doc_id(&mut self) -> Option<u32> {
        let mut doc_id = self.scorer.doc();

        if let Some(alive_bitset) = self.alive_bitset {
            while doc_id != TERMINATED && alive_bitset.is_deleted(doc_id) {
                doc_id = self.scorer.advance();
            }
        }

        if doc_id == TERMINATED {
            return None;
        }
        // Make scorer point to next doc id.
        let _ = self.scorer.advance();

        Some(doc_id)
    }

    pub fn next_doc(&mut self) -> Option<TantivyDocument> {
        match self.next_doc_id() {
            None => None,
            Some(doc_id) => {
                let doc = self
                    .searcher
                    .doc(DocAddress {
                        segment_ord: self.seg_ord,
                        doc_id,
                    })
                    .unwrap();
                Some(doc)
            }
        }
    }
}
