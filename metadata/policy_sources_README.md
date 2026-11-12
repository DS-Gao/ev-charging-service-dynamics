# Policy-source metadata

`policy_sources.csv` is the reader-facing crosswalk for the government policy
records cited as `gov1`–`gov12` in the manuscript.  `P001`–`P012` are stable
short keys used in the tables; they do not replace the original bibliography
keys or the original document identifiers.

The fields have deliberately different roles:

- `source_doc_id` is the long identifier from the registered policy corpus.
- `source_fragment_id` is the stable evidence key for the retained source
  fragment (document plus local extraction coordinates and quoted text).
- `source_line_start` and `source_line_end` are coordinates in the local
  extraction used for coding.  They are not article, section or page numbers.
- `document_number` is the formal issuing number when it is recorded.  An
  unavailable number remains explicitly unavailable.
- `provision_reference` is the readable article, section or item reference. If
  the source table did not record a numbered clause, it is a descriptive
  provision locator rather than an invented article number.
- `official_url` is populated only for a document-level official URL checked
  against the title, document number and cited provision.  A blank URL means
  that no such URL was verified in the saved source metadata; it is not a claim
  that the document never had an official web page.
- `source_corpus_doi` identifies the registered corpus.  The corpus metadata
  is distinct from the underlying policy text, which may require PKULaw or
  another official access route.

The repository contains metadata and short excerpts only.  It does not
redistribute the policy corpus or subscription-restricted full text.  The
manuscript's formal citations remain the `gov*` bibliography entries; the
`P*` keys provide a compact link from the displayed provision to this
crosswalk.
