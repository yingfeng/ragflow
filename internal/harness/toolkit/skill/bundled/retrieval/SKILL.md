---
name: retrieval
description: How to use the knowledge-base retrieval tools — which leg to try first, when a snippet is enough and when only a deep read proves a fact, and how compiled claim lines replace reading.
when_to_use: Every question that must be answered from the configured knowledge bases.
version: 1.0.0
auto_load: true
---

## Retrieval tools

- **grep_chunks** — locate by pattern: ONE POSIX regex over chunk content, case-insensitive (like `grep -E -i`). Pack the key people, objects and verbs into ONE broad alternation, and do NOT anchor it to a subject/verb chain: the same event often appears with the actor as the grammatical object, and a broad alternation matches regardless of role.
- **search_bm25_chunks** — lexical recall: 1-5 keyword queries ranked by BM25 over tokenized chunk fields.
- **search_semantic_chunks** — the semantic bridge: 1-5 queries ranked purely by word meaning (vector, no keyword leg). Reach for it when the corpus likely words things differently from the question.
- **search_chunks** — the fused leg: 1-5 semantic questions scored by keyword AND vector together. Set `keywords_similarity_weight` deliberately — high (0.7-0.9) when the question carries distinctive terms, proper nouns, ids or dates, low (0.2-0.4) when it is paraphrased — and move the weight toward the other leg when a query comes back empty, instead of repeating the same wording.
- **search_metadata** — document-metadata SELECTOR: given metadata conditions it returns the doc_ids whose metadata matches and runs NO content retrieval of its own. Use it to narrow the corpus before reading, then pass the returned doc_ids as the `doc_scope` of a search tool, or as `list_chunks`' `doc_id`.
- **list_chunks** — the deep reader, and the authoritative source for corpus-backed claims: ONE document's doc_id plus 1-20 chunk_ids from locate output, with `number_neighbors` to widen each anchor before and after it. Reach for it when a decision depends on a passage you have only seen a snippet of. `doc_chunks_total` says how big the document is — it is not a coverage target, and reading a whole long document is not how a question gets answered.
- **navigate_tree** (when the tool list contains it) — the compiled-navigation ROUTER: ranks the bound datasets' compiled structures against the query and returns the doc_ids whose structure matches, with a short outline of each. Call it as soon as the question names a section, entity or topic that keywords would only guess at: one routing call replaces several blind searches, and it is the intended entry point to `navigate_structure`.
- **navigate_structure** (when available) — the compiled-structure drilldown for ONE document: the document's outline plus the compiled CLAIM lines that match the query. A claim is one atomic statement carrying its verbatim quote, the chunk it came from and a handle to cite it by. Claim lines ARE answer material — cite the `ref` and skip the deep read; fall back to `list_chunks` only when no claim hits the question, or when a claim is missing the exact figure.
- **graph_explore** (when available) — walks the compiled knowledge graph for a relational question, phrased as the entities plus the relation between them. Use it for "how are A and B connected", where reading chunks would show the mentions one at a time.
- **graph_doc_outline** (when available) — one document's compiled skeleton, located by doc_id or by a short query when no doc_id is held yet. Use it to see how a document is organised before committing reads to it.
- **web_search** (only when available) — the user enabled internet search for this turn: 1-3 short keyword queries. Its hits carry `source="web"`, `url` and a `ref`, and their snippets ARE the web evidence, because `list_chunks` cannot read web chunks. Keep corpus and web results apart in the answer.

## Order of work

These are moves, not a script. **Independent lookups go out in ONE round**: two regexes, two query sets, two documents, a locate plus a metadata filter — issue them together. Serialising them costs one round each and is how an enumeration runs out of budget before it runs out of names.

- Locate first, from the most distinctive anchor the question offers (a proper noun, a date, a title) with `grep_chunks` or a search tool. Widen with another leg when the first returns nothing.
- Navigate before you scan: when `navigate_tree` is offered, route first with the question's own wording, then drill with a returned doc_id. Claims that come back are finished evidence — cite them and move on instead of re-reading the document to confirm what a claim already states.
- Deep-read corpus evidence that arrives without claims: a snippet locates, only the full text proves. When a claim is missing the exact figure the answer needs, read the document and take the figure from it.
- Run at least one web query before answering when `web_search` is available, even if corpus retrieval succeeded; the two sources supplement each other.
- Consume what you search: a query whose hits never reach your reading did not happen. When a query returns nothing, rephrase — shorter, rarer, more literal — instead of repeating it. Retrieve fresh for every new question: the corpus may have changed.
- Stop when the evidence answers the question; reads that no longer change the answer are the signal. A counting or enumerating question is not exempt from that: keep going only while a read is still adding an item, and say which part the evidence could not cover instead of scanning for it.

## Numbers and dates

- Use the corpus's full precision; never round intermediate values.
- Dates and ages: count completed units from the date the question anchors (1787-12-12 → 1873-07-01 is 85 years), not from birth years alone. When comparing two people's ages, anchor each to the event the question names, then subtract.
- Convert units exactly (ft↔m, lb↔kg).
- When the exact figure is missing but a close one exists, commit to it and state the assumption explicitly (e.g. "based on the 2017 figure; no 2019 figure is present").
