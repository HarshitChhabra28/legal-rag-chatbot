"""
Step 5: BM25 keyword index (for hybrid retrieval alongside Milvus).

Embeddings are strong on meaning, weak on exact tokens (section numbers,
citation strings, specific party names). BM25 catches those. At query time
(Step 6) we'll run both and merge rankings with reciprocal rank fusion.

Output:
    bm25_index.pkl   - the fitted BM25 index
    bm25_chunks.pkl  - parallel list of chunk dicts (same order as the index)
"""

import json
import pickle
import re
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def main():
    with open("chunks_with_citations.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    print(f"Loaded {len(chunks)} chunks")

    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open("bm25_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Saved bm25_index.pkl and bm25_chunks.pkl")

    # Sanity check: a query heavy on exact tokens (section number),
    # something embeddings alone tend to blur.
    test_query = "Section 441 compounding offence"
    scores = bm25.get_scores(tokenize(test_query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

    print(f"\nSanity-check BM25 search ('{test_query}'):")
    for i in top_idx:
        print(f"  score={scores[i]:.3f} [{chunks[i]['doc_type']}] {chunks[i]['title']}")


if __name__ == "__main__":
    main()