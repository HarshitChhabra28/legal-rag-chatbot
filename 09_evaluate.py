"""
Evaluation using querycleaned.xlsx as a held-out test set.

Part 1 - Retrieval accuracy (fast, automatic, no LLM call):
    For each of the 37 FAQ questions, check whether the FAQ's own indexed
    chunk appears in the top-k retrieved results (hybrid: vector + BM25 merged).
    This tests whether retrieval is fundamentally sound.

Part 2 - Answer quality (slower, needs the LLM):
    For a sample of N questions, run full generation and print the system's
    answer next to the known-correct answer for manual side-by-side review.

Usage:
    python 09_evaluate.py                # retrieval eval only (fast)
    python 09_evaluate.py --full 8       # also run full generation on 8 sample questions
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_step6_path = Path(__file__).parent / "database" / "06_hybrid_retrieve_and_generate.py"
_spec = importlib.util.spec_from_file_location("step6", _step6_path)
step6 = importlib.util.module_from_spec(_spec)
sys.modules["step6"] = step6
_spec.loader.exec_module(step6)

TOP_K = 6  # same as TOP_K_FINAL in step6, keep in sync


def load_query_records():
    with open("unified_records.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return [r for r in records if r["doc_type"] == "query"]


def evaluate_retrieval(query_records, embed_model, milvus_client, bm25, bm25_chunks):
    hits = 0
    misses = []

    for rec in query_records:
        question = rec["metadata"]["question"]
        own_chunk_id = f"{rec['id']}_0"

        vec_ranked = step6.vector_search(embed_model, milvus_client, question, k=TOP_K)
        bm25_ranked = step6.bm25_search(bm25, bm25_chunks, question, k=TOP_K)
        merged = step6.reciprocal_rank_fusion([vec_ranked, bm25_ranked], top_k=TOP_K)

        retrieved_ids = [c["chunk_id"] for c in merged]
        if own_chunk_id in retrieved_ids:
            hits += 1
        else:
            misses.append((rec["id"], question))

    total = len(query_records)
    print(f"\nRETRIEVAL EVAL: {hits}/{total} FAQ questions retrieved their own source chunk in top {TOP_K}")
    print(f"Hit rate: {hits/total:.1%}")

    if misses:
        print(f"\nMissed ({len(misses)}):")
        for chunk_id, q in misses:
            print(f"  [{chunk_id}] {q[:100]}")


def evaluate_answers(query_records, embed_model, milvus_client, bm25, bm25_chunks, n):
    sample = query_records[:n]
    print(f"\n{'='*80}\nANSWER QUALITY SAMPLE ({n} questions)\n{'='*80}")

    for rec in sample:
        question = rec["metadata"]["question"]
        known_answer = rec["metadata"]["answer"]

        result = step6.ask(question, embed_model, milvus_client, bm25, bm25_chunks)

        print(f"\n--- Q: {question}")
        print(f"KNOWN ANSWER:  {known_answer[:400]}")
        print(f"SYSTEM ANSWER: {result['answer']}")
        print("SOURCES:")
        for s in result["sources"]:
            print(f"  [{s['label']}] {s['citation']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=int, default=0,
                         help="also run full generation on N sample questions")
    args = parser.parse_args()

    query_records = load_query_records()
    print(f"Loaded {len(query_records)} FAQ records for evaluation")

    embed_model, milvus_client, bm25, bm25_chunks = step6.load_resources()

    evaluate_retrieval(query_records, embed_model, milvus_client, bm25, bm25_chunks)

    if args.full > 0:
        evaluate_answers(query_records, embed_model, milvus_client, bm25, bm25_chunks, args.full)


if __name__ == "__main__":
    main()