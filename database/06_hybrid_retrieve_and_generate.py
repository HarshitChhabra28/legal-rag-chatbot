import json
import pickle
import re
from pathlib import Path
import requests
from google import genai
import os
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DB_PATH = str(Path(__file__).resolve().parent.parent / "law_project.db")
COLLECTION_NAME = "legal_chunks"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"

TOP_K_EACH = 10   # how many results to pull from EACH retriever before merging
TOP_K_FINAL = 6   # how many merged chunks to actually send to Claude
RRF_K = 60        # standard RRF damping constant

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_resources():
    embed_model = SentenceTransformer(MODEL_NAME)
    milvus_client = MilvusClient(DB_PATH)
    milvus_client.load_collection(collection_name=COLLECTION_NAME)

    with open(Path(__file__).resolve().parent.parent / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(Path(__file__).resolve().parent.parent / "bm25_chunks.pkl", "rb") as f:
        bm25_chunks = pickle.load(f)

    return embed_model, milvus_client, bm25, bm25_chunks


def vector_search(embed_model, milvus_client, query, k=TOP_K_EACH):
    # bge models expect this instruction prefix on the QUERY side only
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vec = embed_model.encode([prefixed], normalize_embeddings=True)
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=vec.tolist(),
        limit=k,
        output_fields=["chunk_id", "doc_type", "title", "text", "citation"],
    )
    # returns list of (chunk_id, chunk_dict) ranked best-first
    ranked = []
    for hit in results[0]:
        entity = hit["entity"]
        ranked.append((entity["chunk_id"], {
            "chunk_id": entity["chunk_id"],
            "doc_type": entity["doc_type"],
            "title": entity["title"],
            "text": entity["text"],
            "citation": entity["citation"],
        }))
    return ranked


def bm25_search(bm25, bm25_chunks, query, k=TOP_K_EACH):
    scores = bm25.get_scores(tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    ranked = []
    for i in top_idx:
        c = bm25_chunks[i]
        ranked.append((c["chunk_id"], c))
    return ranked


def reciprocal_rank_fusion(ranked_lists, k=RRF_K, top_k=TOP_K_FINAL):
    """ranked_lists: list of ranked lists, each a list of (chunk_id, chunk_dict) best-first."""
    scores = {}
    chunk_lookup = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, chunk) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_lookup[chunk_id] = chunk

    merged_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    return [chunk_lookup[cid] for cid in merged_ids]


def build_prompt(query, chunks):
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] ({c['doc_type']}) {c['title']}\n{c['text']}")
    context_text = "\n\n".join(context_blocks)

    system = (
        "You are a legal research assistant. Answer the user's question using ONLY "
        " the numbered context blocks below. You MUST cite sources inline using ONLY "
        "the exact format [n] where n is the context block number (e.g. [1], [2]) - "
        "never add anything else inside or immediately after the brackets, such as "
        "paragraph numbers, notes, or a second bracket right after a case name. "
        "Use at most one bracket citation per sentence, placed at the end of that "
        "sentence. Every substantive claim in your answer must include at least one "
        "such citation - do not give an uncited answer even if it feels obvious from "
        "context. "
 
        "Be concise and decisive: give a direct answer in 2-4 sentences. Do not list "
        "every retrieved case one by one - synthesize them into a single coherent "
        "position. If the sources genuinely conflict, state the conflict in one "
        "sentence rather than hedging throughout. If the context does not contain "
        "enough information to answer, say so explicitly instead of guessing. Do not "
        "invent case names, citations, or facts not present in the context. "
 
        "Use only the minimum number of sources needed to answer the question. "
        "Do not mention additional cases merely because they are related. "
        "Prefer the most directly relevant source over loosely related sources. "
 
        "Do not answer from your general legal knowledge. "
        "Every legal proposition must be directly supported by the supplied context. "
        "If the context does not directly answer the question, say that the provided "
        "context is insufficient. Do not fill gaps using general legal principles. "
 
        "If the retrieved source answers the question only in a specific legal context, "
        "make that context explicit in the answer. Do not generalize a rule beyond what "
        "the source establishes. "
 
        "Do not add procedural, factual, or legal details merely because they are "
        "common, plausible, or generally known. In particular, when the user asks "
        "for a process, procedure, mechanism, requirements, steps, conditions, "
        "penalties, timelines, or consequences, mention only those details that are "
        "explicitly stated or clearly established by the supplied context. Do not "
        "infer missing steps or fill in procedural gaps from general legal knowledge. "
        "If the source provides only a high-level description, give only that "
        "high-level description and explicitly say that the provided context does "
        "not contain further procedural details. "
 
        "Special case for definitional questions: if the user asks what a term "
        "means and no retrieved source explicitly defines it, do NOT simply refuse. "
        "Instead, describe how the term is actually used across the retrieved "
        "sources - what procedures, decisions, approvals, or consequences are "
        "discussed in connection with it - and then explicitly state that no "
        "formal definition was found in the supplied context. This is still fully "
        "grounded: you are reporting what the sources say ABOUT the term, not "
        "supplying your own definition of it. "
 
        "Before using a retrieved case or article to answer, check whether it "
        "actually addresses the specific question asked, not just the general "
        "topic or area of law. A case about a different type of dispute, a "
        "different statute, or a different factual scenario is NOT evidence for "
        "the question just because it shares a keyword or subject area. If none "
        "of the retrieved sources actually address the specific question, say the "
        "context is insufficient rather than stretching a loosely related source "
        "into an answer.")
    user_content = f"Context:\n{context_text}\n\nQuestion: {query}"
    return system, user_content


def call_gemini(system, user_content):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=user_content,
        config=genai.types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1000,
        ),
    )

    return response.text


def ask(query, embed_model, milvus_client, bm25, bm25_chunks):
    vec_ranked = vector_search(embed_model, milvus_client, query)
    bm25_ranked = bm25_search(bm25, bm25_chunks, query)
    merged_chunks = reciprocal_rank_fusion([vec_ranked, bm25_ranked])

    system, user_content = build_prompt(query, merged_chunks)
    answer_text = call_gemini(system, user_content)

    # Map [n] labels the model actually used back to real citation strings
    # Lenient on purpose: matches [2] and also malformed cases like [2, Para 18]
    # that a model occasionally produces despite the prompt instruction above.
    used_labels = sorted(set(int(n) for n in re.findall(r"\[(\d+)", answer_text)))
    sources = []
    for label in used_labels:
        if 1 <= label <= len(merged_chunks):
            sources.append({"label": label, "citation": merged_chunks[label - 1]["citation"]})

    return {"answer": answer_text, "sources": sources}


def main():
    embed_model, milvus_client, bm25, bm25_chunks = load_resources()

    test_query = "Can promoters of a corporate debtor file application for insolvency resolution process against it? What are the recent judgments on it ?"
    result = ask(test_query, embed_model, milvus_client, bm25, bm25_chunks)

    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  [{s['label']}] {s['citation']}")


if __name__ == "__main__":
    main()
