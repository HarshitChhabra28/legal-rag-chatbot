"""
Step 4: Embed chunks and store in Milvus (Milvus Lite - embedded, file-based,
no separate server needed for a corpus this size).

Model: BAAI/bge-base-en-v1.5 (local, free, no API key)

IMPORTANT: bge models expect a query-side instruction prefix at search time
(NOT added to documents here). That happens in Step 6 when we build search.

Output: law_project.db (Milvus Lite database file, created in this folder)
"""

import json
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DB_PATH = "law_project.db"
COLLECTION_NAME = "legal_chunks"


def load_chunks(path="chunks_with_citations.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_collection(client, dim):
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="doc_type", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1000)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8000)
    schema.add_field(field_name="citation", datatype=DataType.VARCHAR, max_length=2000)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )

    client = MilvusClient(DB_PATH)
    build_collection(client, dim)

    rows = []
    for c, vec in zip(chunks, embeddings):
        rows.append({
            "chunk_id": c["chunk_id"],
            "vector": vec.tolist(),
            "doc_type": c["doc_type"],
            "title": (c["title"] or "")[:990],
            "text": c["text"][:7990],
            "citation": c["citation"][:1990],
        })

    result = client.insert(collection_name=COLLECTION_NAME, data=rows)
    print(f"Inserted {result['insert_count']} vectors into '{COLLECTION_NAME}' at {DB_PATH}")

    # Sanity check: run one search to confirm the index actually works
    test_query = "Represent this sentence for searching relevant passages: compounding an offence under the Companies Act"
    query_vec = model.encode([test_query], normalize_embeddings=True)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=query_vec.tolist(),
        limit=3,
        output_fields=["title", "doc_type", "citation"],
    )
    print("\nSanity-check search ('compounding an offence under the Companies Act'):")
    for hit in results[0]:
        print(f"  score={hit['distance']:.3f} [{hit['entity']['doc_type']}] {hit['entity']['title']}")


if __name__ == "__main__":
    main()