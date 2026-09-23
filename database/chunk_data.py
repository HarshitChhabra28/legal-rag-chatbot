import json
import re

TARGET_WORDS = 400
OVERLAP_SENTENCES = 1  # small overlap so a holding split across chunks isn't lost

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentence_chunks(text, target_words=TARGET_WORDS, overlap=OVERLAP_SENTENCES):
    sentences = SENTENCE_SPLIT_RE.split(text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > target_words and current:
            chunks.append(" ".join(current))
            # start next chunk with overlap from the tail of this one
            current = current[-overlap:] if overlap else []
            current_len = sum(len(s.split()) for s in current)
        current.append(sent)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_record(rec):
    chunks = []
    if rec["doc_type"] == "case":
        pieces = split_into_sentence_chunks(rec["main_text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{rec['id']}_{i}",
                "doc_id": rec["id"],
                "doc_type": rec["doc_type"],
                "title": rec["title"],
                "text": piece,
                "metadata": rec["metadata"],
            })
    else:
        # short doc types: single chunk
        chunks.append({
            "chunk_id": f"{rec['id']}_0",
            "doc_id": rec["id"],
            "doc_type": rec["doc_type"],
            "title": rec["title"],
            "text": rec["main_text"],
            "metadata": rec["metadata"],
        })
    return chunks


def main():
    all_chunks = []
    with open("unified_records.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            all_chunks.extend(chunk_record(rec))

    with open("chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    from collections import Counter
    counts = Counter(c["doc_type"] for c in all_chunks)
    print(f"Total chunks: {len(all_chunks)}")
    for dtype, c in counts.items():
        print(f"  {dtype}: {c}")

    case_chunk_counts = Counter()
    for c in all_chunks:
        if c["doc_type"] == "case":
            case_chunk_counts[c["doc_id"]] += 1
    multi = sum(1 for v in case_chunk_counts.values() if v > 1)
    print(f"Cases split into >1 chunk: {multi} / {len(case_chunk_counts)}")


if __name__ == "__main__":
    main()
