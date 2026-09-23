"""
Step 3: Citation formatting.

Turns a chunk's metadata into the citation string a human would expect
to see under "Sources:". This runs at RETRIEVAL time in the final app -
never ask the LLM to reproduce a citation itself.
"""

import json


def format_citation(chunk):
    dtype = chunk["doc_type"]
    md = chunk["metadata"]

    if dtype == "case":
        # md['citation'] already ends with the court in parens, e.g.
        # '[2026] 192 CLA 271 (NCLAT)' - so we only add the date and bench.
        judges = ", ".join(md.get("judges", [])) or "Judge not recorded"
        return (
            f"{chunk['title']} — {md.get('citation', 'Citation not recorded')}, "
            f"{md.get('date_of_judgement', 'date unknown')} "
            f"[Bench: {judges}]"
        )

    if dtype == "article":
        author = md.get("author") or "Author not listed"
        return f"\"{chunk['title']}\" by {author} ({md.get('issue_year', 'year unknown')})"

    if dtype == "notification":
        return (
            f"{chunk['title']} — Notification No. {md.get('notification_no', 'N/A')}, "
            f"dated {md.get('notification_date', 'date unknown')}"
        )

    if dtype == "query":
        return f"FAQ: \"{chunk['title']}\""

    return chunk["title"]


def main():
    with open("chunks.jsonl", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    missing_fields = []
    samples = {}

    for c in chunks:
        citation = format_citation(c)
        c["citation"] = citation
        if "None" in citation or "not recorded" in citation or "unknown" in citation:
            missing_fields.append((c["chunk_id"], citation))
        if c["doc_type"] not in samples:
            samples[c["doc_type"]] = citation

    print("Sample citation per doc_type:")
    for dtype, sample in samples.items():
        print(f"  [{dtype}] {sample}")

    print(f"\nChunks with a missing/suspect metadata field: {len(missing_fields)} / {len(chunks)}")
    for chunk_id, citation in missing_fields[:5]:
        print(f"  {chunk_id}: {citation}")

    with open("chunks_with_citations.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()