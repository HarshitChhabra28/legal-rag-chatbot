import json
import re
import pandas as pd

HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text):
    if not isinstance(text, str):
        return ""
    return HTML_TAG_RE.sub(" ", text).strip()


def clean_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def split_judges(judge_str):
    if not isinstance(judge_str, str):
        return []
    # Judge names are '#'-separated, e.g. 'ASHOK BHUSHAN#BARUN MITRA'
    return [j.strip() for j in judge_str.split("#") if j.strip()]


def load_caselaw(path):
    df = pd.read_excel(path)
    records = []
    for _, row in df.iterrows():
        head_note = clean_ws(strip_html(row["HeadNote"]))
        judges = split_judges(row.get("Judge"))
        metadata = {
            "versus": row.get("Versus"),
            "citation": row.get("Citation"),
            "case_no": row.get("CaseNo"),
            "judges": judges,
            "court_name": row.get("CourtName"),
            "category": row.get("Category"),
            "subject": row.get("Subject"),
            "date_of_judgement": str(row.get("Date_of_Judgement")),
            "file_name": row.get("FileName"),
        }
        records.append({
            "id": f"case_{row['id']}",
            "doc_type": "case",
            "title": row.get("Versus"),
            "main_text": head_note,
            "date": str(row.get("Date_of_Judgement")),
            "metadata": metadata,
        })
    return records


def load_articles(path):
    df = pd.read_excel(path)
    records = []
    for _, row in df.iterrows():
        author = row.get("Author")
        author = None if author == "Not Found" else author
        title = row.get("Title") or ""
        summary = clean_ws(strip_html(str(row.get("Summary", ""))))
        # Prepend the title to the indexed text. ~12% of articles in this
        # dataset have Summary = "Not Found" (never scraped) or a summary
        # that doesn't match the title at all - without the title in the
        # searchable text, these are effectively unretrievable by content.
        if not summary or summary.strip().lower() == "not found":
            main_text = title
        else:
            main_text = f"{title}\n\n{summary}"
        metadata = {
            "author": author,
            "category": row.get("Category"),
            "subject": row.get("Subject"),
            "topics": row.get("Topics"),
            "issue_year": row.get("IssueYear"),
            "file_name": row.get("FileName"),
        }
        records.append({
            "id": f"article_{row['ID']}",
            "doc_type": "article",
            "title": title,
            "main_text": main_text,
            "date": str(row.get("IssueYear")),
            "metadata": metadata,
        })
    return records


def load_notifications(path):
    df = pd.read_excel(path)
    records = []
    for _, row in df.iterrows():
        metadata = {
            "notification_no": row.get("NotificationNo"),
            "notification_date": row.get("NotificationDate"),
            "type": row.get("Type"),
            "category": row.get("Category"),
            "subject": row.get("Subject"),
            "file_name": row.get("FileName"),
        }
        records.append({
            "id": f"notification_{row['Id']}",
            "doc_type": "notification",
            "title": row.get("Title"),
            # notifications have no separate body text field beyond the title
            "main_text": clean_ws(strip_html(str(row.get("Title", "")))),
            "date": str(row.get("NotificationDate")),
            "metadata": metadata,
        })
    return records


def load_queries(path):
    df = pd.read_excel(path)
    records = []
    for _, row in df.iterrows():
        metadata = {
            "subject": row.get("Subject"),
            "topics": row.get("Topics"),
            "issue_year": row.get("IssueYear"),
            "file_name": row.get("FileName"),
        }
        # Title = the question, Summary = the answer -> concatenate for retrieval,
        # but keep them separate in metadata for eval use later.
        main_text = f"Q: {row.get('Title')}\nA: {clean_ws(strip_html(str(row.get('Summary', ''))))}"
        metadata["question"] = row.get("Title")
        metadata["answer"] = clean_ws(strip_html(str(row.get("Summary", ""))))
        records.append({
            "id": f"query_{row['ID']}",
            "doc_type": "query",
            "title": row.get("Title"),
            "main_text": main_text,
            "date": str(row.get("IssueYear")),
            "metadata": metadata,
        })
    return records


def main():
    all_records = []
    all_records += load_caselaw("data/raw/caselaw.cleaned.xlsx")
    all_records += load_articles("data/raw/columns=Articles_cleaned.xlsx")
    all_records += load_notifications("data/raw/columns=Notification_cleaned.xlsx")
    all_records += load_queries("data/raw/querycleaned.xlsx")

    with open("unified_records.jsonl", "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # Quick sanity report
    from collections import Counter
    counts = Counter(r["doc_type"] for r in all_records)
    print(f"Total records: {len(all_records)}")
    for dtype, c in counts.items():
        print(f"  {dtype}: {c}")


if __name__ == "__main__":
    main()
