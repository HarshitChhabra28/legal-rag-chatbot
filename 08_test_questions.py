import importlib.util
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
_step6_path = BASE_DIR / "database" / "06_hybrid_retrieve_and_generate.py"
_spec = importlib.util.spec_from_file_location("step6", _step6_path)
step6 = importlib.util.module_from_spec(_spec)
sys.modules["step6"] = step6
_spec.loader.exec_module(step6)


TEST_QUESTIONS = [
   	"Which companies are required to file form DPT – 3 and what is the due date of filing it for F.Y. 2025-26 ?",
	"A foreign company wants to open it liaison office in India ? How cant it do so ?",
	"Who are person acting in concert ? What are landmark case on it ?",
	"Can promoters of a corporate debtor file application for insolvency resolution process against it? What are the recent judgments on it ?",
	"What are the liabilities of directors of a company in case of dishonour of cheque issued during moratorium ?",
]


def main():
    embed_model, milvus_client, bm25, bm25_chunks = step6.load_resources()

    for i, q in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n{'='*80}\nQ{i}: {q}\n{'='*80}")
        result = step6.ask(q, embed_model, milvus_client, bm25, bm25_chunks)
        print("ANSWER:", result["answer"])
        print("SOURCES:")
        for s in result["sources"]:
            print(f"  [{s['label']}] {s['citation']}")


if __name__ == "__main__":
    main()
