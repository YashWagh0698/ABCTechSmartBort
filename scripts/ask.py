"""Step 12 helper: ask the WHOLE pipeline a question and watch every stage.

    python scripts/ask.py "How many paid leaves do I get?"
    python scripts/ask.py "Does ABC Tech offer dental insurance?" --show-prompt
    python scripts/ask.py "..." --threshold 0.30        try another relevance threshold for this question
    python scripts/ask.py                                interactive: type questions, empty line to stop

Stages:  [1] retrieve  ->  [2] relevance check  ->  [3] LLM (skipped when evidence is too weak)  ->  [4] answer validation
The answer you see is exactly what the chat API will return in Step 13.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                           # noqa: E402
from backend.rag.embeddings import EmbeddingError                           # noqa: E402
from backend.rag.llm import LLMConfigError                                  # noqa: E402
from backend.rag.prompt_builder import PromptBuildError                     # noqa: E402
from backend.rag.retriever import RetrievalError                            # noqa: E402
from backend.rag.vector_store import VectorStoreError                       # noqa: E402
from backend.services import Services                                       # noqa: E402

STATUS_TEXT = {
    "answered": "ANSWERED  (the model answered and the answer passed validation)",
    "refused": "REFUSED   (the model itself said it could not find the answer)",
    "no_evidence": "NO EVIDENCE  (no passage was similar enough; the LLM was NOT called)",
    "rejected": "REJECTED (the model answered, but the validator would not show it)",
    "unavailable": "UNAVAILABLE  (the LLM failed; a safe message is shown)",
}


def show(service, question: str, show_prompt: bool) -> None:
    result = service.ask(question)
    line = "=" * 90
    print(f"\n{line}\nQUESTION: {result.question}\n{line}")
    print(f"[1] RETRIEVAL   {len(result.retrieved)} passages")
    for rank, hit in enumerate(result.retrieved, 1):
        keep = "" if result.decision is None else ("  <- used" if hit in result.decision.passages else "")
        print(f"      {rank}. {hit.similarity:+.3f}  {hit.heading}{keep}")
    print(f"[2] RELEVANCE   threshold {result.decision.threshold:.2f}: {result.decision.reason}")
    if result.llm_called:
        print(f"[3] LLM         called ({len(result.sources) or len(result.prompt.sources)} passages in the prompt)"
              + (f", FAILED: {result.llm_error}" if result.llm_error else ""))
    else:
        print("[3] LLM         NOT called")
    if result.verdict:
        extra = f"  [{', '.join(result.verdict.ungrounded_numbers)}]" if result.verdict.ungrounded_numbers else ""
        print(f"[4] VALIDATION  {result.verdict.status}: {result.verdict.reason}{extra}")
    if show_prompt and result.prompt:
        print(f"\n----- SYSTEM -----\n{result.prompt.system}\n----- USER -----\n{result.prompt.user}")
    print(f"\n{STATUS_TEXT[result.status]}\n\nANSWER:\n{result.answer}")
    if result.sources:
        print("\nSOURCES:")
        for s in result.sources:
            print(f"  [{s.label}] {s.heading}  ({s.source}, version {s.document_version})")
    print(f"\n(total {result.seconds:.2f} s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ask the whole RAG pipeline a question")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--show-prompt", action="store_true", help="also print the exact prompt sent to the LLM")
    ap.add_argument("--threshold", type=float, help="relevance threshold for this run (default: RELEVANCE_THRESHOLD)")
    ap.add_argument("--top-k", type=int, help="passages to retrieve (default: TOP_K)")
    ap.add_argument("--provider", help="EMBEDDING provider: sentence-transformers or hashing")
    ap.add_argument("--model", help="embedding model name")
    ap.add_argument("--llm-provider", help="LLM provider (default: from .env)")
    ap.add_argument("--llm-model", help="LLM model (default: from .env)")
    ap.add_argument("--llm-base-url", help="LLM base URL (default: from .env)")
    args = ap.parse_args()

    for name, value in (("EMBEDDING_PROVIDER", args.provider), ("EMBEDDING_MODEL", args.model),
                        ("LLM_PROVIDER", args.llm_provider), ("LLM_MODEL", args.llm_model),
                        ("LLM_BASE_URL", args.llm_base_url)):
        if value:
            setattr(Config, name, value.strip().lower() if name.endswith("PROVIDER") else value.strip())

    try:
        service = Services().rag_service
        service.threshold, service.top_k = args.threshold, args.top_k
        if args.question:
            show(service, args.question, args.show_prompt)
            return
        print("Type a question and press Enter. Press Enter on an empty line to stop.")
        while True:
            question = input("\nquestion> ").strip()
            if not question:
                break
            try:
                show(service, question, args.show_prompt)
            except ValueError as exc:
                print(f"  Cannot answer: {exc}")
    except LLMConfigError as exc:
        raise SystemExit(f"LLM is not configured: {exc}\nRun: python scripts/check_llm.py")
    except (RetrievalError, EmbeddingError, VectorStoreError, PromptBuildError) as exc:
        raise SystemExit(f"ERROR: {exc}")
    except ValueError as exc:
        raise SystemExit(f"Cannot answer: {exc}")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
