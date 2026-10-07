"""Step 11 helper: see the EXACT prompt the LLM would receive, and optionally ask your real model.

    python scripts/show_prompt.py "How many paid leaves do I get?"           print the prompt (no LLM call, no key needed)
    python scripts/show_prompt.py "How many paid leaves do I get?" --ask     also send it to YOUR LLM and print the answer
    python scripts/show_prompt.py "..." --top-k 3 --max-context-chars 2000   see how the budget drops passages

Flow: question -> retriever (real index) -> prompt builder -> [optional] your LLM.
Nothing here decides whether to answer (that is Step 12): you see what the model would be given.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                          # noqa: E402
from backend.rag.embeddings import EmbeddingError, get_embedding_service   # noqa: E402
from backend.rag.llm import LLMError, LLMResponseError, get_llm_service    # noqa: E402
from backend.rag.prompt_builder import PromptBuildError, build_prompt     # noqa: E402
from backend.rag.retriever import RetrievalError, Retriever                # noqa: E402
from backend.rag.vector_store import VectorStoreError, get_vector_store   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Show the grounded prompt for a question")
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=Config.TOP_K, help="how many passages to retrieve")
    ap.add_argument("--max-context-chars", type=int, default=Config.CONTEXT_MAX_CHARS)
    ap.add_argument("--ask", action="store_true", help="send the prompt to your LLM and print its answer")
    ap.add_argument("--provider", help="EMBEDDING provider: sentence-transformers or hashing (default: from .env)")
    ap.add_argument("--model", help="embedding model name (default: from .env)")
    ap.add_argument("--llm-provider", help="LLM provider for --ask (default: from .env)")
    ap.add_argument("--llm-model", help="LLM model for --ask (default: from .env)")
    ap.add_argument("--llm-base-url", help="LLM base URL for --ask (default: from .env)")
    args = ap.parse_args()

    if args.provider:
        Config.EMBEDDING_PROVIDER = args.provider.strip().lower()
    if args.model:
        Config.EMBEDDING_MODEL = args.model.strip()
    if args.llm_provider:
        Config.LLM_PROVIDER = args.llm_provider.strip().lower()
    if args.llm_model:
        Config.LLM_MODEL = args.llm_model.strip()
    if args.llm_base_url:
        Config.LLM_BASE_URL = args.llm_base_url.strip()

    line = "=" * 90
    try:
        response = Retriever(get_embedding_service(), get_vector_store()).retrieve(args.question, args.top_k)
        prompt = build_prompt(response.question, response.results, args.max_context_chars)
    except (RetrievalError, EmbeddingError, VectorStoreError, PromptBuildError) as exc:
        raise SystemExit(f"ERROR: {exc}")
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"Cannot build a prompt: {exc}")

    print(line)
    print(f"PROMPT FOR: {prompt.question}")
    print(line)
    print(f"  retrieved {len(response.results)} passages, {len(prompt.sources)} placed in the prompt, "
          f"{len(prompt.dropped_chunk_ids)} dropped by the budget; {prompt.context_chars} context characters")
    for source in prompt.sources:
        hit = next(r for r in response.results if r.chunk_id == source.chunk_id)
        print(f"    [{source.label}] similarity {hit.similarity:+.3f}  {source.heading}   ({source.chunk_id})")
    for chunk_id in prompt.dropped_chunk_ids:
        print(f"    dropped: {chunk_id}")
    print(f"\n----- SYSTEM MESSAGE (constant rules, {len(prompt.system)} characters) " + "-" * 20)
    print(prompt.system)
    print(f"\n----- USER MESSAGE ({len(prompt.user)} characters) " + "-" * 40)
    print(prompt.user)

    if args.ask:
        print("\n" + "-" * 90)
        try:
            service = get_llm_service()
            print(f"asking {service.provider_name} / {service.model_name} ...")
            reply = service.generate(prompt.system, prompt.user)
        except LLMError as exc:
            print(f"\nLLM FAILED ({type(exc).__name__}): {exc}")
            if isinstance(exc, LLMResponseError):
                print("Tip: models that think before answering can use the whole token budget. Raise LLM_MAX_OUTPUT_TOKENS in .env.")
            print("Run `python scripts/check_llm.py --verbose` to diagnose the provider.")
            sys.exit(1)
        print(f"\nANSWER:\n{reply.text}")
        print(f"\n(time {reply.seconds:.2f} s, tokens in {reply.input_tokens} / out {reply.output_tokens}, "
              f"finish: {reply.finish_reason}" + (", CUT OFF: raise LLM_MAX_OUTPUT_TOKENS" if reply.truncated else "") + ")")
    print(line)


if __name__ == "__main__":
    main()
