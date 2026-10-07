"""Step 10 helper: check that your LLM provider works. YOU run this with your real key.

    python scripts/check_llm.py                          use the provider from .env
    python scripts/check_llm.py --provider fake          offline stand-in (no key, no internet; NOT a real answer)
    python scripts/check_llm.py --verbose                also show the provider's own error message
    python scripts/check_llm.py --prompt "Say hello"     send your own tiny prompt

It sends ONE tiny, harmless request (a few tokens, a fraction of a cent) and prints what came back.
Nothing from your document is sent. Your API key is never printed.
"""
import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                    # noqa: E402
from backend.rag.llm import (LLMAuthError, LLMConfigError, LLMError, LLMRateLimitError,   # noqa: E402
                             LLMRequestError, LLMResponseError, LLMTimeoutError,
                             LLMUnavailableError, get_llm_service)

HINTS = {
    LLMConfigError: "Open your .env file and fix the setting named in the message above.",
    LLMAuthError: "The key was rejected. Check LLM_API_KEY: no spaces, no quotes, the right provider, and that it is active.",
    LLMRequestError: "The provider refused the request. Most often the model name in LLM_MODEL is wrong or you cannot use it. "
                     "Run again with --verbose to see the provider's own message.",
    LLMRateLimitError: "Too many requests or your quota is used up. Wait a minute, or check your plan and billing.",
    LLMTimeoutError: "No answer in time. Try again, or raise LLM_TIMEOUT_SECONDS in .env.",
    LLMUnavailableError: "Could not reach the provider. Check your internet connection, firewall or proxy "
                         "(for Ollama: is it running?).",
    LLMResponseError: "The provider answered but with no usable text. For models that think before answering, "
                      "raise LLM_MAX_OUTPUT_TOKENS in .env. Run with --verbose for details.",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the LLM provider with one tiny request")
    ap.add_argument("--provider", help="fake, anthropic, openai, openai-compatible or ollama (default: from .env)")
    ap.add_argument("--model", help="model name (default: from .env)")
    ap.add_argument("--base-url", help="only for gateways or local servers (default: from .env)")
    ap.add_argument("--prompt", default="Reply with the single word OK.")
    ap.add_argument("--verbose", action="store_true", help="show the provider's own error message")
    args = ap.parse_args()

    if args.provider:
        Config.LLM_PROVIDER = args.provider.strip().lower()
    if args.model:
        Config.LLM_MODEL = args.model.strip()
    if args.base_url:
        Config.LLM_BASE_URL = args.base_url.strip()

    line = "=" * 78
    print(line)
    print("LLM CHECK")
    print(line)
    try:
        service = get_llm_service()
        print(f"  provider : {service.provider_name}")
        print(f"  model    : {service.model_name}")
        base = getattr(service, "base_url", None)
        if base:
            parts = urlsplit(base)
            print(f"  endpoint : {parts.scheme}://{parts.netloc}")
        print(f"  key set  : {'yes' if Config.LLM_API_KEY else 'no'}   (the key itself is never shown)")
        print(f"  timeout  : {Config.LLM_TIMEOUT_SECONDS:g} s, up to {Config.LLM_MAX_RETRIES} retries on temporary errors")
        if service.provider_name == "fake":
            print("  NOTE: 'fake' returns canned text. It proves the wiring, NOT a real model.")
        print("\n  sending one tiny request ...")
        reply = service.generate("You are a connectivity test. Keep the answer very short.", args.prompt)
    except LLMError as exc:
        print(f"\n  FAILED ({type(exc).__name__}): {exc}")
        if args.verbose and exc.detail:
            print(f"  provider says: {exc.detail}")
        print(f"  What to do: {HINTS.get(type(exc), 'Run again with --verbose.')}")
        print(line)
        sys.exit(1)

    shown = reply.text if len(reply.text) <= 200 else reply.text[:200] + " ..."
    print(f"\n  answer        : {shown}")
    print(f"  time          : {reply.seconds:.2f} s   (attempts: {reply.attempts})")
    print(f"  tokens        : in {reply.input_tokens}, out {reply.output_tokens}")
    print(f"  finish reason : {reply.finish_reason}" + ("   (CUT OFF: raise LLM_MAX_OUTPUT_TOKENS)" if reply.truncated else ""))
    print("\n  RESULT: the LLM provider works.")
    print(line)


if __name__ == "__main__":
    main()
