"""Step 10 - LLM integration.

The LLM is the "G" in RAG (Generation). Given a prompt it writes text. It is the
ONLY part of the system that talks to an outside service, so this file is built to be:

  * REPLACEABLE  - the rest of the application only ever calls
                   get_llm_service().generate(system, user). Switching from Claude
                   to OpenAI to a local model is a change in .env, nothing else.
  * SAFE         - the API key and the prompt text never appear in an error message,
                   a log line or repr(). Errors carry a SAFE message for logs and a
                   separate user-facing text (specification section 68).
  * BOUNDED      - a timeout on every call, and a small number of retries that apply
                   only to temporary problems (rate limit, timeout, server overloaded).

Classes
    LLMService (abstract)
      |- FakeLLMService          canned answers; for tests and offline work. NOT real answers.
      |- AnthropicService        Claude        (POST /v1/messages)
      |- OpenAICompatibleService OpenAI, Ollama and any server using the OpenAI chat format

HOW WE TALK TO PROVIDERS: plain HTTPS with Python's standard library (urllib). No vendor
SDK is needed, so there is nothing extra to install, and the tests can run the real
network code against a local fake server. The request formats were checked against the
providers' current documentation (see the README, Step 10).

This file does NOT build prompts (Step 11), decide whether to answer (Step 12) or
expose an endpoint (Step 13).
"""
import json
import socket
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Sequence
from urllib.parse import urlsplit

from backend.config import Config
from backend.utils.logger import get_logger

logger = get_logger(__name__)

PROVIDERS = ("fake", "anthropic", "openai", "openai-compatible", "ollama")
SAFE_USER_MESSAGE = "The answer service is temporarily unavailable."     # specification section 68
MAX_PROMPT_CHARS = 60_000          # refuse absurdly large prompts (cost and abuse guard)
MAX_RESPONSE_BYTES = 2_000_000
MAX_BACKOFF_SECONDS = 8.0
MAX_RETRY_AFTER_SECONDS = 10.0
ANTHROPIC_VERSION = "2023-06-01"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")


# =============================================================================
# Errors: every one has a SAFE message. None may contain the key or the prompt.
# =============================================================================
class LLMError(Exception):
    """Base class. str(error) is safe to log. `detail` is extra developer information
    (the provider's own message, redacted and shortened): shown only by check_llm --verbose."""
    retryable = False
    user_message = SAFE_USER_MESSAGE

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message)
        self.detail = detail


class LLMConfigError(LLMError):
    """Settings are missing or wrong (provider, model, key, URL ...). Fix .env."""


class LLMAuthError(LLMError):
    """The provider rejected the API key (HTTP 401 or 403)."""


class LLMRequestError(LLMError):
    """The provider rejected the request itself (wrong model name, unsupported parameter ...)."""


class LLMRateLimitError(LLMError):
    """Too many requests (HTTP 429). Temporary."""
    retryable = True

    def __init__(self, message: str, detail: Optional[str] = None, retry_after: Optional[float] = None):
        super().__init__(message, detail)
        self.retry_after = retry_after


class LLMTimeoutError(LLMError):
    """The provider did not answer in time. Temporary."""
    retryable = True


class LLMUnavailableError(LLMError):
    """The provider is down, overloaded or unreachable. Temporary."""
    retryable = True


class LLMResponseError(LLMError):
    """The provider answered, but not with usable text (empty, refused, malformed)."""


# =============================================================================
# Result
# =============================================================================
@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    seconds: float = 0.0
    attempts: int = 1
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    truncated: bool = False            # True when the model stopped because it hit the output limit


# =============================================================================
# The interface (with the retry logic shared by every provider)
# =============================================================================
class LLMService(ABC):
    provider_name = "unknown"
    model_name = "unknown"

    def __init__(self, max_retries: int = 0, backoff_seconds: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep

    @abstractmethod
    def _generate_once(self, system: str, user: str) -> LLMResponse:
        """One attempt. Raise an LLMError on failure."""

    def __repr__(self) -> str:                      # never includes secrets
        return f"{type(self).__name__}(provider={self.provider_name!r}, model={self.model_name!r})"

    @staticmethod
    def _validate_prompt(system: str, user: str) -> None:
        for name, value in (("system", system), ("user", user)):
            if not isinstance(value, str):
                raise TypeError(f"the {name} prompt must be text")
            if not value.strip():
                raise ValueError(f"the {name} prompt is empty")
        if len(system) + len(user) > MAX_PROMPT_CHARS:
            raise ValueError(f"the prompt is too long (maximum {MAX_PROMPT_CHARS} characters)")

    def generate(self, system: str, user: str) -> LLMResponse:
        """Send the prompt, return the answer. Retries temporary failures a few times."""
        self._validate_prompt(system, user)
        started = time.perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self._generate_once(system, user)
            except LLMError as exc:
                if not exc.retryable or attempt > self.max_retries:
                    logger.error("LLM call failed (%s, %s): %s after %d attempt(s)",
                                 self.provider_name, self.model_name, type(exc).__name__, attempt)
                    raise
                delay = min(MAX_BACKOFF_SECONDS, self.backoff_seconds * (2 ** (attempt - 1)))
                if isinstance(exc, LLMRateLimitError) and exc.retry_after:
                    delay = max(delay, min(exc.retry_after, MAX_RETRY_AFTER_SECONDS))
                logger.warning("LLM call failed (%s), retry %d of %d in %.1f s",
                               type(exc).__name__, attempt, self.max_retries, delay)
                self._sleep(delay)
                continue
            response.attempts = attempt
            response.seconds = time.perf_counter() - started
            # Log sizes and counts only. Never the prompt, the answer or the key.
            logger.info("LLM call ok: provider=%s model=%s seconds=%.2f attempts=%d tokens_in=%s tokens_out=%s "
                        "prompt_chars=%d answer_chars=%d truncated=%s",
                        response.provider, response.model, response.seconds, attempt,
                        response.input_tokens, response.output_tokens, len(system) + len(user),
                        len(response.text), response.truncated)
            return response


# =============================================================================
# Fake provider (tests and offline work)
# =============================================================================
class FakeLLMService(LLMService):
    """A scriptable stand-in. It does NOT understand anything: it returns canned text.

        FakeLLMService("Employees get 24 days.")                          one fixed answer
        FakeLLMService(answers={"notice period": "60 days"})              answer by a word in the prompt
        FakeLLMService(failures=[LLMTimeoutError("t"), None])             first call fails, second works

    Every prompt it receives is kept in `calls`, so tests can check exactly what would have been sent.
    """
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, answer: str = "This is a canned answer from the fake LLM.",
                 answers: Optional[Dict[str, str]] = None,
                 failures: Optional[Sequence[Optional[Exception]]] = None, **kwargs):
        super().__init__(**kwargs)
        self.answer = answer
        self.answers = {k.lower(): v for k, v in (answers or {}).items()}
        self._failures = list(failures or [])
        self.calls = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _generate_once(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if self._failures:
            failure = self._failures.pop(0)
            if failure is not None:
                raise failure
        text = self.answer
        for key, canned in self.answers.items():
            if key in user.lower():
                text = canned
                break
        return LLMResponse(text=text, provider=self.provider_name, model=self.model_name,
                           input_tokens=len((system + " " + user).split()), output_tokens=len(text.split()),
                           finish_reason="stop")


# =============================================================================
# Shared HTTP code
# =============================================================================
class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: urllib would forward our API key header to the new address."""
    def redirect_request(self, *args, **kwargs):
        return None


def _validate_base_url(url: str, has_key: bool) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise LLMConfigError("LLM_BASE_URL must start with http:// or https:// and contain a host name.")
    if "@" in parts.netloc:
        raise LLMConfigError("LLM_BASE_URL must not contain a user name or password.")
    if has_key and parts.scheme == "http" and parts.hostname not in LOCAL_HOSTS:
        raise LLMConfigError("Refusing to send the API key over plain http to a remote host. Use https.")
    return url.rstrip("/")


class _HTTPService(LLMService):
    def __init__(self, model: str, api_key: str, base_url: str, timeout: float,
                 temperature: Optional[float], max_output_tokens: int, **kwargs):
        super().__init__(**kwargs)
        if not model:
            raise LLMConfigError("LLM_MODEL is not set. Put the model name in your .env file.")
        if not 0 < timeout <= 300:
            raise LLMConfigError("LLM_TIMEOUT_SECONDS must be between 0 and 300.")
        if max_output_tokens < 1:
            raise LLMConfigError("LLM_MAX_OUTPUT_TOKENS must be at least 1.")
        self.model_name = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._api_key = api_key
        self.base_url = _validate_base_url(base_url, bool(api_key))

    # ---- helpers that keep the key out of everything ----------------------------------
    def _redact(self, text: str) -> str:
        text = " ".join(str(text).split())
        if self._api_key:
            text = text.replace(self._api_key, "[REDACTED]")
        return text[:300]

    def _error_detail(self, body: bytes) -> Optional[str]:
        """The provider's own error message, best effort, redacted and shortened."""
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
            error = data.get("error", data) if isinstance(data, dict) else {}
            message = error.get("message") if isinstance(error, dict) else str(error)
            return self._redact(message) if message else None
        except (ValueError, AttributeError):
            return self._redact(body.decode("utf-8", errors="replace")) if body else None

    def _error_for_status(self, status: int, body: bytes, headers) -> LLMError:
        detail = self._error_detail(body)
        if status in (401, 403):
            return LLMAuthError(f"The provider rejected the API key (HTTP {status}). Check LLM_API_KEY.", detail)
        if status == 404:
            return LLMRequestError("Model or endpoint not found (HTTP 404). Check LLM_MODEL and LLM_BASE_URL.", detail)
        if status == 429:
            retry_after = None
            try:
                retry_after = float(headers.get("Retry-After")) if headers and headers.get("Retry-After") else None
            except ValueError:
                pass
            return LLMRateLimitError("The provider is rate limiting requests (HTTP 429).", detail, retry_after)
        if status == 408 or status >= 500:
            return LLMUnavailableError(f"The provider is unavailable or overloaded (HTTP {status}).", detail)
        if 300 <= status < 400:
            return LLMRequestError(f"The provider answered with an unexpected redirect (HTTP {status}). "
                                   "Check LLM_BASE_URL.", detail)
        return LLMRequestError(f"The provider rejected the request (HTTP {status}).", detail)

    def _post_json(self, url: str, headers: Dict[str, str], payload: dict) -> dict:
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "abc-tech-smart-bot", **headers})
        opener = urllib.request.build_opener(_NoRedirects)
        try:
            with opener.open(request, timeout=self.timeout) as reply:
                raw = reply.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read(4096)
            except OSError:
                pass
            raise self._error_for_status(exc.code, body, exc.headers) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMTimeoutError(f"The provider did not answer within {self.timeout:g} seconds.") from None
            raise LLMUnavailableError("Could not reach the provider (network or server problem).",
                                      self._redact(exc.reason)) from None
        except (TimeoutError, socket.timeout):
            raise LLMTimeoutError(f"The provider did not answer within {self.timeout:g} seconds.") from None
        except OSError as exc:                                   # connection reset and similar
            raise LLMUnavailableError("The connection to the provider failed.", self._redact(exc)) from None

        if len(raw) > MAX_RESPONSE_BYTES:
            raise LLMResponseError("The provider's answer was unexpectedly large.")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise LLMResponseError("The provider's answer was not valid JSON.") from None
        if not isinstance(data, dict):
            raise LLMResponseError("The provider's answer had an unexpected shape.")
        return data

    @staticmethod
    def _tokens(usage, *names) -> Optional[int]:
        if not isinstance(usage, dict):
            return None
        for name in names:
            if isinstance(usage.get(name), int):
                return usage[name]
        return None


# =============================================================================
# Anthropic (Claude)
# =============================================================================
class AnthropicService(_HTTPService):
    """Claude via the Messages API: POST {base}/v1/messages.

    Headers x-api-key and anthropic-version; the system prompt is a TOP-LEVEL field
    (not a message); max_tokens is required; the answer is a list of content blocks."""
    provider_name = "anthropic"
    DEFAULT_BASE_URL = "https://api.anthropic.com"

    def __init__(self, model: str, api_key: str, base_url: str = "", timeout: float = 30.0,
                 temperature: Optional[float] = 0.0, max_output_tokens: int = 600, **kwargs):
        if not api_key:
            raise LLMConfigError("LLM_API_KEY is not set. Put your Anthropic API key in your .env file.")
        if temperature is not None and not 0 <= temperature <= 1:
            raise LLMConfigError("For Anthropic, LLM_TEMPERATURE must be between 0 and 1 (or empty).")
        super().__init__(model, api_key, base_url or self.DEFAULT_BASE_URL, timeout, temperature,
                         max_output_tokens, **kwargs)

    def _generate_once(self, system: str, user: str) -> LLMResponse:
        payload = {"model": self.model_name, "max_tokens": self.max_output_tokens, "system": system,
                   "messages": [{"role": "user", "content": user}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = self._post_json(f"{self.base_url}/v1/messages",
                               {"x-api-key": self._api_key, "anthropic-version": ANTHROPIC_VERSION}, payload)

        blocks = data.get("content")
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text") if isinstance(blocks, list) else ""
        stop_reason = data.get("stop_reason")
        if not text.strip():
            raise LLMResponseError(f"The model returned no text (stop reason: {stop_reason}).")
        usage = data.get("usage")
        return LLMResponse(text=text.strip(), provider=self.provider_name, model=self.model_name,
                           input_tokens=self._tokens(usage, "input_tokens"),
                           output_tokens=self._tokens(usage, "output_tokens"),
                           finish_reason=stop_reason, truncated=stop_reason == "max_tokens")


# =============================================================================
# OpenAI and everything that speaks its chat format (Ollama, LM Studio, gateways ...)
# =============================================================================
class OpenAICompatibleService(_HTTPService):
    """POST {base}/chat/completions with a system and a user message.

    Differences handled here (verified against current documentation):
      * Official OpenAI uses `max_completion_tokens`; most other servers still use `max_tokens`.
      * OpenAI reasoning models (o-series, gpt-5 and newer) REJECT `temperature`, so it is left out for them.
        Their token budget also includes hidden reasoning, so a small LLM_MAX_OUTPUT_TOKENS can give an empty answer.
      * Local servers such as Ollama ignore the API key, so it is optional there."""
    OPENAI_URL = "https://api.openai.com/v1"
    OLLAMA_URL = "http://localhost:11434/v1"

    def __init__(self, provider: str, model: str, api_key: str = "", base_url: str = "", timeout: float = 30.0,
                 temperature: Optional[float] = 0.0, max_output_tokens: int = 600, **kwargs):
        self.provider_name = provider
        if provider == "openai":
            if not api_key:
                raise LLMConfigError("LLM_API_KEY is not set. Put your OpenAI API key in your .env file.")
            base_url = base_url or self.OPENAI_URL
        elif provider == "ollama":
            base_url = base_url or self.OLLAMA_URL
        elif not base_url:
            raise LLMConfigError("LLM_BASE_URL is required for the 'openai-compatible' provider.")
        super().__init__(model, api_key, base_url, timeout, temperature, max_output_tokens, **kwargs)

    def _reasoning_model(self) -> bool:
        name = self.model_name.lower().split("/")[-1]
        return self.provider_name == "openai" and (name.startswith("gpt-5") or name[:1] == "o" and name[1:2].isdigit())

    def _generate_once(self, system: str, user: str) -> LLMResponse:
        token_field = "max_completion_tokens" if self.provider_name == "openai" else "max_tokens"
        payload = {"model": self.model_name,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   token_field: self.max_output_tokens}
        if self.temperature is not None and not self._reasoning_model():
            payload["temperature"] = self.temperature
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        data = self._post_json(f"{self.base_url}/chat/completions", headers, payload)

        choices = data.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, list):                              # some servers return a list of parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        finish = choice.get("finish_reason")
        if not isinstance(content, str) or not content.strip():
            why = "the model refused" if message.get("refusal") else f"finish reason: {finish}"
            raise LLMResponseError(f"The model returned no text ({why}).")
        usage = data.get("usage")
        return LLMResponse(text=content.strip(), provider=self.provider_name, model=self.model_name,
                           input_tokens=self._tokens(usage, "prompt_tokens", "input_tokens"),
                           output_tokens=self._tokens(usage, "completion_tokens", "output_tokens"),
                           finish_reason=finish, truncated=finish == "length")


# =============================================================================
# Factory
# =============================================================================
@lru_cache(maxsize=None)
def _build(provider: str, model: str, api_key: str, base_url: str, timeout: float,
           temperature: Optional[float], max_output_tokens: int, max_retries: int,
           backoff_seconds: float) -> LLMService:
    if not provider:
        raise LLMConfigError("LLM_PROVIDER is not set. Choose one in .env: " + ", ".join(PROVIDERS) + ".")
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Unknown LLM_PROVIDER '{provider}'. Valid values: {', '.join(PROVIDERS)}.")
    if not 0 <= max_retries <= 5:
        raise LLMConfigError("LLM_MAX_RETRIES must be between 0 and 5.")
    if backoff_seconds < 0:
        raise LLMConfigError("LLM_RETRY_BACKOFF_SECONDS cannot be negative.")
    common = {"max_retries": max_retries, "backoff_seconds": backoff_seconds}
    if provider == "fake":
        return FakeLLMService(**common)
    http = {"timeout": timeout, "temperature": temperature, "max_output_tokens": max_output_tokens, **common}
    if provider == "anthropic":
        return AnthropicService(model, api_key, base_url, **http)
    return OpenAICompatibleService(provider, model, api_key, base_url, **http)


def get_llm_service() -> LLMService:
    """The ONE place the rest of the app gets its LLM from (built once, reused)."""
    return _build(Config.LLM_PROVIDER, Config.LLM_MODEL, Config.LLM_API_KEY, Config.LLM_BASE_URL,
                  Config.LLM_TIMEOUT_SECONDS, Config.LLM_TEMPERATURE, Config.LLM_MAX_OUTPUT_TOKENS,
                  Config.LLM_MAX_RETRIES, Config.LLM_RETRY_BACKOFF_SECONDS)
