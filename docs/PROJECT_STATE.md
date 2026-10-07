# ABC Tech Smart Bot - Project State (living document)

**Read this file first.** Every chat that works on this project must update it at the end of its step
(status table, section 10 change log, sections 4 and 11). If this file and the code disagree, trust the code and say so.

Last updated: after **Step 12** (refusal logic + answer validator). Tests: **602 passing** (Step 12 has NO dedicated pytest tests yet: see section 11).

---

## 1. What this project is

A **closed-domain RAG chatbot** for one company document (`data/raw/company.html`, ABC Tech Solutions).
Principle from the specification (`ProjectDocumentation.docx`): **NO EVIDENCE -> NO ANSWER.** The LLM only phrases answers;
the company document is the source of truth.

Pipeline: `company.html -> parse -> chunk -> embed -> ChromaDB` (indexing) and
`question -> embed -> top-K chunks -> relevance check -> grounded prompt -> LLM -> answer validation -> answer + sources` (asking).
Indexing and retrieval are DONE. Everything from the LLM onward is NOT done yet.

## 2. The person and their environment

- A **learner**. They want to understand every component, so every step is explained in plain language.
- Windows 11, `cmd.exe`, a Python virtual environment (`.venv`) inside the project folder. They run commands from the project root.
- The real embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is installed and works on their machine.
- They apply **changes-only zips** to their existing project (they do not want to recreate the venv), so every step must ship the
  exact list of new and modified files. Only mention `pip install` when `requirements.txt` changed.
- They want **API run instructions** (curl in cmd, PowerShell, Postman, browser) whenever an API endpoint is created.
- No LLM provider has been chosen yet (see section 11).

## 3. Status

| Step | What | Status | Key facts |
|---|---|---|---|
| 1 | Project setup, Flask app, `/api/health` | Done | App factory, config, logger |
| 2 | Company HTML + inspection | Done | 14 sections, 46 subsections, 11 tables; deliberate noise and a hidden "sports club" note |
| 3 | HTML parser | Done | 52 sections; table rows become labelled sentences; FAQ becomes Q/A pairs |
| 4 | Chunking | Done | 56 chunks at 150 words / 20 overlap; heading kept inside every chunk |
| 5 | Embeddings | Done | `EmbeddingService`; real model + offline `hashing` stand-in |
| 6 | ChromaDB vector store | Done | Cosine space; only file that imports chromadb |
| 7 | Indexing + first APIs | Done | `/api/ingest`, `/api/reindex`, `/api/sources`; fingerprint-based index validity |
| 8 | Retriever | Done | Refuses empty index and embedding-model mismatch |
| 9 | Retrieval evaluation | Done | 49 questions, Recall@K / Precision@K / MRR, threshold sweep |
| 10 | LLM integration (`llm.py`) | Done | `LLMService.generate(system, user)`; fake / anthropic / openai / ollama / openai-compatible over plain HTTPS (urllib); safe errors; bounded retries; `scripts/check_llm.py` |
| 11 | Grounded prompt (`prompt_builder.py`) | Done | Pure `build_prompt(question, chunks)`; constant system rules; fenced `<context>`/`<question>`; `<` neutralised; lowest-ranked dropped first; `scripts/show_prompt.py` |
| 12 | Refusal / no-evidence logic (`relevance.py`, `answer_validator.py`) | Done | Gate before the LLM (threshold 0.20, provisional); validator after it (numbers per best-matching passage, rules leak, refusals); `RAGService`; `scripts/ask.py` |
| 13 | Flask chat API (`POST /api/chat`) | **NEXT** | |
| 14 | Frontend chat UI | Planned | |
| 15 | Source citations | Planned | |
| 16 | End-to-end testing | Planned | |
| 17 | RAG evaluation | Planned | |
| 18 | Optimization | Planned | |
| 19 | Documentation | Planned | |
| 20 | Interview preparation | Planned | |

Tests per file: api 2, company_document 27, html_parser 62, chunker 45, embeddings 42, vector_store 39, ingestion 34,
ingestion_api 19, retrieval 39, evaluation 67, project_state 29, llm 121, prompt_builder 76 = **602**
(573 for the pipeline, the LLM and the prompt + 29 that guard the hand-over files; `pytest` and `python -m pytest` must both show 602 at the start of the next step).

## 4. Verified vs NOT verified (keep this honest)

**Verified by the user on their own machine:**
- `scripts/check_embeddings.py` with the real model: the right chunk was in the top 3 for 10 of 10 preview questions.
- The index was built with the real model: 56 chunks, `embedding_model = sentence-transformers/all-MiniLM-L6-v2`.
- Observed best-similarity scores with the real model: correct answers about 0.27-0.51 (laptop 0.273, time off 0.473, travel expenses 0.506); "no answer" questions:
  revenue 0.298, stock price 0.222, **dental insurance 0.434** (it is close to the real Benefits chunk). So a similarity
  threshold alone cannot separate answerable from unanswerable questions.

**NOT yet verified (do not claim otherwise):**
- The real-model **evaluation report** (`python scripts/evaluate_retrieval.py`): never seen. Needed before choosing a threshold in Step 12.
- The token-limit check (`check_embeddings.py` section [5]): output never seen.
- Plain `pytest` on the user's machine after `pytest.ini` was added (worked on Linux).
- curl / PowerShell / Postman commands on Windows (only curl on Linux was run against a live server).
- **Any real LLM provider call.** The Step 10 providers were only exercised against a LOCAL FAKE SERVER. The Anthropic and OpenAI request formats were checked
  against current documentation but never sent to the real services. The user must run `python scripts/check_llm.py` with their own key and report the result.
- **Step 12 behaviour with the user's real model.** The relevance gate, validator and `RAGService` were verified ONLY with the offline embedder and `FakeLLMService`, via `scripts/verify_step12.py` (41 checks). No result of `scripts/ask.py` against Groq has been reported yet.
- **Step 12 has no pytest tests** (built without running pytest at the user's request).
- **How a real model behaves with the Step 11 prompt** (obeys the refusal rule? resists injection? answers the near-miss question correctly?). The user runs `scripts/show_prompt.py --ask`; no result has been reported yet.
  (The user HAS confirmed `check_llm.py` works with Groq: provider `openai-compatible`, model `openai/gpt-oss-20b`, base URL `https://api.groq.com/openai/v1`. gpt-oss is a reasoning model: if answers come back empty, raise `LLM_MAX_OUTPUT_TOKENS`.)

**Sandbox limits seen by the previous chat:** no access to Hugging Face (cannot download embedding models). PyPI works.
Always re-check what your own sandbox can reach before designing around it.

## 5. Architecture and file map

```
backend/
  app.py              create_app(services=None): registers routes, JSON errors (404/405/413/500), request-size guard
  config.py           class Config: every setting (env vars). Other modules never read os.environ
  services.py         Services: LAZY embedding_service / vector_store / retriever (health stays instant, tests inject fakes)
  models/             chunk.py (Chunk) document.py (ParsedDocument/Section/Block) search.py (SearchResult, IndexInfo, RetrievalResponse)
  rag/
    html_parser.py    HTML -> structured sections (ParseError)
    chunker.py        sections -> Chunk list; find_problems(); save/load_chunks
    embeddings.py     EmbeddingService (abstract) | SentenceTransformerService | HashingEmbeddingService | get_embedding_service()
    vector_store.py   VectorStore (ChromaDB wrapper, VectorStoreError) | get_vector_store()
    ingestion.py      ingest_document / reindex_document / index_is_current / compute_fingerprint
    retriever.py      Retriever.retrieve(question, top_k) -> RetrievalResponse
    prompt_builder.py build_prompt(question, chunks, max_context_chars) -> Prompt(system, user, sources, dropped_chunk_ids); NO_ANSWER_TEMPLATE / NO_ANSWER_GENERIC; PromptBuildError
    relevance.py      assess_relevance(results, threshold) -> RelevanceDecision(sufficient, best_similarity, passages at/above threshold, reason)
    answer_validator.py validate_answer(answer, passages, question) -> Verdict(ok | refused | rejected)
    rag_service.py    RAGService(retriever, llm).ask(question) -> RAGAnswer(status: answered|refused|no_evidence|rejected|unavailable, answer, sources, ...)
    llm.py            LLMService.generate(system, user) -> LLMResponse | FakeLLMService | AnthropicService | OpenAICompatibleService | get_llm_service() | LLMError family
    evaluation.py     question dataset loading + validation, metrics, EvaluationRun, report writer
  routes/             health.py  ingestion.py (POST /ingest, /reindex)  sources.py (GET /sources)   [all under /api]
  utils/logger.py     get_logger(name)
scripts/              inspect_html parse_company build_chunks check_embeddings ingest_company inspect_index search evaluate_retrieval
tests/                conftest.py + one test file per module (see section 3)
evaluation/           questions.json (hand-written, 49 questions); retrieval_results.json + evaluation_report.md are GENERATED
data/raw/company.html the knowledge source (never modified by code)
```

Generated, never commit or ship: `.env`, `data/processed/*.json`, `evaluation/retrieval_results.json`,
`evaluation/evaluation_report.md`, everything in `vector_db/chroma/`.

## 6. Conventions every step must follow (the "contract")

1. **Settings:** add to `Config` in `backend/config.py` AND `.env.example`; put the **unit in the name**
   (`CHUNK_MAX_WORDS`); give a default so an existing `.env` keeps working; document it in the README env table.
2. **Pattern for anything external (model, DB, LLM):** abstract interface + real implementation + offline/fake implementation
   + cached factory (`lru_cache`) + lazy creation through `Services`. One module owns each external library.
3. **Routes stay thin;** all logic lives in `backend/rag/`. Routes read services from `current_app.extensions["services"]`.
4. **Errors:** custom exception classes with messages that tell the user what to do. HTTP errors are JSON `{"status": ..., "error": ...}`
   with **no file paths or stack traces** (details go to the log). Known codes: 405, 409 (conflict/busy), 413, 500, 503.
5. **Logging:** `get_logger(__name__)`. Log sizes and scores, **never user question text, prompts or secrets.**
6. **Secrets:** only in `.env` (git-ignored). `.env.example` holds placeholders. Never in code, tests, logs, README or zips.
7. **Tests:** real code with offline substitutes; every test uses temporary folders/databases (fixtures in `tests/conftest.py`:
   `embedder`, `other_embedder`, `failing_embedder`, `store`, `chunks`, `doc_copy`). Never touch the user's real index. No network calls.
   Expected values for any maths are computed by hand. Plain `pytest` must work (`pytest.ini` sets `pythonpath = .`).
8. **Scripts:** each step ships a runnable script in `scripts/` so the user can SEE the behaviour. Scripts add the project root to
   `sys.path`; many accept `--provider` / `--model` overrides.
9. **Chroma facts:** cosine space; chunk ids look like `leave-policy-001`; metadata cannot contain `None`; duplicate ids are
   silently ignored by ChromaDB (we check ourselves); index stores `fingerprint`, `embedding_model`, `indexed_at` on every record.
10. **README:** keep the progress table, the folder tree (legend markers: `*` step 2, `+` 3, `~` 4, `^` 5, `#` 6-7, `@` 8-9; add a new
    marker per step), the env table, the run commands, API docs, limitations and the build log current.

## 7. Settings (`.env.example`)

`APPLICATION_ENV DEBUG HOST PORT` | `COMPANY_DOCUMENT_PATH CHUNKS_PATH CHROMA_PATH CHROMA_COLLECTION` |
`EMBEDDING_PROVIDER (sentence-transformers|hashing) EMBEDDING_MODEL EMBEDDING_BATCH_SIZE` |
`LLM_PROVIDER (fake|anthropic|openai|openai-compatible|ollama) LLM_MODEL (no default, by design) LLM_API_KEY (SECRET) LLM_BASE_URL LLM_TIMEOUT_SECONDS=30 LLM_TEMPERATURE=0 (empty = not sent) LLM_MAX_OUTPUT_TOKENS=600 LLM_MAX_RETRIES=2 LLM_RETRY_BACKOFF_SECONDS=1.0` (**used only by `llm.py`**) |
`TOP_K=5` | `CONTEXT_MAX_CHARS=12000` (Step 11: passage text per prompt) | `RELEVANCE_THRESHOLD=0.20` (Step 12: minimum cosine similarity; PROVISIONAL and deliberately low; the old placeholder 0.75 would refuse almost everything and may still sit in the user's own `.env`) | `CHUNK_MAX_WORDS=150 CHUNK_OVERLAP_WORDS=20`.
Also in `Config`: `MAX_CONTENT_LENGTH=16 KB`, `MAX_QUESTION_CHARS=1000`.

## 8. HTTP API (all under `/api`, server: `python -m backend.app`, port 5000)

| Method | Path | Status |
|---|---|---|
| GET | `/health` | done |
| POST | `/ingest` | done (idempotent; 409 if the index does not match document/model/settings) |
| POST | `/reindex` | done (embeds BEFORE deleting the old index) |
| GET | `/sources` | done (includes `up_to_date`) |
| POST | `/chat` | Step 13 |
| GET | `/config` | planned (must never expose secrets) |

**No authentication yet:** anyone who can reach the server can call `/ingest` and `/reindex`. Keep it local.

## 9. Key decisions and findings (the "why")

- Chunks: structure first, size second; 1 `<h3>` subsection = 1 chunk; split between blocks, never inside a table row or Q/A pair;
  overlap only when one block must be split; heading text is inside every chunk; sizes in WORDS (no tokenizer).
- Most facts in `company.html` appear **twice** (policy section and FAQ). The FAQ chunks often rank first. Both are valid evidence.
- Relevance in evaluation is decided by **evidence text**, not by section name.
- `company.html` contains traps on purpose: a hidden "sports club" note (must never be answered from), and absent topics
  (revenue, stock price, dental, sabbatical, ESOP, joining bonus, pet insurance, stipend). Individual salaries are confidential.
- The offline `hashing` embedder matches shared words only and produces many exact ties at 0.000; never base tests on tie order.
- ChromaDB telemetry is off.
- **LLM (Step 10):** providers are called with plain `urllib` (no vendor SDK, no new dependency), which lets the tests run the REAL network code against a local fake server.
  Provider quirks verified from current docs: Anthropic = `x-api-key` + `anthropic-version: 2023-06-01` + top-level `system` + required `max_tokens`;
  OpenAI = `max_completion_tokens`, and reasoning models (o-series, gpt-5+) reject `temperature` (we omit it automatically for them); their token budget includes hidden reasoning
  (a small `LLM_MAX_OUTPUT_TOKENS` can yield an empty answer: reported as `LLMResponseError`); Ollama and other local servers use `max_tokens` and ignore the key.
  There is deliberately NO default model name. Redirects are never followed (urllib would forward the key header); the key is never sent over plain http to a remote host;
  only http/https base URLs are accepted. User-facing failure text is always "The answer service is temporarily unavailable." (spec s.68).
  Error classes: Config / Auth / Request / ResponseError are permanent; RateLimit / Timeout / Unavailable are temporary (retried, bounded).

## 10. Change log (append one entry per step; list changes to EARLIER steps explicitly)

- Steps 1-9: see section 3. Changes made to earlier steps while doing later ones:
  - Step 7 modified `app.py`: the request-size limit now applies to every route (Flask only enforces it when a body is read).
  - Step 8-9 added `pytest.ini` (plain `pytest` previously failed with `ModuleNotFoundError`; instructions had only been tested with `python -m pytest`).
  - Step 4 renamed `CHUNK_SIZE/CHUNK_OVERLAP` to `CHUNK_MAX_WORDS/CHUNK_OVERLAP_WORDS` (old lines in a user's `.env` are ignored harmlessly).
  - Bugs found in the chat's own tests during mutation checks (vacuous tests) were rewritten.
- Hand-over (after Step 9): added `docs/PROJECT_STATE.md`, `docs/HANDOFF_PROMPT.md`, `scripts/refresh_project_state.py` and `tests/test_project_state.py`.
- Step 10 (LLM integration): added `backend/rag/llm.py`, `scripts/check_llm.py`, `tests/test_llm.py` (120 tests); changed `backend/config.py` (new `LLM_*` settings; existing
  `LLM_PROVIDER/MODEL/API_KEY` now stripped of spaces), `.env.example`, `backend/services.py` (lazy `llm_service`, optional `llm_service=` in the constructor), README, this file.
  No new dependency, no endpoint. No change to the behaviour of Steps 1-9. 20 deliberate breakages of `llm.py` were all caught by the tests (two early mutation runs used a sloppy output filter and were repeated: see trap 14).
- Step 11 (grounded prompt): added `backend/rag/prompt_builder.py`, `scripts/show_prompt.py`, `tests/test_prompt_builder.py` (76 tests); changed `backend/config.py` + `.env.example` (`CONTEXT_MAX_CHARS`), README, this file.
  **Changes to earlier steps:** `tests/conftest.py` blanks `LLM_PROVIDER/MODEL/API_KEY/BASE_URL` before `backend.config` loads `.env` (python-dotenv never overrides existing variables);
  `tests/test_llm.py::run_check` keeps blanks instead of deleting the variables; new test `test_the_test_session_is_isolated_from_the_developers_env_file`. Reason: with a real key in `.env`, two script tests
  would have made REAL provider calls from `pytest` (reproduced with a fake `.env`, then fixed). Three weak tests of my own (always-true assertions) were strengthened; 14 deliberate breakages of the prompt builder are all caught.
- Step 12 (refusal logic + answer validation): added `backend/rag/relevance.py`, `backend/rag/answer_validator.py`, `backend/rag/rag_service.py`, `scripts/ask.py`, `scripts/verify_step12.py`; changed `backend/config.py`
  and `.env.example` (`RELEVANCE_THRESHOLD` default **0.75 -> 0.20**), `backend/services.py` (lazy `rag_service`), README, this file, `tests/test_project_state.py` (the `announced` set is now empty). **No test was run, no pytest test was written** (user's request);
  everything was verified by the standalone `scripts/verify_step12.py`. Findings while building: number grounding checked against ALL passages let "30 paid leave days" through because "30 days" appears in the notice-period passage -> tightened to
  the best-matching passage; a list marker "1." was counted as a number after sentence splitting -> markers are stripped first; "31" was a bad test number (it appears as "31 March" in the same passage) -> documented as a limit.
- Step 13: _(next chat appends here)_

## 11. Open questions and risks

1. **(Partly resolved: the user ran `check_llm.py` successfully with Groq.)** Earlier text: Providers are implemented (anthropic, openai, ollama, openai-compatible; Gemini works through its OpenAI-compatible URL) but no real call has been verified.
   Steps 11-12 can be built and tested with `FakeLLMService`, but the first real end-to-end answer needs the user's configured provider.
2. Real-model evaluation numbers are unseen (section 4). Do not choose `RELEVANCE_THRESHOLD` without them.
3. Similar-but-incorrect questions (for example dental insurance) score as high as correct ones, so a threshold alone is not enough:
   the grounded prompt (Step 11) and answer validator (Step 12) must carry the "no evidence" rule.
4. Prompt injection: retrieved text is DATA, never instructions (spec section 65). Keep system instructions and retrieved content clearly separated.
5. No authentication on admin endpoints; reindex deletes the old index before writing (a failed write leaves it empty, never half-built).
6. **Step 12 has no pytest tests.** FIRST job of the next step: turn `scripts/verify_step12.py` into pytest tests (relevance boundary, validator cases, `RAGService` statuses with `FakeLLMService`), run them, then mutation-check the validator
   (for example: grounding against all passages, list markers counted, refusal not recognised). Also confirm the real-model behaviour of `scripts/ask.py` with the user.
7. The validator checks numbers and leaks, not meaning; the threshold is provisional. Re-tune with `evaluation/evaluation_report.md` when the user provides it.

## 12. Roadmap with specification pointers (`ProjectDocumentation.docx`, bold section numbers)

| Step | Read these sections |
|---|---|
| 10 LLM | 37 `llm.py`, 38 LLM provider, 82 Phase 9 |
| 11 Prompt | 35 `prompt_builder.py`, 36 grounded prompt structure, 65 prompt injection |
| 12 Relevance | 33 `relevance.py`, 34 important RAG rule, 39 `answer_validator.py`, 46 unsupported question, 64 hallucination prevention |
| 13 Chat API | 40 complete RAG service, 41 API design, 45 `POST /api/chat`, 46, 66-69 input/API security, errors, logging, 73 API vs RAG layer |
| 14 Frontend | 49-52 frontend, `index.html`, `style.css`, `app.js` |
| 15 Citations | 53-54 |
| 16-17 Testing / RAG evaluation | 55-63 |
| 18-20 | spec phases 16-20 and sections 74+ |

## 13. Handoff protocol (two Claude chats alternate steps)

1. The user starts a chat with the prompt in `docs/HANDOFF_PROMPT.md` and attaches: `ProjectDocumentation.docx` + the latest
   `ABCTechSmartBort_full_stepN.zip` (+ optionally the real-model `evaluation_report.md`, and says which LLM provider to use).
2. That chat does ONE step (or a few only if each is small AND fully verifiable), then returns: a **changes-only zip** (for the user's local project),
   a **full zip** (for the next chat), and the list of files to attach to the next chat. This file and the README are updated inside both zips.
3. The user takes the newest full zip to whichever chat does the next step. **Only one chat works at a time.** Chats never rely on memory of earlier
   conversations; everything they need is in the repository. If the code and this file disagree (for example the test count), STOP and tell the user.

## 14. Traps already hit (do not repeat them)

1. **Plain `pytest` failed** (`No module named backend`) until `pytest.ini` was added. Always check BOTH `pytest` and `python -m pytest`.
2. **ChromaDB** silently ignores duplicate ids, rejects `None` in metadata, needs collection names of 3+ characters, and raises an error for a different
   vector size (we turn it into "run reindex"). **Always probe a third-party library's real behaviour with a small script before designing around it.**
3. **Vacuous tests:** a test that faked a crash *before* anything was written passed even without the cleanup code. Only mutation testing exposed it.
4. The `hashing` embedder gives many **exact ties (0.000)**; tie order is arbitrary and differs between processes. Never assert on it.
5. **Noise protection hid behind `<main>`:** the parser only reads `<main>`, so header/footer/hidden text was never reached. A test that turns `<main>` into a `<div>`
   proves the removal rules themselves work.
6. The assistant's **sandbox cannot reach Hugging Face or any LLM API**: real models/LLMs cannot be exercised there. Test the glue with stubs and give the user a check
   script to run (`check_embeddings.py` is the model; expect `check_llm.py` to be the same).
7. **`pkill -f backend.app` kills your own shell** if that text is in the same command. Start a server with `&`, keep `$!`, then `kill $PID`.
8. **Never put `.env` or `__pycache__` in a zip.** An early zip contained `.env`; delete any `.env` you copied for testing before zipping.
9. `MAX_CONTENT_LENGTH` is only enforced when a route reads the body: the `before_request` check in `app.py` covers every route.
10. Changing the embedding model, chunk settings or `company.html` requires `POST /api/reindex` (or `scripts/ingest_company.py --reindex`).
11. `sentence-transformers` pulls PyTorch (several GB of CUDA libraries on Linux). For sandbox verification create a venv from
    `grep -v sentence-transformers requirements.txt`; the offline embedder covers all tests.
12. **A metric or label you wrote yourself can be wrong:** validate evaluation labels against real data and recompute metrics independently (this found a
    mislabelled multi-section question).
13. **urllib follows redirects and forwards custom headers** (the API key header would reach the new address). `llm.py` installs a no-redirect handler; keep it.
14. **Check your own test harness before trusting a result:** a double `-q`, `-p no:logging` (disables `caplog`, turning tests into ERRORS) and a grep that matched log lines containing
    "failed" each made a mutation run look inconclusive. Use `pytest -q -rf` and read the summary line.
15. **Provider APIs have quirks that change** (OpenAI `max_tokens` vs `max_completion_tokens`; reasoning models reject `temperature`; hidden reasoning tokens use the output budget). Verify against current docs.
16. **Tests inherit the developer's `.env`:** `backend/config.py` runs `load_dotenv`, so child processes (scripts launched by tests) and in-process imports can pick up a REAL LLM key and make billable calls. `tests/conftest.py` now blanks `LLM_*` first. Any new test helper that launches a script must keep those blanks (set to "", do not delete).

## 15. What to do NEXT: Step 13, the chat API (`POST /api/chat`)

Specification: doc s.41 (API design), s.45 (`POST /api/chat`), s.46 (unsupported question), s.66-69 (input and API security, errors, logging), s.73 (API layer vs RAG layer), s.40 (the complete RAG service, already built as `RAGService`).

What to build:
- A thin Flask route (new blueprint registered under `/api`, same pattern as the ingestion and sources routes) that calls `Services.rag_service.ask(question)`. All logic stays in the RAG layer.
- Request: JSON `{"question": "..."}`. Validate: JSON content type, `question` is a non-empty string, at most `Config.MAX_QUESTION_CHARS`; 400 with a clear JSON error otherwise. The 16 KB body limit already applies.
- Response (200) per the spec: the answer text, a `status` (answered / refused / no_evidence / rejected / unavailable), `sources` (label, heading, source, document version) only when `answered`, and the best similarity.
  Decide and document how each status maps to HTTP (refusals are normal answers, not errors).
- Errors as JSON without internals: empty index or embedding-model mismatch from the retriever (503, tell the user to build the index), LLM misconfiguration (503, generic text), anything unexpected (500, generic). Never return provider messages, paths or prompts. Log counts, never question text.
- `GET /api/config` (spec): safe settings only, NEVER a secret (no key, no base URL with credentials).
- Per the user's standing request: copy-paste instructions for curl (Windows cmd), PowerShell (`Invoke-RestMethod`, quoting for JSON), Postman and the browser, tested against a live server; include the error table.
- Tests FIRST (see section 11, item 6): convert `scripts/verify_step12.py` into pytest tests, then test the route with the Flask test client and `Services(llm_service=FakeLLMService, ...)`.
- Out of scope: the frontend (Step 14) and citation formatting (Step 15).

## 16. How to finish a step (mandatory checklist)

1. Update section 3 (status, "Key facts", "Tests per file" line), sections 4, 5, 7-9, 11 and 15 if anything changed. Replace section 15 with the next step's brief.
2. Append the step to section 10 (and list every change made to EARLIER steps).
3. If the new section 15 names scripts that do not exist yet, add them to the `announced` set in `tests/test_project_state.py` (and remove the ones that now exist), otherwise the guard test fails.
   Then run `python scripts/refresh_project_state.py` (rewrites section 17) and the whole test suite (`pytest` and `python -m pytest`).
4. Ship: changes-only zip + full zip (`ABCTechSmartBort_full_stepN.zip`) + the exact list of files the user must give the next chat.

## 17. Inventory (generated: do not edit by hand)

<!-- AUTO:INVENTORY:START -->
_Generated by `python scripts/refresh_project_state.py`. Do not edit by hand._

**Project files** (the text is the first line of the file's docstring)

| File | What it is |
|---|---|
| `.env.example` |  |
| `.gitignore` |  |
| `README.md` |  |
| `backend/__init__.py` |  |
| `backend/app.py` | Flask application entry point. |
| `backend/config.py` | Central configuration. |
| `backend/models/__init__.py` |  |
| `backend/models/chunk.py` | The Chunk data structure (Step 4). |
| `backend/models/document.py` | Data structures produced by the HTML parser (Step 3). |
| `backend/models/search.py` | Data structures returned by the vector store (Steps 6-7). |
| `backend/rag/__init__.py` |  |
| `backend/rag/answer_validator.py` | Step 12 - Answer validator: check the LLM's answer BEFORE the user sees it. |
| `backend/rag/chunker.py` | Step 4 - Chunker. |
| `backend/rag/embeddings.py` | Step 5 - Embeddings. |
| `backend/rag/evaluation.py` | Step 9 - Retrieval evaluation. |
| `backend/rag/html_parser.py` | Step 3 - HTML parser. |
| `backend/rag/ingestion.py` | Step 7 - Ingestion: from company.html to a searchable index. |
| `backend/rag/llm.py` | Step 10 - LLM integration. |
| `backend/rag/prompt_builder.py` | Step 11 - Grounded prompt builder. |
| `backend/rag/rag_service.py` | Step 12 - The complete RAG pipeline for ONE question (specification section 40). |
| `backend/rag/relevance.py` | Step 12 - Relevance: is there enough evidence to even ask the LLM? |
| `backend/rag/retriever.py` | Step 8 - Retriever: from a question to the closest chunks. |
| `backend/rag/vector_store.py` | Step 6 - Vector store (ChromaDB). |
| `backend/routes/__init__.py` |  |
| `backend/routes/health.py` | GET /api/health - tells us the backend is alive. |
| `backend/routes/ingestion.py` | POST /api/ingest and POST /api/reindex. |
| `backend/routes/sources.py` | GET /api/sources - what the knowledge base currently contains. |
| `backend/services.py` | Shared services (embedding model, vector database, retriever, LLM) for the Flask app. |
| `backend/utils/__init__.py` |  |
| `backend/utils/logger.py` | Logging helper. Every module gets its logger via get_logger(__name__). |
| `data/raw/company.html` |  |
| `docs/HANDOFF_PROMPT.md` |  |
| `docs/PROJECT_STATE.md` |  |
| `docs/html_structure.md` |  |
| `evaluation/questions.json` |  |
| `pytest.ini` |  |
| `requirements.txt` |  |
| `scripts/ask.py` | Step 12 helper: ask the WHOLE pipeline a question and watch every stage. |
| `scripts/build_chunks.py` | Step 4 helper: build chunks from company.html and look at them. |
| `scripts/check_embeddings.py` | Step 5 helper: check that embeddings work. |
| `scripts/check_llm.py` | Step 10 helper: check that your LLM provider works. YOU run this with your real key. |
| `scripts/evaluate_retrieval.py` | Step 9 helper: measure how well the retriever finds the right chunks. |
| `scripts/ingest_company.py` | Step 7 helper: build the search index WITHOUT starting the Flask server. |
| `scripts/inspect_html.py` | Step 2 helper: inspect the structure of the company HTML document. |
| `scripts/inspect_index.py` | Step 6/7 helper: look inside the vector database and verify that the records exist. |
| `scripts/parse_company.py` | Step 3 helper: run the HTML parser and look at what it produces. |
| `scripts/refresh_project_state.py` | Keep the machine-written part of docs/PROJECT_STATE.md accurate. |
| `scripts/search.py` | Step 8 helper: ask the retriever a question and inspect what it finds. |
| `scripts/show_prompt.py` | Step 11 helper: see the EXACT prompt the LLM would receive, and optionally ask your real model. |
| `scripts/verify_step12.py` | Step 12 verification: 41 checks of the relevance gate, the answer validator and the whole RAG ser... |
| `tests/conftest.py` | Shared test fixtures. |
| `tests/test_api.py` |  |
| `tests/test_chunker.py` | Step 4 tests: the chunker. |
| `tests/test_company_document.py` | Step 2 tests: is data/raw/company.html a good knowledge source? |
| `tests/test_embeddings.py` | Step 5 tests: the embedding service. |
| `tests/test_evaluation.py` | Step 9 tests: retrieval evaluation. |
| `tests/test_html_parser.py` | Step 3 tests: the HTML parser. |
| `tests/test_ingestion.py` | Step 7 tests: the ingestion pipeline (parse -> chunk -> embed -> store). |
| `tests/test_ingestion_api.py` | Step 7 tests: the HTTP endpoints. |
| `tests/test_llm.py` | Step 10 tests: the LLM service. |
| `tests/test_project_state.py` | Tests for the hand-over tooling: scripts/refresh_project_state.py and the two documents |
| `tests/test_prompt_builder.py` | Step 11 tests: the grounded prompt builder. |
| `tests/test_retrieval.py` | Step 8 tests: the retriever. |
| `tests/test_vector_store.py` | Step 6 tests: the ChromaDB vector store. |

**Tests: 602 in total**

| Test file | Tests |
|---|---|
| `tests/test_api.py` | 2 |
| `tests/test_chunker.py` | 45 |
| `tests/test_company_document.py` | 27 |
| `tests/test_embeddings.py` | 42 |
| `tests/test_evaluation.py` | 67 |
| `tests/test_html_parser.py` | 62 |
| `tests/test_ingestion.py` | 34 |
| `tests/test_ingestion_api.py` | 19 |
| `tests/test_llm.py` | 121 |
| `tests/test_project_state.py` | 29 |
| `tests/test_prompt_builder.py` | 76 |
| `tests/test_retrieval.py` | 39 |
| `tests/test_vector_store.py` | 39 |
| **Total** | **602** |
<!-- AUTO:INVENTORY:END -->
