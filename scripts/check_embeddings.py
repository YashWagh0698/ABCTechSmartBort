"""Step 5 helper: check that embeddings work.

Run from the project root:

    python scripts/check_embeddings.py                       use the model from .env (real model)
    python scripts/check_embeddings.py --provider hashing    use the tiny offline stand-in
    python scripts/check_embeddings.py --top-k 5             show more results per question

What it does (in the order the project documentation recommends):
    [1] load the model
    [2] ONE chunk -> ONE embedding               (before embedding everything)
    [3] similarity sanity check                  (similar meaning = higher score?)
    [4] embed ALL chunks
    [5] token-limit check                        (are any chunks longer than the model reads?)
    [6] ranking preview                          (which chunk is closest to each question?)

The first run with a real model DOWNLOADS it (about 90 MB, needs internet).
"""
import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Config                                          # noqa: E402
from backend.rag.chunker import chunk_document                            # noqa: E402
from backend.rag.embeddings import (EmbeddingError, cosine_similarity,    # noqa: E402
                                    get_embedding_service)
from backend.rag.html_parser import parse_html_file                       # noqa: E402

# (question, headings that count as a correct answer location)
QUESTIONS = [
    ("How many paid leaves do employees get?", ["Leave Policy > Annual Paid Leave"]),
    ("What are the working hours?", ["Working Hours and Attendance > Standard Working Hours"]),
    ("When is my salary credited?", ["Salary and Compensation > Pay Cycle"]),
    ("How long is the notice period?", ["HR Information > Notice Period"]),
    ("How much is the health insurance cover?", ["Benefits > Health and Insurance"]),
    ("Which days do I have to come to the office?", ["Work Model > Hybrid Work Policy"]),
    ("Can I get maternity leave?", ["Leave Policy > Special Leave"]),
    ("How do I claim travel expenses?", ["Internal Processes > How to Claim an Expense Reimbursement",
                                         "Employee Policies > Travel and Expense Policy"]),
    # Same meaning as the leave question, but almost NO shared words:
    ("How much time off do I get each year?", ["Leave Policy > Annual Paid Leave"]),
    ("My laptop got stolen, what should I do?", ["Employee Policies > Confidentiality and Information Security",
                                                  "Frequently Asked Questions"]),
]

# Questions whose answer is NOT in the document.
UNANSWERABLE = [
    "What was the company's revenue last year?",
    "What is the company's stock price?",
    "Does the company offer dental insurance?",
]

# (question, text that SHOULD be closer, text that should be further)
SANITY_TRIPLES = [
    ("How many paid leaves do employees get?",
     "Employees receive 24 paid leave days per year.",
     "Salary is credited on the last working day of each month."),
    ("How much time off do I get each year?",
     "Employees receive 24 paid leave days per year.",
     "Salary is credited on the last working day of each month."),
    ("When do I get my pay?",
     "Salary is credited on the last working day of each month.",
     "Employees receive 24 paid leave days per year."),
]


def heading(chunk) -> str:
    return chunk.text.split("\n", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Check the embedding service")
    ap.add_argument("--provider", help="sentence-transformers or hashing (default: from .env)")
    ap.add_argument("--model", help="model name (default: from .env)")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    if args.provider:
        Config.EMBEDDING_PROVIDER = args.provider.strip().lower()
    if args.model:
        Config.EMBEDDING_MODEL = args.model.strip()

    line = "=" * 92
    print(line)
    print("EMBEDDING CHECK")
    print(line)
    technical_failures = []

    # ---- [1] model ----------------------------------------------------------
    print("\n[1] LOADING THE MODEL")
    started = time.perf_counter()
    try:
        service = get_embedding_service()
    except (EmbeddingError, ValueError) as exc:
        raise SystemExit(f"\n    ERROR: {exc}")
    print(f"    provider        : {Config.EMBEDDING_PROVIDER}")
    print(f"    model           : {service.model_name}")
    print(f"    numbers per text: {service.dimension}")
    print(f"    load time       : {time.perf_counter() - started:.1f} s")
    if Config.EMBEDDING_PROVIDER == "hashing":
        print("    NOTE: 'hashing' only matches shared WORDS. It cannot understand meaning.")

    # chunks come straight from the parser + chunker (not from a possibly stale file)
    chunks = chunk_document(parse_html_file(Config.COMPANY_DOCUMENT_PATH))

    # ---- [2] one chunk -> one embedding ------------------------------------
    print("\n[2] ONE CHUNK -> ONE EMBEDDING")
    one = chunks[18]                               # Leave Policy > Annual Paid Leave
    vector = service.embed_documents([one.text])
    length = math.sqrt(sum(x * x for x in vector[0]))
    print(f"    chunk           : {one.chunk_id}  ({heading(one)})")
    print(f"    embeddings back : {len(vector)}   (expected 1)")
    print(f"    vector length   : {len(vector[0])}   (expected {service.dimension})")
    print(f"    first 5 numbers : {[round(x, 3) for x in vector[0][:5]]}")
    print(f"    vector norm     : {length:.4f}   (expected 1.0000)")
    if len(vector) != 1 or len(vector[0]) != service.dimension or abs(length - 1) > 1e-3:
        technical_failures.append("one-chunk test")

    # ---- [3] similarity sanity ---------------------------------------------
    print("\n[3] SIMILARITY SANITY CHECK  (the closer text should get the higher score)")
    passed = 0
    for question, closer, further in SANITY_TRIPLES:
        q, a, b = service.embed_documents([question, closer, further])
        s_close, s_far = cosine_similarity(q, a), cosine_similarity(q, b)
        ok = s_close > s_far
        passed += ok
        print(f"    {'PASS' if ok else 'FAIL'}  \"{question}\"")
        print(f"          closer text: {s_close:+.3f}    further text: {s_far:+.3f}")
    print(f"    -> {passed}/{len(SANITY_TRIPLES)} passed")

    # ---- [4] all chunks ----------------------------------------------------
    print("\n[4] EMBEDDING ALL CHUNKS")
    started = time.perf_counter()
    vectors = service.embed_chunks(chunks)
    seconds = time.perf_counter() - started
    bad_norm = sum(1 for v in vectors if abs(math.sqrt(sum(x * x for x in v)) - 1) > 1e-3)
    print(f"    chunks          : {len(chunks)}")
    print(f"    vectors         : {len(vectors)}   (must equal the number of chunks)")
    print(f"    all have {service.dimension} numbers: {all(len(v) == service.dimension for v in vectors)}")
    print(f"    vectors with norm != 1: {bad_norm}")
    print(f"    time            : {seconds:.2f} s")
    if len(vectors) != len(chunks) or bad_norm or not all(len(v) == service.dimension for v in vectors):
        technical_failures.append("embedding all chunks")

    # ---- [5] token limit ---------------------------------------------------
    print("\n[5] TOKEN-LIMIT CHECK")
    counts = service.token_counts([c.text for c in chunks])
    if counts is None or service.max_tokens is None:
        print("    (this provider has no tokenizer, nothing to check)")
    else:
        limit = service.max_tokens
        too_long = [(c, n) for c, n in zip(chunks, counts) if n > limit]
        print(f"    the model reads at most {limit} tokens; the longest chunk has {max(counts)} tokens")
        if too_long:
            print(f"    WARNING: {len(too_long)} chunk(s) are longer than that. The extra text is NOT embedded:")
            for c, n in too_long:
                print(f"      - {c.chunk_id}: {n} tokens")
            print("    Fix: lower CHUNK_MAX_WORDS in .env and run scripts/build_chunks.py again.")
        else:
            print("    OK: every chunk fits completely.")

    # ---- [6] ranking preview -----------------------------------------------
    print(f"\n[6] RANKING PREVIEW  (top {args.top_k} chunks for each question)")
    hits = 0
    for question, accepted in QUESTIONS:
        qv = service.embed_query(question)
        ranked = sorted(((cosine_similarity(qv, v), c) for v, c in zip(vectors, chunks)),
                        key=lambda item: item[0], reverse=True)[:args.top_k]
        found = any(any(heading(c).startswith(a) for a in accepted) for _, c in ranked)
        hits += found
        print(f"\n    {'HIT ' if found else 'MISS'}  \"{question}\"")
        for rank, (score, c) in enumerate(ranked, 1):
            print(f"          {rank}. {score:+.3f}  {heading(c)}")
    print(f"\n    -> the right chunk was in the top {args.top_k} for {hits}/{len(QUESTIONS)} questions")

    print("\n    Questions with NO answer in the document (look at the best score):")
    for question in UNANSWERABLE:
        qv = service.embed_query(question)
        score, c = max(((cosine_similarity(qv, v), c) for v, c in zip(vectors, chunks)),
                       key=lambda item: item[0])
        print(f"          best {score:+.3f}  {heading(c):<50}  \"{question}\"")
    print("    Step 12 will use scores like these to decide when the bot must refuse.")

    print("\n" + line)
    if technical_failures:
        print("RESULT: PROBLEM in: " + ", ".join(technical_failures))
        sys.exit(1)
    print("RESULT: embeddings work (one chunk, all chunks, vector sizes and norms are correct)")
    print(line)


if __name__ == "__main__":
    main()
