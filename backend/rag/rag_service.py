"""Step 12 - The complete RAG pipeline for ONE question (specification section 40).

    question -> retrieve -> relevance check -> grounded prompt -> LLM -> answer validation -> answer

Every outcome is a RAGAnswer with a `status`, so callers (the terminal script now, the
chat API in Step 13) never have to guess what happened:

    answered      the LLM answered and the validator accepted it; `sources` lists the passages used
    refused       the LLM itself said it could not find the answer; no sources are shown
    no_evidence   no passage was similar enough: the LLM was NOT called; fixed refusal message
    rejected      the LLM answered but the validator refused to show it; fixed refusal message
    unavailable   the LLM failed (timeout, bad key ...): a SAFE message, never provider details

Problems that are not about this question (empty index, wrong embedding model, a bad question)
are raised, not hidden: the caller decides how to report them.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from backend.config import Config
from backend.rag.answer_validator import Verdict, validate_answer
from backend.rag.citation_validator import validate_citations
from backend.rag.llm import LLMError, LLMService
from backend.rag.prompt_builder import (
    NO_ANSWER_GENERIC,
    Prompt,
    PromptSource,
    build_prompt,
)
from backend.rag.relevance import RelevanceDecision, assess_relevance
from backend.rag.retriever import Retriever
from backend.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    status: str
    sources: List[PromptSource] = field(default_factory=list)
    best_similarity: Optional[float] = None
    llm_called: bool = False
    reason: str = ""
    seconds: float = 0.0
    retrieved: List = field(default_factory=list)
    decision: Optional[RelevanceDecision] = None
    prompt: Optional[Prompt] = None
    verdict: Optional[Verdict] = None
    llm_error: Optional[str] = None


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMService,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.threshold = threshold
        self.top_k = top_k

    def ask(self, question: str) -> RAGAnswer:
        started = time.perf_counter()

        response = self.retriever.retrieve(
            question,
            self.top_k,
        )

        decision = assess_relevance(
            response.results,
            self.threshold,
        )

        def done(answer: RAGAnswer) -> RAGAnswer:
            answer.question = response.question
            answer.retrieved = response.results
            answer.decision = decision
            answer.best_similarity = decision.best_similarity
            answer.seconds = time.perf_counter() - started

            logger.info(
                "Question answered: status=%s llm_called=%s "
                "passages_used=%d best_similarity=%s seconds=%.2f",
                answer.status,
                answer.llm_called,
                len(answer.sources),
                answer.best_similarity,
                answer.seconds,
            )

            return answer

        # ---------------------------------------------------------
        # 1. No sufficient evidence
        # ---------------------------------------------------------

        if not decision.sufficient:
            return done(
                RAGAnswer(
                    response.question,
                    NO_ANSWER_GENERIC,
                    "no_evidence",
                    reason=decision.reason,
                )
            )

        # ---------------------------------------------------------
        # 2. Build grounded prompt
        # ---------------------------------------------------------

        prompt = build_prompt(
            response.question,
            decision.passages,
        )

        used_ids = {source.chunk_id for source in prompt.sources}

        used = [
            passage
            for passage in decision.passages
            if passage.chunk_id in used_ids
        ]

        # ---------------------------------------------------------
        # 3. Ask the LLM
        # ---------------------------------------------------------

        try:
            reply = self.llm.generate(
                prompt.system,
                prompt.user,
            )

        except LLMError as exc:
            return done(
                RAGAnswer(
                    response.question,
                    exc.user_message,
                    "unavailable",
                    llm_called=True,
                    prompt=prompt,
                    reason="the language model could not be reached or failed",
                    llm_error=type(exc).__name__,
                )
            )

        # ---------------------------------------------------------
        # 4. Validate the answer
        # ---------------------------------------------------------

        verdict = validate_answer(
            reply.text,
            used,
            response.question,
        )

        # ---------------------------------------------------------
        # 5. Validator rejected the answer
        # ---------------------------------------------------------

        if verdict.status == "rejected":
            return done(
                RAGAnswer(
                    response.question,
                    NO_ANSWER_GENERIC,
                    "rejected",
                    llm_called=True,
                    prompt=prompt,
                    verdict=verdict,
                    reason=verdict.reason,
                )
            )

        # ---------------------------------------------------------
        # 6. LLM explicitly refused
        # ---------------------------------------------------------

        if verdict.status == "refused":
            return done(
                RAGAnswer(
                    response.question,
                    reply.text,
                    "refused",
                    llm_called=True,
                    prompt=prompt,
                    verdict=verdict,
                    reason=verdict.reason,
                )
            )

        # ---------------------------------------------------------
        # 7. Validate source citations
        # ---------------------------------------------------------

        citation_decision = validate_citations(
            reply.text,
            [source.label for source in prompt.sources],
        )

        if not citation_decision.valid:
            return done(
                RAGAnswer(
                    response.question,
                    NO_ANSWER_GENERIC,
                    "rejected",
                    llm_called=True,
                    prompt=prompt,
                    verdict=verdict,
                    reason=citation_decision.reason,
                )
            )

        # ---------------------------------------------------------
        # 8. Successful answer
        # ---------------------------------------------------------

        return done(
            RAGAnswer(
                response.question,
                reply.text,
                "answered",
                sources=prompt.sources,
                llm_called=True,
                prompt=prompt,
                verdict=verdict,
                reason=verdict.reason,
            )
        )