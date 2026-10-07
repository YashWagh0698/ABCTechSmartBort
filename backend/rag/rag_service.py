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
from backend.rag.llm import LLMError, LLMService
from backend.rag.prompt_builder import NO_ANSWER_GENERIC, Prompt, PromptSource, build_prompt
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
    reason: str = ""                      # short, safe explanation of the outcome
    seconds: float = 0.0
    retrieved: List = field(default_factory=list)
    decision: Optional[RelevanceDecision] = None
    prompt: Optional[Prompt] = None
    verdict: Optional[Verdict] = None
    llm_error: Optional[str] = None       # the error CLASS name only, never a message


class RAGService:
    def __init__(self, retriever: Retriever, llm: LLMService, threshold: Optional[float] = None,
                 top_k: Optional[int] = None):
        self.retriever = retriever
        self.llm = llm
        self.threshold = threshold            # None = read Config.RELEVANCE_THRESHOLD at question time
        self.top_k = top_k

    def ask(self, question: str) -> RAGAnswer:
        started = time.perf_counter()
        response = self.retriever.retrieve(question, self.top_k)            # may raise: index / model / bad question
        decision = assess_relevance(response.results, self.threshold)

        def done(answer: RAGAnswer) -> RAGAnswer:
            answer.question, answer.retrieved, answer.decision = response.question, response.results, decision
            answer.best_similarity = decision.best_similarity
            answer.seconds = time.perf_counter() - started
            logger.info("Question answered: status=%s llm_called=%s passages_used=%d best_similarity=%s seconds=%.2f",
                        answer.status, answer.llm_called, len(answer.sources), answer.best_similarity, answer.seconds)
            return answer

        if not decision.sufficient:
            return done(RAGAnswer(response.question, NO_ANSWER_GENERIC, "no_evidence", reason=decision.reason))

        prompt = build_prompt(response.question, decision.passages)
        used_ids = {s.chunk_id for s in prompt.sources}
        used = [p for p in decision.passages if p.chunk_id in used_ids]
        try:
            reply = self.llm.generate(prompt.system, prompt.user)
        except LLMError as exc:
            return done(RAGAnswer(response.question, exc.user_message, "unavailable", llm_called=True, prompt=prompt,
                                  reason="the language model could not be reached or failed", llm_error=type(exc).__name__))

        verdict = validate_answer(reply.text, used, response.question)
        if verdict.status == "rejected":
            return done(RAGAnswer(response.question, NO_ANSWER_GENERIC, "rejected", llm_called=True, prompt=prompt,
                                  verdict=verdict, reason=verdict.reason))
        if verdict.status == "refused":
            return done(RAGAnswer(response.question, reply.text, "refused", llm_called=True, prompt=prompt,
                                  verdict=verdict, reason=verdict.reason))
        return done(RAGAnswer(response.question, reply.text, "answered", sources=prompt.sources, llm_called=True,
                              prompt=prompt, verdict=verdict, reason=verdict.reason))
