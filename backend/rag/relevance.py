"""Step 12 - Relevance: is there enough evidence to even ask the LLM?

Runs BEFORE the LLM. If no retrieved passage is similar enough to the question,
the LLM is not called at all and the fixed "I couldn't find information ..." message is
used (cheaper, faster, and one less chance for a model to make something up).

HONEST LIMIT: this is a COARSE filter. Questions about a RELATED topic that the document
never states (for example "dental insurance" when the document describes health insurance)
score as high as correct questions, so no threshold can stop them. The grounded prompt
(Step 11) and the answer validator carry that part of the "NO EVIDENCE -> NO ANSWER" rule.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from backend.config import Config
from backend.utils.logger import get_logger

logger = get_logger(__name__)

SUSPICIOUS_THRESHOLD = 0.60     # real scores rarely get this high with the default model


@dataclass
class RelevanceDecision:
    sufficient: bool
    best_similarity: Optional[float]
    threshold: float
    passages: List = field(default_factory=list)       # the passages AT OR ABOVE the threshold, best first
    reason: str = ""


def assess_relevance(results: Sequence, threshold: Optional[float] = None) -> RelevanceDecision:
    """Keep the passages whose similarity is at least `threshold` (a passage exactly at it is kept)."""
    threshold = Config.RELEVANCE_THRESHOLD if threshold is None else threshold
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) \
            or not -1.0 <= threshold <= 1.0:
        raise ValueError("the relevance threshold must be a number between -1 and 1")
    if threshold > SUSPICIOUS_THRESHOLD:
        logger.warning("RELEVANCE_THRESHOLD is %.2f. That is very high for real similarity scores (usually 0.2-0.6) "
                       "and will refuse most questions. Is this the old placeholder 0.75? Set it in .env.", threshold)

    if not results:
        return RelevanceDecision(False, None, threshold, [], "no passages were retrieved")
    best = max(r.similarity for r in results)
    kept = [r for r in results if r.similarity >= threshold]
    if not kept:
        return RelevanceDecision(False, best, threshold, [],
                                 f"the best similarity {best:.3f} is below the threshold {threshold:.2f}")
    return RelevanceDecision(True, best, threshold, kept,
                             f"{len(kept)} of {len(results)} passages are at or above the threshold {threshold:.2f}")
