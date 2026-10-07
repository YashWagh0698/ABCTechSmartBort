"""Step 12 - Answer validator: check the LLM's answer BEFORE the user sees it.

An LLM can still ignore its instructions. This is a second line of defence with simple,
EXPLAINABLE checks (specification sections 39 and 64):

  1. not empty, not absurdly long
  2. no leaked rules: no long run of words copied from the system message, no <context> /
     <question> tags
  3. NUMBER GROUNDING: every number in the answer (days, amounts, times, years) must come from the passage
     that BEST MATCHES THE WORDING of the sentence it appears in (or from the question). Checking against
     all passages at once is not enough: "30 paid leave days" would pass just because "30 days" appears in an
     unrelated passage (the probation notice period). Invented numbers are the most damaging kind of
     hallucination, and this check catches them without understanding language.
  4. a REFUSAL ("I couldn't find information about ...") is recognised and is a valid outcome.

HONEST LIMITS: this does NOT check meaning. A wrong claim that uses no new numbers (wrong names,
reversed policies) passes. A number that happens to appear in the SAME passage but belongs to another fact passes ("31 paid leave
days" next to "31 March"): grounding is per passage, not per claim. A correct answer that computes a new number,
or combines numbers from two different passages in ONE sentence, is rejected, which is the safe direction. Numbers written as words are not checked.

Verdicts: "ok" (show it), "refused" (the model said it could not find the answer: show that),
"rejected" (do NOT show it; the caller falls back to the fixed refusal message).
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

from backend.rag.prompt_builder import SYSTEM_PROMPT

MAX_ANSWER_CHARS = 1500
LEAK_WORDS = 7                      # this many words in a row copied from the system message = a leak

_REFUSAL = re.compile(r"I couldn['\u2019]t find information about .{1,300}? in the available company documentation\.?",
                      re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<\s*/?\s*(context|question)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:[.:]\d+)*")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")
_PASSAGE_LABEL = re.compile(r"\[\d+\]")
_WORD = re.compile(r"[a-z0-9']+")
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset("the and for are was were been its this that with from per you your our they their can will may must not does".split())


@dataclass
class Verdict:
    status: str                                   # "ok" | "refused" | "rejected"
    reason: str = ""
    ungrounded_numbers: List[str] = field(default_factory=list)
    partial_refusal: bool = False                 # an answer that gives some facts and says another part was not found

    @property
    def show_answer(self) -> bool:
        return self.status in ("ok", "refused")


def _numbers(text: str) -> Set[str]:
    """Numbers in a text, normalised: '5,00,000' -> '500000'; '9:00' also counts as '9'."""
    text = _PASSAGE_LABEL.sub(" ", _LIST_MARKER.sub("", text))
    found: Set[str] = set()
    for match in _NUMBER.finditer(text):
        token = match.group().replace(",", "").rstrip(".:")
        if not token:
            continue
        found.add(token)
        if ":" in token:                          # a time like 9:00 may be written "9 AM" in the answer
            hour, _, minutes = token.partition(":")
            found.add(hour)
            if set(minutes) <= {"0"}:
                found.add(f"{hour}:{minutes}")
    return found


def _words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _content_words(text: str) -> Set[str]:
    return {w for w in _words(text) if len(w) > 2 and w not in _STOPWORDS and not w.isdigit()}


def _ungrounded_numbers(text: str, passage_texts: List[str], question_numbers: Set[str]) -> List[str]:
    """Numbers that are not backed by the passage that best matches the sentence they appear in."""
    bad: Set[str] = set()
    passage_numbers = [_numbers(t) for t in passage_texts]
    passage_words = [_content_words(t) for t in passage_texts]
    text = _PASSAGE_LABEL.sub(" ", _LIST_MARKER.sub("", text))       # '1.' list markers and [2] labels are not numbers
    for sentence in _SENTENCES.split(text):
        needed = _numbers(sentence) - question_numbers
        if not needed:
            continue
        words = _content_words(sentence)
        overlaps = [len(words & pw) for pw in passage_words]
        best = max(overlaps, default=0)
        if best == 0:                                   # nothing to match the wording against: accept any passage
            candidates = list(range(len(passage_texts)))
        else:
            candidates = [i for i, o in enumerate(overlaps) if o == best]
        if not any(needed <= passage_numbers[i] for i in candidates):
            bad |= needed - set().union(*(passage_numbers[i] for i in candidates)) if candidates else needed
            if not bad:                                  # numbers exist in the candidates but never together
                bad |= needed
    return sorted(bad)


_RULE_WINDOWS = {tuple(w[i:i + LEAK_WORDS]) for w in [_words(SYSTEM_PROMPT)] for i in range(len(w) - LEAK_WORDS + 1)
                 if "topic" not in w[i:i + LEAK_WORDS]}


def validate_answer(answer: str, passages: Sequence, question: str = "") -> Verdict:
    """Judge an answer against the passages that were placed in the prompt.

    passages: the passage objects (with .text) or plain strings that the model was given.
    """
    if not isinstance(answer, str) or not answer.strip():
        return Verdict("rejected", "the answer is empty")
    answer = answer.strip()
    if len(answer) > MAX_ANSWER_CHARS:
        return Verdict("rejected", f"the answer is unusually long ({len(answer)} characters)")
    if _TAGS.search(answer):
        return Verdict("rejected", "the answer contains <context> or <question> tags (echoed prompt structure)")
    answer_words = _words(answer)
    for i in range(len(answer_words) - LEAK_WORDS + 1):
        if tuple(answer_words[i:i + LEAK_WORDS]) in _RULE_WINDOWS:
            return Verdict("rejected", "the answer repeats the system rules")

    # a pure refusal is a valid outcome; a refusal next to real content is a partial answer
    remainder = _REFUSAL.sub(" ", answer)
    had_refusal = remainder != answer
    if had_refusal and not re.search(r"\w", remainder):
        return Verdict("refused", "the model said the information is not in the documentation")

    passage_texts = [p.text if hasattr(p, "text") else str(p) for p in passages]
    ungrounded = _ungrounded_numbers(remainder, passage_texts, _numbers(question))
    if ungrounded:
        return Verdict("rejected", "the answer contains numbers that the matching passage does not support: " + ", ".join(ungrounded),
                       ungrounded_numbers=ungrounded)
    return Verdict("ok", "partial answer: part of the question was not found" if had_refusal else "numbers are grounded",
                   partial_refusal=had_refusal)
