"""Evaluation metrics for clinical speech recognition.

* WER (word error rate): the standard ASR metric. (S + D + I) / N, where S, D, I
  are word substitutions, deletions and insertions against N reference words.
* Medical-term recall: of the medical terms said in the reference, what fraction
  appear in the ASR output? Plain WER weighs "um" and "amoxicillin" the same,
  but for documentation the clinical terms matter far more.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import jiwer

FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "mhm", "uhm", "ah", "eh", "oh"}

DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "data" / "medical_terms.txt"


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and filler words, collapse whitespace.

    Both reference and hypothesis go through the same normalization so that
    formatting differences (capitalization, commas) are not counted as errors.
    """
    text = text.lower()
    text = re.sub(r"[‘’`]", "'", text)
    text = re.sub(r"(\d)%", r"\1 percent", text)
    text = re.sub(r"[^a-z0-9'\s-]", " ", text)   # drop punctuation
    text = re.sub(r"(?<![a-z0-9])['-]|['-](?![a-z0-9])", " ", text)  # stray ' and -
    text = text.replace("-", " ")
    words = [w for w in text.split() if w not in FILLERS]
    return " ".join(words)


def wer_details(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize(reference), normalize(hypothesis)
    out = jiwer.process_words(ref, hyp)
    n = out.hits + out.substitutions + out.deletions
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "ref_words": n,
    }


# --------------------------------------------------------------------------- #
# Medical-term recall
# --------------------------------------------------------------------------- #
def load_lexicon(path: str | Path = DEFAULT_LEXICON) -> list[str]:
    terms = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            terms.append(normalize(line))
    # Longest first, so "chest pain" is matched before "pain".
    return sorted(set(t for t in terms if t), key=lambda t: (-len(t.split()), t))


def count_terms(text: str, lexicon: list[str]) -> Counter:
    """Count lexicon terms in normalized text. Matched spans are consumed so a
    multi-word term is not double-counted by its sub-terms."""
    padded = f" {normalize(text)} "
    counts: Counter = Counter()
    for term in lexicon:
        pattern = re.compile(rf"(?<= ){re.escape(term)}(?= )")
        padded, n = pattern.subn("|", padded)
        if n:
            counts[term] += n
    return counts


def term_recall(reference: str, hypothesis: str, lexicon: list[str]) -> dict:
    ref_counts = count_terms(reference, lexicon)
    hyp_counts = count_terms(hypothesis, lexicon)
    total = sum(ref_counts.values())
    hit = sum(min(c, hyp_counts[t]) for t, c in ref_counts.items())
    missed = {t: c - min(c, hyp_counts[t]) for t, c in ref_counts.items() if c > hyp_counts[t]}
    return {
        "term_recall": hit / total if total else float("nan"),
        "terms_in_ref": total,
        "terms_hit": hit,
        "missed_terms": missed,
    }


def term_bias_prompt(lexicon: list[str], max_terms: int = 50) -> str:
    """Build a Whisper prompt from vocabulary.

    Whisper's prompt is capped at ~224 tokens, so only a limited number of terms
    fit. We prefer long single words (e.g. "amoxicillin", "gastroenteritis"):
    they are rare in general speech, so the model is least likely to know their
    spelling, and they gain the most from biasing.
    """
    singles = sorted((t for t in lexicon if " " not in t), key=lambda t: (-len(t), t))
    chosen = sorted(singles[:max_terms])
    return "Medical consultation between a doctor and a patient. Terms: " + ", ".join(chosen) + "."
