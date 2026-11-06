"""Turn a consultation transcript into structured clinical documentation with an LLM.

Two backends:
  * "hf"        an open-weight instruction model run locally with Transformers
                (default Qwen/Qwen2.5-1.5B-Instruct, which fits a free Colab T4).
                Open weights matter for clinical data: the audio and text never
                leave the machine, which is how PHI must be handled.
  * "anthropic" a hosted model through the Anthropic API (set ANTHROPIC_API_KEY).
                Only for *public* data such as PriMock57, never for real patients.

Evaluation compares the generated note with the clinician-written note:
  * ROUGE-L F1: overall wording overlap
  * highlight recall: share of the clinician's key findings ("highlights")
    that appear in the generated note
"""

from __future__ import annotations

import json
import re

from .metrics import normalize

NOTE_FIELDS = [
    "presenting_complaint",
    "history_of_presenting_complaint",
    "past_medical_history",
    "medications",
    "allergies",
    "social_history",
    "examination_findings",
    "impression",
    "plan",
]

SYSTEM_PROMPT = (
    "You are a clinical scribe. Read the transcript of a doctor-patient consultation "
    "and write the consultation note. Use only information stated in the transcript; "
    "if something was not discussed, write \"not discussed\". The transcript comes from "
    "automatic speech recognition and may contain errors.\n"
    "Return ONLY a JSON object with these string fields: " + ", ".join(NOTE_FIELDS) + "."
)


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {k: str(data.get(k, "")) for k in NOTE_FIELDS}
        except json.JSONDecodeError:
            pass
    return {k: "" for k in NOTE_FIELDS} | {"history_of_presenting_complaint": text}


class NoteGenerator:
    def __init__(self, backend: str = "hf", model_id: str | None = None):
        self.backend = backend
        if backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model_id = model_id or "Qwen/Qwen2.5-1.5B-Instruct"
            self.tok = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
        elif backend == "anthropic":
            import anthropic

            self.model_id = model_id or "claude-haiku-5-5"
            self.client = anthropic.Anthropic()
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def generate(self, transcript: str) -> dict:
        user = f"Transcript:\n{transcript}\n\nWrite the JSON note."
        if self.backend == "hf":
            import torch

            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user}]
            enc = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self.model.device)
            ids = enc["input_ids"]
            with torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=700, do_sample=False)
            text = self.tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
        else:
            resp = self.client.messages.create(
                model=self.model_id,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_json(text)


def note_to_text(note: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in note.items() if v)


def highlight_recall(generated: str, highlights: list[str], min_overlap: float = 0.5) -> float:
    """A highlight counts as covered if at least ``min_overlap`` of its words
    appear in the generated note (a lenient match, since clinicians abbreviate)."""
    if not highlights:
        return float("nan")
    gen_words = set(normalize(generated).split())
    covered = 0
    for h in highlights:
        words = normalize(h).split()
        if words and sum(w in gen_words for w in words) / len(words) >= min_overlap:
            covered += 1
    return covered / len(highlights)


def rouge_l(generated: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, generated)["rougeL"].fmeasure
