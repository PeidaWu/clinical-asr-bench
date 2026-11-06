import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinasr.augment import add_noise_at_snr, make_babble  # noqa: E402
from clinasr.data import clean_annotation, parse_textgrid  # noqa: E402
from clinasr.metrics import (  # noqa: E402
    count_terms, load_lexicon, normalize, term_bias_prompt, term_recall, wer_details,
)
from clinasr.structure import _parse_json, highlight_recall  # noqa: E402


def test_normalize_drops_fillers_and_punctuation():
    assert normalize("Um, I've had DIARRHEA... for 3 days!") == "i've had diarrhea for 3 days"


def test_wer_identical_and_one_substitution():
    assert wer_details("take paracetamol twice", "Take paracetamol, twice.")["wer"] == 0
    d = wer_details("take paracetamol twice", "take paracetamo twice")
    assert d["substitutions"] == 1 and abs(d["wer"] - 1 / 3) < 1e-9


def test_term_counting_prefers_longest_match():
    lex = ["chest pain", "pain", "asthma"]
    lex = sorted(lex, key=lambda t: (-len(t.split()), t))
    c = count_terms("chest pain and some pain, pain. Asthma", lex)
    assert c["chest pain"] == 1 and c["pain"] == 2 and c["asthma"] == 1


def test_term_recall():
    lex = load_lexicon()
    r = term_recall("I take amoxicillin for my asthma", "I take a moxy cillin for my asthma", lex)
    assert r["terms_in_ref"] == 2 and r["terms_hit"] == 1
    assert r["missed_terms"] == {"amoxicillin": 1}


def test_bias_prompt_is_short_enough():
    prompt = term_bias_prompt(load_lexicon())
    assert "amoxicillin" in prompt and len(prompt.split()) < 120


def test_snr_scaling():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(16000).astype(np.float32)
    noise = rng.standard_normal(8000).astype(np.float32)
    mixed = add_noise_at_snr(sig, noise, 0.0)
    assert mixed.shape == sig.shape and np.abs(mixed).max() <= 0.951


def test_babble_shape():
    others = [np.ones(1000, np.float32), np.ones(50, np.float32)]
    assert make_babble(others, 3000).shape == (3000,)


def test_textgrid_parsing(tmp_path):
    tg = tmp_path / "x.TextGrid"
    tg.write_text(
        'File type = "ooTextFile"\nitem []:\n    item [1]:\n        intervals: size = 2\n'
        '        intervals [1]:\n            xmin = 0 \n            xmax = 2.5 \n            text = "" \n'
        '        intervals [2]:\n            xmin = 2.5 \n            xmax = 4.0 \n'
        '            text = "<UNSURE>Hello how</UNSURE> um. Good <UNIN/> morning" \n'
    )
    utts = parse_textgrid(tg, "Doctor")
    assert len(utts) == 1 and utts[0].text == "Hello how um. Good morning" and utts[0].start == 2.5


def test_clean_annotation():
    assert clean_annotation('say ""hi"" <INAUDIBLE/> there') == 'say "hi" there'


def test_parse_json_and_highlights():
    note = _parse_json('Sure! {"impression": "gastroenteritis", "plan": "fluids"} done')
    assert note["impression"] == "gastroenteritis" and note["allergies"] == ""
    assert highlight_recall("impression: gastroenteritis", ["gastroenteritis", "LLQ pain"]) == 0.5
