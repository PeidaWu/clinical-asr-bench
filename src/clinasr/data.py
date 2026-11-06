"""Download and prepare the PriMock57 dataset.

PriMock57 (Korfiatis et al., 2022; CC BY 4.0) contains 57 mock primary-care
consultations. Each consultation has:
  * two audio files (doctor mic, patient mic), stored with Git LFS
  * two Praat TextGrid transcripts (one per speaker), with timestamps
  * a JSON file with the clinician-written note and key "highlights"

We download only the files we need over HTTPS (no full git clone), merge the two
speaker channels into one "room microphone" signal, and build a reference
transcript by interleaving the two speakers' utterances by start time.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAW_BASE = "https://raw.githubusercontent.com/babylonhealth/primock57/main"
# .wav files are stored with Git LFS, so they are served from the media host.
LFS_BASE = "https://media.githubusercontent.com/media/babylonhealth/primock57/main"

SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono audio

# Consultation IDs: day1..day5, each with up to 12 consultations (57 total).
# We enumerate candidates and skip any that fail to download.
CANDIDATE_IDS = [f"day{d}_consultation{c:02d}" for d in range(1, 6) for c in range(1, 13)]


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class Consultation:
    cid: str
    audio: np.ndarray            # mixed "room mic" audio, float32, 16 kHz
    utterances: list[Utterance]  # reference utterances, sorted by start time
    note: str                    # clinician-written consultation note
    highlights: list[str]        # key findings the clinician highlighted

    @property
    def reference_text(self) -> str:
        return " ".join(u.text for u in self.utterances if u.text)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def download_consultation(cid: str, data_dir: Path) -> None:
    """Download audio, transcripts and note for one consultation."""
    for spk in ("doctor", "patient"):
        _download(f"{LFS_BASE}/audio/{cid}_{spk}.wav", data_dir / "audio" / f"{cid}_{spk}.wav")
        _download(
            f"{RAW_BASE}/transcripts/{cid}_{spk}.TextGrid",
            data_dir / "transcripts" / f"{cid}_{spk}.TextGrid",
        )
    _download(f"{RAW_BASE}/notes/{cid}.json", data_dir / "notes" / f"{cid}.json")


# --------------------------------------------------------------------------- #
# TextGrid parsing
# --------------------------------------------------------------------------- #
_INTERVAL_RE = re.compile(
    r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"(.*?)"\s*\n',
    re.DOTALL,
)
# Annotation tags used by PriMock57, e.g. <UNIN/> (unintelligible),
# <UNSURE>..</UNSURE>, <INAUDIBLE/>.  We drop self-closing tags and keep the
# words inside paired tags.
_SELF_CLOSING_TAG_RE = re.compile(r"<[A-Z_]+/>")
_PAIRED_TAG_RE = re.compile(r"</?[A-Z_]+>")


def clean_annotation(text: str) -> str:
    text = text.replace('""', '"')  # TextGrid escapes quotes by doubling them
    text = _SELF_CLOSING_TAG_RE.sub(" ", text)
    text = _PAIRED_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_textgrid(path: Path, speaker: str) -> list[Utterance]:
    content = path.read_text(encoding="utf-8", errors="replace")
    # Skip the file/tier headers: intervals start after "intervals: size".
    body = content.split("intervals: size", 1)[-1]
    utts = []
    for xmin, xmax, raw in _INTERVAL_RE.findall(body):
        text = clean_annotation(raw)
        if text:
            utts.append(Utterance(float(xmin), float(xmax), speaker, text))
    return utts


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def load_audio(path: Path) -> np.ndarray:
    """Load a file as 16 kHz mono float32."""
    import librosa

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def mix_channels(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sum two close-talk channels into one simulated room-microphone signal."""
    n = max(len(a), len(b))
    mixed = np.zeros(n, dtype=np.float32)
    mixed[: len(a)] += a
    mixed[: len(b)] += b
    peak = np.abs(mixed).max()
    return mixed / peak * 0.95 if peak > 0 else mixed


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_consultation(cid: str, data_dir: Path) -> Consultation:
    download_consultation(cid, data_dir)
    doc = load_audio(data_dir / "audio" / f"{cid}_doctor.wav")
    pat = load_audio(data_dir / "audio" / f"{cid}_patient.wav")
    utts = parse_textgrid(data_dir / "transcripts" / f"{cid}_doctor.TextGrid", "Doctor")
    utts += parse_textgrid(data_dir / "transcripts" / f"{cid}_patient.TextGrid", "Patient")
    utts.sort(key=lambda u: u.start)
    note = json.loads((data_dir / "notes" / f"{cid}.json").read_text(encoding="utf-8"))
    return Consultation(
        cid=cid,
        audio=mix_channels(doc, pat),
        utterances=utts,
        note=note.get("note", ""),
        highlights=note.get("highlights", []),
    )


def load_dataset(n: int, data_dir: str | Path = "data/primock57") -> list[Consultation]:
    """Load the first ``n`` consultations that download successfully."""
    data_dir = Path(data_dir)
    out: list[Consultation] = []
    for cid in CANDIDATE_IDS:
        if len(out) >= n:
            break
        try:
            out.append(load_consultation(cid, data_dir))
            print(f"[data] loaded {cid} ({out[-1].duration_s / 60:.1f} min)")
        except Exception as exc:  # missing consultation IDs are expected
            print(f"[data] skip {cid}: {exc}")
    return out
