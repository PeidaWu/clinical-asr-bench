"""Generate structured consultation notes with an LLM and score them.

For each consultation we generate a note from:
  * the human reference transcript ("oracle"): an upper bound with perfect ASR
  * each ASR transcript produced by run_benchmark.py
Comparing the two isolates how much ASR errors hurt the final documentation.

Example:
    python scripts/run_notes.py --n 10 \
        --asr-tags whisper-large-v3-turbo__clean__longform whisper-large-v3-turbo__babble_snr5__longform
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinasr.data import load_dataset  # noqa: E402
from clinasr.structure import NoteGenerator, highlight_recall, note_to_text, rouge_l  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--asr-tags", nargs="*", default=[],
                   help="transcript tags from results/transcripts, e.g. whisper-small__clean__longform")
    p.add_argument("--backend", choices=["hf", "anthropic"], default="hf")
    p.add_argument("--model", default=None, help="override the LLM model id")
    p.add_argument("--data-dir", default="data/primock57")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    out = Path(args.out)
    (out / "notes").mkdir(parents=True, exist_ok=True)
    consultations = load_dataset(args.n, args.data_dir)
    gen = NoteGenerator(args.backend, args.model)

    rows = []
    for c in consultations:
        sources = {"oracle": "\n".join(f"{u.speaker}: {u.text}" for u in c.utterances)}
        for tag in args.asr_tags:
            f = out / "transcripts" / f"{c.cid}__{tag}.txt"
            if f.exists():
                sources[tag] = f.read_text(encoding="utf-8")
            else:
                print(f"[notes] missing {f}, run run_benchmark.py first")
        for src, transcript in sources.items():
            note = gen.generate(transcript)
            (out / "notes" / f"{c.cid}__{src}.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
            text = note_to_text(note)
            rows.append({
                "cid": c.cid, "source": src, "llm": gen.model_id,
                "rougeL": rouge_l(text, c.note),
                "highlight_recall": highlight_recall(text, c.highlights),
            })
            print(f"[notes] {c.cid} {src}: ROUGE-L={rows[-1]['rougeL']:.3f} "
                  f"highlights={rows[-1]['highlight_recall']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "notes_results.csv", index=False)
    summary = df.groupby(["llm", "source"])[["rougeL", "highlight_recall"]].mean().reset_index()
    (out / "notes_summary.md").write_text(summary.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
