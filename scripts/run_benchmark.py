"""Run the ASR benchmark on PriMock57.

Example (Colab GPU, ~10 consultations):
    python scripts/run_benchmark.py --n 10 \
        --models openai/whisper-small openai/whisper-large-v3-turbo \
        --snrs clean 20 10 5 --bias

Outputs (in --out, default results/):
    asr_results.csv      one row per (consultation, model, condition, decoding)
    missed_terms.csv     which medical terms each config failed to transcribe
    transcripts/*.txt    every hypothesis transcript (also used as a cache)
    summary.md           averaged table, ready to paste into the README
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinasr.augment import add_noise_at_snr, make_babble  # noqa: E402
from clinasr.data import load_dataset  # noqa: E402
from clinasr.metrics import load_lexicon, term_bias_prompt, term_recall, wer_details  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10, help="number of consultations")
    p.add_argument("--models", nargs="+", default=["openai/whisper-small"])
    p.add_argument("--snrs", nargs="+", default=["clean"],
                   help='"clean" and/or babble-noise SNRs in dB, e.g. clean 10 5 0')
    p.add_argument("--bias", action="store_true",
                   help="also run chunked decoding without and with a medical-vocabulary "
                        "prompt (the chunked run without the prompt is the fair baseline)")
    p.add_argument("--data-dir", default="data/primock57")
    p.add_argument("--out", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    (out / "transcripts").mkdir(parents=True, exist_ok=True)

    consultations = load_dataset(args.n, args.data_dir)
    if not consultations:
        sys.exit("no consultations loaded")
    if len(consultations) < 2 and any(s != "clean" for s in args.snrs):
        sys.exit("babble noise is built from *other* consultations: use --n 2 or more")
    lexicon = load_lexicon()
    prompt = term_bias_prompt(lexicon)
    print(f"[bench] {len(consultations)} consultations, "
          f"{sum(c.duration_s for c in consultations) / 60:.0f} min of audio")

    # Precompute the noisy versions once so every model hears identical audio.
    conditions: dict[str, dict[str, object]] = {}
    for snr in args.snrs:
        cond = "clean" if snr == "clean" else f"babble_snr{snr}"
        conditions[cond] = {}
        for i, c in enumerate(consultations):
            if snr == "clean":
                conditions[cond][c.cid] = c.audio
            else:
                others = [o.audio for o in consultations if o.cid != c.cid]
                babble = make_babble(others, len(c.audio), n_talkers=4, seed=i)
                conditions[cond][c.cid] = add_noise_at_snr(c.audio, babble, float(snr))

    from clinasr.asr import WhisperASR  # heavy import: torch/transformers

    rows, missed_rows = [], []
    # longform: Whisper's sequential long-form decoding (best default)
    # chunked / chunked_prompt: independent ~28 s chunks, without / with vocab prompt
    decodings = ["longform", "chunked", "chunked_prompt"] if args.bias else ["longform"]
    for model_id in args.models:
        asr = None
        for cond, audios in conditions.items():
            for decoding in decodings:
                tag = f"{model_id.split('/')[-1]}__{cond}__{decoding}"
                for c in consultations:
                    tfile = out / "transcripts" / f"{c.cid}__{tag}.txt"
                    if tfile.exists():  # cached: lets you resume after a Colab disconnect
                        hyp, secs, rtf = tfile.read_text(encoding="utf-8"), float("nan"), float("nan")
                    else:
                        if asr is None:
                            print(f"[bench] loading {model_id}")
                            asr = WhisperASR(model_id)
                        if decoding == "longform":
                            res = asr.transcribe(audios[c.cid])
                        else:
                            res = asr.transcribe_chunked(
                                audios[c.cid],
                                prompt=prompt if decoding == "chunked_prompt" else None)
                        hyp, secs, rtf = res["text"], res["seconds"], res["rtf"]
                        tfile.write_text(hyp, encoding="utf-8")
                    w = wer_details(c.reference_text, hyp)
                    t = term_recall(c.reference_text, hyp, lexicon)
                    rows.append({
                        "cid": c.cid, "model": model_id, "condition": cond, "decoding": decoding,
                        "wer": w["wer"], "sub": w["substitutions"], "del": w["deletions"],
                        "ins": w["insertions"], "ref_words": w["ref_words"],
                        "term_recall": t["term_recall"], "terms_in_ref": t["terms_in_ref"],
                        "terms_hit": t["terms_hit"], "seconds": secs, "rtf": rtf,
                    })
                    for term, n in t["missed_terms"].items():
                        missed_rows.append({"model": model_id, "condition": cond,
                                            "decoding": decoding, "term": term, "missed": n})
                    print(f"[bench] {tag} {c.cid}: WER={w['wer']:.3f} "
                          f"term_recall={t['term_recall']:.3f}")
        if asr is not None:
            asr.unload()

    df = pd.DataFrame(rows)
    df.to_csv(out / "asr_results.csv", index=False)
    pd.DataFrame(missed_rows).to_csv(out / "missed_terms.csv", index=False)

    # Corpus-level metrics: pool errors over all files (standard practice), rather
    # than averaging per-file WERs, which over-weights short files.
    g = df.groupby(["model", "condition", "decoding"])
    summary = pd.DataFrame({
        "WER": (g["sub"].sum() + g["del"].sum() + g["ins"].sum()) / g["ref_words"].sum(),
        "term_recall": g["terms_hit"].sum() / g["terms_in_ref"].sum(),
        "RTF": g["rtf"].mean(),
        "files": g.size(),
    }).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(summary.to_markdown(index=False, floatfmt=".3f"), encoding="utf-8")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
