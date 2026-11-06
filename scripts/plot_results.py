"""Plot WER and medical-term recall versus noise level from results/summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def snr_value(condition: str) -> float:
    # "clean" is plotted at the far right as the best case
    return 30.0 if condition == "clean" else float(condition.replace("babble_snr", ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    args = p.parse_args()
    res = Path(args.results)
    df = pd.read_csv(res / "summary.csv")
    df["snr"] = df["condition"].map(snr_value)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    styles = {"longform": "-", "chunked": ":", "chunked_prompt": "--"}
    for (model, decoding), g in df.groupby(["model", "decoding"]):
        g = g.sort_values("snr")
        label = f"{model.split('/')[-1]} ({decoding})"
        style = styles.get(decoding, "-")
        axes[0].plot(g["snr"], g["WER"], style, marker="o", label=label)
        axes[1].plot(g["snr"], g["term_recall"], style, marker="o", label=label)

    ticks = sorted(df["snr"].unique())
    for ax, ylabel in zip(axes, ["Word error rate (lower is better)",
                                 "Medical-term recall (higher is better)"]):
        ax.set_xlabel("Babble-noise SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ticks, ["clean" if t == 30 else f"{t:g}" for t in ticks])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Whisper on PriMock57 with simulated multi-talker noise")
    fig.tight_layout()
    fig.savefig(res / "wer_vs_snr.png", dpi=150)
    print(f"saved {res / 'wer_vs_snr.png'}")


if __name__ == "__main__":
    main()
