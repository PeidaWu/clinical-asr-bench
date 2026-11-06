# Clinical ASR Bench

**How well does open-weight speech recognition handle clinical conversations, and how much do its errors hurt automatically generated documentation?**

AI scribes work well in quiet one-on-one clinic visits. Settings such as emergency departments and trauma bays are harder: several people talk at once, a single room microphone picks everything up, and the vocabulary is dense with drug names and diagnoses. This project builds a small, reproducible benchmark to measure these effects with open-weight models that can run entirely on local (PHI-safe) hardware.

## Pipeline

```
PriMock57 audio (doctor + patient mics)
        │  mix into one "room microphone" channel
        │  optionally add multi-talker babble at a set SNR
        ▼
Whisper (Hugging Face Transformers, long-form decoding)
        │  optional: medical-vocabulary prompt (contextual biasing)
        ▼
ASR transcript ──► WER + medical-term recall vs. human transcript
        │
        ▼
Open-weight LLM (Qwen2.5-Instruct) ──► structured JSON note
        │
        ▼
ROUGE-L + key-finding recall vs. clinician-written note
```

## Data

[PriMock57](https://github.com/babylonhealth/primock57) (Korfiatis et al., 2022, CC BY 4.0): 57 mock primary-care consultations between clinicians and actor patients, with separate doctor and patient audio, time-aligned human transcripts, and the clinician's consultation note. All data is synthetic (no real patients). The scripts download only the files they need.

## Experiments

| # | Question | How |
|---|---|---|
| 1 | Baseline accuracy | `whisper-small` vs `whisper-large-v3-turbo` on clean mixed audio |
| 2 | Robustness to crowded rooms | Add babble built from *other* consultations at 20 / 10 / 5 dB SNR |
| 3 | Medical vocabulary | Prompt every ~28 s chunk with a medical term list; compare against the same chunking without the prompt |
| 4 | Error propagation | Generate notes from human vs ASR transcripts and compare note quality |

### Metrics
- **WER**, corpus-level (errors pooled over all files) after text normalization (lowercase, no punctuation, no fillers like "um").
- **Medical-term recall**: fraction of medical-term occurrences in the reference that also appear in the ASR output, using the lexicon in [`data/medical_terms.txt`](data/medical_terms.txt). WER treats "um" and "amoxicillin" equally; this metric doesn't.
- **RTF** (real-time factor): processing time ÷ audio length.
- **Note quality**: ROUGE-L F1 against the clinician note, and recall of the clinician's highlighted key findings.

## Results

> 🚧 Run [`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) and paste `results/summary.md`, `results/notes_summary.md` and `results/wer_vs_snr.png` here.

**ASR** (10 consultations)

| model | condition | decoding | WER | medical-term recall | RTF |
|---|---|---|---|---|---|
| _to be filled_ | | | | | |

**Notes**

| transcript source | ROUGE-L | key-finding recall |
|---|---|---|
| _to be filled_ | | |

**Observations:** _(write 3–5 bullets: which terms are missed most, how fast WER degrades with noise, whether prompting helps, how much ASR errors reduce note quality)_

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest -q tests

# ASR benchmark (GPU recommended)
python scripts/run_benchmark.py --n 10 \
    --models openai/whisper-small openai/whisper-large-v3-turbo \
    --snrs clean 20 10 5 --bias
python scripts/plot_results.py

# Structured notes from the human transcript and from ASR transcripts
python scripts/run_notes.py --n 10 \
    --asr-tags whisper-large-v3-turbo__clean__longform whisper-large-v3-turbo__babble_snr5__longform
```

Or open the notebook in Google Colab with a free T4 GPU.

## Repository layout

```
src/clinasr/
  data.py       download PriMock57, parse TextGrid transcripts, mix channels
  augment.py    babble / white noise at a target SNR
  asr.py        Whisper long-form decoding; silence-aware chunked decoding with optional vocabulary prompt
  metrics.py    normalization, WER, medical-term recall
  structure.py  LLM transcript → JSON note, ROUGE-L, key-finding recall
scripts/        run_benchmark.py, run_notes.py, plot_results.py
data/           medical_terms.txt
notebooks/      colab_quickstart.ipynb
tests/          unit tests for metrics, parsing and augmentation
```

## Limitations and next steps

- PriMock57 has 2 speakers in a quiet room; a real trauma bay has 10+ speakers, alarms and reverberation. Babble noise is only an approximation.
- Re-applying a prompt inside Whisper's long-form decoding (`condition_on_prev_tokens=True`) caused repetition loops that dropped most of the transcript, so prompting uses independent chunks instead.
- The lexicon is hand-built; a better version would extract terms with scispaCy/UMLS.
- Next: speaker diarization (pyannote / WhisperX) so the note can attribute findings to speakers, fine-tuning Whisper on medical speech (LoRA), and a structured template that matches the trauma primary survey (ABCDE).

## Citation

Korfiatis, A. P., Moramarco, F., Sarac, R., & Savkov, A. (2022). *PriMock57: A Dataset Of Primary Care Mock Consultations.* ACL 2022.
