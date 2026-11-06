"""Speech recognition with open-weight Whisper models via Hugging Face Transformers."""

from __future__ import annotations

import time

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .data import SAMPLE_RATE


def split_at_silence(audio: np.ndarray, max_s: float = 28.0, search_s: float = 4.0,
                     frame_s: float = 0.05) -> list[tuple[int, int]]:
    """Chunk boundaries (sample indices) of at most ``max_s`` seconds each.

    Each cut is placed at the quietest 50 ms frame within the last ``search_s``
    seconds of the window, so we rarely cut through the middle of a word.
    """
    sr, n = SAMPLE_RATE, len(audio)
    max_len, search, frame = int(max_s * sr), int(search_s * sr), int(frame_s * sr)
    bounds, start = [], 0
    while n - start > max_len:
        lo, hi = start + max_len - search, start + max_len
        frames = audio[lo:hi][: (hi - lo) // frame * frame].reshape(-1, frame)
        quietest = int(np.argmin((frames.astype(np.float64) ** 2).mean(axis=1)))
        cut = lo + quietest * frame + frame // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, n))
    return bounds


class WhisperASR:
    """Thin wrapper around a Hugging Face Whisper checkpoint.

    Uses Whisper's *sequential long-form* decoding (30 s windows, each window
    starting where the previous transcription ended), which is more accurate on
    long recordings than naive fixed-size chunking.
    """

    def __init__(self, model_id: str = "openai/whisper-small", device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray) -> dict:
        """Transcribe 16 kHz mono audio with Whisper's sequential long-form decoding.

        For vocabulary prompting see ``transcribe_chunked``.

        Returns:
            {"text": str, "seconds": float, "rtf": float}
            rtf = real-time factor = processing time / audio duration.
        """
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,          # keep the whole recording (> 30 s)
            padding="longest",
            return_attention_mask=True,
        )
        if inputs.input_features.shape[-1] < 3000:  # short clip: pad to 30 s
            inputs = self.processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                return_attention_mask=True,
            )
        inputs = inputs.to(self.device, self.dtype)

        gen_kwargs = dict(
            language="en",
            task="transcribe",
            return_timestamps=True,          # required for long-form decoding
            condition_on_prev_tokens=False,  # reduces hallucination loops
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),  # fallback if decoding fails
        )
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        sequences = out["sequences"] if isinstance(out, dict) else out
        text = self.processor.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        duration = len(audio) / SAMPLE_RATE
        return {"text": text, "seconds": elapsed, "rtf": elapsed / max(duration, 1e-6)}

    @torch.inference_mode()
    def transcribe_chunked(self, audio: np.ndarray, prompt: str | None = None) -> dict:
        """Split audio into <=30 s chunks at quiet points, transcribe each chunk
        independently, and optionally give *every* chunk the same vocabulary prompt.

        Why not use the prompt with long-form decoding? Re-applying a prompt to every
        window there requires ``condition_on_prev_tokens=True``, which in our tests
        made Whisper fall into repetition loops and drop most of the transcript.
        Independent chunks avoid that failure mode.
        """
        prompt_ids = None
        if prompt:
            prompt_ids = torch.tensor(self.processor.get_prompt_ids(prompt), device=self.device)
        texts = []
        t0 = time.perf_counter()
        for start, end in split_at_silence(audio):
            feats = self.processor(
                audio[start:end], sampling_rate=SAMPLE_RATE, return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device, self.dtype)
            gen_kwargs = dict(language="en", task="transcribe", max_new_tokens=220)
            if prompt_ids is not None:
                gen_kwargs["prompt_ids"] = prompt_ids
            seq = self.model.generate(**feats, **gen_kwargs)
            t = self.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
            if prompt and t.startswith(prompt.strip()):  # some versions echo the prompt
                t = t[len(prompt.strip()):].strip()
            texts.append(t)
        elapsed = time.perf_counter() - t0
        duration = len(audio) / SAMPLE_RATE
        return {"text": " ".join(x for x in texts if x), "seconds": elapsed,
                "rtf": elapsed / max(duration, 1e-6)}

    def unload(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
