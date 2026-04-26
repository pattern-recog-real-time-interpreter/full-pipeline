from __future__ import annotations

import collections
from typing import Optional

import numpy as np
import torch

from .config import PipelineConfig


class VADSegmenter:
    """Silero VAD wrapper.

    Batch mode: segment_file() splits a full audio array into speech segments.
    Streaming mode: push_chunk() accumulates mic chunks and returns a segment
    when silence or max duration is detected.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
        self._get_speech_timestamps = None
        self._collect_chunks = None
        self._load_silero()

        # Streaming state
        self._buffer: list[np.ndarray] = []
        self._buffer_samples: int = 0
        self._silence_samples: int = 0
        self._min_silence_samples = int(config.vad_min_silence_ms * config.sample_rate / 1000)
        self._max_samples = int(config.max_segment_seconds * config.sample_rate)
        self._rms_threshold = config.rms_threshold

    def _load_silero(self) -> None:
        model, utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        (get_speech_timestamps, _, _, _, collect_chunks) = utils
        self._model = model
        self._get_speech_timestamps = get_speech_timestamps
        self._collect_chunks = collect_chunks

    def _to_tensor(self, audio: np.ndarray) -> torch.Tensor:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return torch.from_numpy(audio)

    def segment_file(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split audio array into speech segments using Silero VAD."""
        wav = self._to_tensor(audio)
        timestamps = self._get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=self.config.sample_rate,
            min_silence_duration_ms=self.config.vad_min_silence_ms,
            min_speech_duration_ms=self.config.vad_min_speech_ms,
        )
        if not timestamps:
            return []
        segments = [
            self._collect_chunks([ts], wav).numpy()
            for ts in timestamps
        ]
        return segments

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def push_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Add a mic chunk; returns a trimmed speech segment when ready."""
        self._buffer.append(chunk)
        self._buffer_samples += len(chunk)

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < self._rms_threshold:
            self._silence_samples += len(chunk)
        else:
            self._silence_samples = 0

        silence_reached = self._silence_samples >= self._min_silence_samples
        max_reached = self._buffer_samples >= self._max_samples

        if silence_reached or max_reached:
            return self._flush()
        return None

    def flush(self) -> Optional[np.ndarray]:
        """Force-return whatever is buffered (e.g., on shutdown)."""
        if self._buffer_samples == 0:
            return None
        return self._flush()

    def _flush(self) -> Optional[np.ndarray]:
        audio = np.concatenate(self._buffer)
        self._buffer.clear()
        self._buffer_samples = 0
        self._silence_samples = 0

        # Layer 2: Silero trim within the accumulated segment
        segments = self.segment_file(audio)
        if not segments:
            return None
        return np.concatenate(segments)
