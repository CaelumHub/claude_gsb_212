"""
realtime.py — Per-frame real-time analysis.

A browser captures microphone (or streams a file) and POSTs small PCM buffers;
the server analyses each buffer and returns instantaneous metrics.  All of the
work is self-implemented in :mod:`backend.dsp` and runs in a few milliseconds
per frame, comfortably inside a real-time budget.

``RealtimeAnalyzer`` keeps lightweight state (previous magnitude spectrum and
previous RMS) so that *onset strength* — which is inherently a differential
quantity — can be reported as well.  Because that state is per *stream of
frames*, analyzers must never be shared across clients: :class:`SessionManager`
hands out one analyzer per analysis session, serialises frame processing
within a session, and reaps sessions that have gone idle.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional, Sequence

from . import dsp


class RealtimeAnalyzer:
    def __init__(self):
        self.prev_mag: Optional[List[float]] = None
        self.prev_rms = 0.0

    def process(self, samples: Sequence[float], sr: float) -> Dict:
        samples = list(samples)
        n = len(samples)

        rms_v = dsp.rms(samples)
        level_db = dsp.db(rms_v)
        zcr = dsp.zero_crossing_rate(samples)

        # Spectral features via one FFT.
        nfft = dsp.next_pow2(max(n, 4))
        spec = dsp.fft(samples)
        bins = nfft // 2 + 1
        mag = [abs(spec[k]) for k in range(bins)]
        freqs = dsp.rfft_freqs(nfft, sr)
        centroid = dsp.spectral_centroid(mag, freqs)
        flux = dsp.spectral_flux(mag, self.prev_mag)
        self.prev_mag = mag

        # Pitch (autocorrelation) when enough samples are present.
        pitch = None
        if n >= 128:
            pitch = dsp.pitch_autocorr(samples, sr, fmin=60.0, fmax=1200.0)

        # Energy-based onset (normalised rise in RMS).
        onset = 0.0
        if self.prev_rms > 1e-6:
            rise = (rms_v - self.prev_rms) / self.prev_rms
            onset = max(0.0, min(1.0, rise))
        self.prev_rms = rms_v

        return {
            "frames": n,
            "sr": sr,
            "rms": round(rms_v, 6),
            "level_db": round(level_db, 2),
            "zcr": round(zcr, 4),
            "centroid_hz": round(centroid, 1),
            "flux": round(flux, 4),
            "onset": round(onset, 4),
            "pitch_hz": round(pitch, 2) if pitch else None,
            "note": dsp.note_display(pitch) if pitch else "--",
            "clipping": bool(any(abs(x) > 0.999 for x in samples)),
        }

    def reset(self) -> None:
        self.prev_mag = None
        self.prev_rms = 0.0


# A per-session analyser is instantiated by the Flask app; expose a factory.
def new_analyzer() -> RealtimeAnalyzer:
    return RealtimeAnalyzer()


class _Session:
    """One analyzer plus the lock that serialises its frame processing."""

    __slots__ = ("analyzer", "lock", "last_seen")

    def __init__(self) -> None:
        self.analyzer = RealtimeAnalyzer()
        self.lock = threading.Lock()
        self.last_seen = time.monotonic()


class SessionManager:
    """Thread-safe registry of per-session :class:`RealtimeAnalyzer` state.

    Each browser tab running real-time analysis creates its own session, so
    differential metrics (flux, onset) of one session can never leak into
    another.  Frames of a single session are processed under that session's
    lock, which keeps concurrent in-flight requests from interleaving state
    updates.  Idle sessions are swept after ``ttl`` seconds and the number of
    live sessions is capped at ``max_sessions`` (oldest evicted first).
    """

    def __init__(self, ttl: float = 900.0, max_sessions: int = 64):
        self._ttl = ttl
        self._max_sessions = max_sessions
        self._sessions: Dict[str, _Session] = {}
        self._guard = threading.Lock()

    def _sweep(self, now: float) -> None:
        expired = [sid for sid, s in self._sessions.items()
                   if now - s.last_seen > self._ttl]
        for sid in expired:
            del self._sessions[sid]

    def create(self) -> str:
        with self._guard:
            now = time.monotonic()
            self._sweep(now)
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions,
                             key=lambda sid: self._sessions[sid].last_seen)
                del self._sessions[oldest]
            sid = uuid.uuid4().hex
            self._sessions[sid] = _Session()
            return sid

    def process(self, sid: str, samples: Sequence[float], sr: float) -> Dict:
        """Analyse one frame under the session's lock.

        Raises :class:`KeyError` if the session is unknown or has expired.
        """
        with self._guard:
            sess = self._sessions.get(sid)
            now = time.monotonic()
            if sess is None or now - sess.last_seen > self._ttl:
                self._sessions.pop(sid, None)
                raise KeyError(sid)
            sess.last_seen = now
        with sess.lock:
            return sess.analyzer.process(samples, sr)

    def close(self, sid: str) -> bool:
        with self._guard:
            return self._sessions.pop(sid, None) is not None
