"""
realtime.py — Per-frame real-time analysis.

A browser captures microphone (or streams a file) and POSTs small PCM buffers;
the server analyses each buffer and returns instantaneous metrics.  All of the
work is self-implemented in :mod:`backend.dsp` and runs in a few milliseconds
per frame, comfortably inside a real-time budget.

``RealtimeAnalyzer`` keeps lightweight state (previous magnitude spectrum and
previous RMS) so that *onset strength* — which is inherently a differential
quantity — can be reported as well.

State isolation
---------------
Onset strength and spectral flux are differential quantities, so an analyser
is stateful.  Earlier versions kept a single process-wide analyser; with more
than one browser tab (or a stop/restart cycle) the sessions overwrote each
other's ``prev_mag`` / ``prev_rms``, and frames arriving out of order mixed
state between adjacent frames.  :class:`RealtimeSessionManager` now gives every
"start real-time analysis" action its own :class:`RealtimeAnalyzer`, a lock
that serialises frames of that session, and a frame sequence guard that
rejects stale (late) frames.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from . import dsp

# Sessions idle for longer than this are reaped automatically.
SESSION_TTL_SECONDS = 120.0


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
        flux = dsp.spectral_flux(mag, self._aligned_prev_mag(bins))
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

    def _aligned_prev_mag(self, bins: int) -> Optional[List[float]]:
        """Previous frame's magnitude spectrum, safe to diff against ``bins``.

        Frame sizes are normally fixed, but a defensive resize prevents an
        index error (and bogus flux) if two consecutive frames ever differ.
        """
        prev = self.prev_mag
        if prev is None:
            return None
        if len(prev) == bins:
            return prev
        if len(prev) < bins:
            return prev + [0.0] * (bins - len(prev))
        return prev[:bins]

    def reset(self) -> None:
        self.prev_mag = None
        self.prev_rms = 0.0


# Backwards-compatible factory.
def new_analyzer() -> RealtimeAnalyzer:
    return RealtimeAnalyzer()


class _Session:
    __slots__ = ("analyzer", "lock", "last_seq", "last_used")

    def __init__(self):
        self.analyzer = RealtimeAnalyzer()
        # Serialises frame processing for this session so differential state
        # always advances in the client's frame order, never HTTP-arrival order.
        self.lock = threading.Lock()
        self.last_seq = -1
        self.last_used = time.monotonic()


class RealtimeSessionManager:
    """Owns one :class:`RealtimeAnalyzer` per real-time analysis session."""

    def __init__(self, ttl: float = SESSION_TTL_SECONDS):
        self._sessions: Dict[str, _Session] = {}
        self._guard = threading.Lock()
        self._ttl = ttl

    def start(self) -> str:
        self._reap_if_due()
        sid = uuid.uuid4().hex
        with self._guard:
            self._sessions[sid] = _Session()
        return sid

    def stop(self, session_id: str) -> bool:
        with self._guard:
            return self._sessions.pop(session_id, None) is not None

    def process(self, session_id: str, samples: Sequence[float], sr: float,
                seq: int = 0) -> Tuple[Optional[Dict], bool]:
        """Process one frame for ``session_id``.

        Returns ``(result, stale)``: ``result`` is ``None`` for an unknown
        session, and ``stale`` is True when a late, out-of-order frame was
        dropped (its state must not contaminate newer frames).
        """
        self._reap_if_due()
        with self._guard:
            session = self._sessions.get(session_id)
        if session is None:
            return None, False

        with session.lock:
            if seq <= session.last_seq:
                session.last_used = time.monotonic()
                return None, True
            result = session.analyzer.process(samples, sr)
            session.last_seq = seq
            session.last_used = time.monotonic()
        result["seq"] = seq
        return result, False

    # -- housekeeping ------------------------------------------------------ #

    def _reap_if_due(self) -> None:
        now = time.monotonic()
        with self._guard:
            dead = [sid for sid, s in self._sessions.items()
                    if now - s.last_used > self._ttl]
            for sid in dead:
                self._sessions.pop(sid, None)
