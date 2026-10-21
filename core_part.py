#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morse Code Bitmap Burst Encoder - core DSP / codec module
==========================================================
Turns a grayscale bitmap into Morse code and exports it as a baseband IQ file
(WAV or RF64, 16-bit stereo, Left = I, Right = Q). The decoder imports IQ
files of any common bit depth (8/16/24/32-bit PCM, 32/64-bit float), or
listens to an audio input (e.g. an SDR program in raw/binaural IQ mode),
finds the header, shows the call sign and decodes the picture.

Transmission layout (everything is sent as Morse):

    CALL  CALL  CALL  RES<n> RES<n>  <SYNC>  <5 x symbol per pixel>  <SYNC>

<SYNC> is a short word made only of '=' (-...-). '=' can never occur in the
pixel data (A-Z / 0-9 only), so the markers can be short without any risk of
a false sync inside the picture.

Requires: Python 3.9+, numpy
"""
from __future__ import annotations

import math
import os
import re
import struct
from collections import Counter

import numpy as np

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
APP_NAME = "Morse Code Bitmap Burst Encoder"
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
RESOLUTIONS = (20, 40, 80, 120, 240)

MAX_CALL = 16                          # longest call sign the encoder accepts
FRAME_CHAR = "="                       # never produced by pixel data
MARKER_BREAKS = ((300, 6), (3000, 5), (float("inf"), 4))   # (max WPM, marker length)


def marker_len(wpm: float) -> int:
    for lim, length in MARKER_BREAKS:
        if wpm <= lim:
            return length
    return 4


def frame_str(wpm: float) -> str:
    return FRAME_CHAR * marker_len(wpm)


BASE_RATE = 48000
STD_RATES = (48000, 96000, 192000, 384000, 768000, 1536000)
MIN_WPM = 5.0
WAV_MAX_WPM = 45000.0
RF64_MAX_WPM = 150000.0
WAV_MAX_FRAMES = (0xFFFFFFFF - 36) // 4          # classic RIFF/WAV size limit
LEVEL = 0.9


def pad_seconds(wpm: float) -> float:
    """Silence before/after the burst: 0.25 s at normal speeds, much shorter when the burst itself is fast."""
    return min(0.25, max(0.04, 400.0 * 1.2 / wpm))


MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.",
    "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.",
    "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-",
    "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.", "=": "-...-", "-": "-....-",
    "+": ".-.-.", ":": "---...", "(": "-.--.", ")": "-.--.-", "'": ".----.", "!": "-.-.--",
    "@": ".--.-.", "&": ".-...",
}
PATTERN = {c: np.array([x for k, e in enumerate(code) for x in ((0 if k == 0 else 1), 1 if e == "." else 3)],
                       dtype=np.int64) for c, code in MORSE.items()}


def _code_key(code: str) -> int:
    return sum((1 if c == "." else 2) * 3 ** i for i, c in enumerate(code))


_KEYS = sorted((_code_key(code), ord(c)) for c, code in MORSE.items())
KEY_ARR = np.array([k for k, _ in _KEYS], dtype=np.int64)
KEY_CHR = np.array([c for _, c in _KEYS], dtype=np.uint8)


# --------------------------------------------------------------------------- #
#  Bandwidth estimate
# --------------------------------------------------------------------------- #
def min_bandwidth_hz(wpm: float, ramp_ms: float = 5.0, fs: int | None = None) -> dict:
    """Receive passband needed around the tone (total width, e.g. an SSB filter).

    The signal is one on/off-keyed tone, so there are no harmonics or subcarriers to pass.
    With dot length T (PARIS timing, T = 1.2 / WPM seconds):
      minimum  : 1 / T   - the Nyquist width for a keying rate of 1/T; dots just survive, edges get soft
      comfort  : 2 / T   - the whole main lobe of the keying spectrum; recommended
      clean    : 1 / rise time of the shaped key edges - also passes the sidelobes (no edge smearing)
    Rule-of-thumb figures, not a regulatory emission mask.
    """
    fs = fs or BASE_RATE
    unit_s = 1.2 / wpm
    rise_s = min(ramp_ms / 1000.0, unit_s / 2.0)
    if rise_s <= 0:
        rise_s = unit_s / 8.0
    minimum = 1.0 / unit_s
    comfortable = 2.0 / unit_s
    clean = max(comfortable, 1.0 / rise_s)
    return {"min_hz": minimum, "comfortable_hz": comfortable, "clean_hz": clean,
            "tight_hz": minimum, "unit_s": unit_s, "rise_s": rise_s}


# --------------------------------------------------------------------------- #
#  Text codec (image <-> text)
# --------------------------------------------------------------------------- #
def quantize_pixels(pixels: np.ndarray) -> np.ndarray:
    """What a perfect decode returns: the image reduced to the 36 symbol levels."""
    idx = np.clip(np.floor(pixels.astype(np.float64) / 255.0 * 35 + 0.5), 0, 35)
    return np.floor(idx / 35.0 * 255 + 0.5).astype(np.uint8)


def pixels_to_text(pixels: np.ndarray, n: int, call: str, wpm: float = 20.0) -> str:
    idx = np.clip(np.floor(pixels.reshape(-1).astype(np.float64) / 255.0 * 35 + 0.5), 0, 35).astype(np.int64)
    sym = np.frombuffer(ALPHABET.encode(), np.uint8)[idx]
    stream = np.repeat(sym, 5).tobytes().decode()
    fr = frame_str(wpm)
    return f"{call}  {call}  {call} RES{n} RES{n} {fr} {stream} {fr}"


def vote_word(words):
    words = [w for w in words if w]
    if not words:
        return None
    ln = Counter(len(w) for w in words).most_common(1)[0][0]
    ws = [w for w in words if len(w) == ln]
    out = ""
    for i in range(ln):
        c = Counter(w[i] for w in ws if w[i] != "?")
        out += c.most_common(1)[0][0] if c else "?"
    return out


def _is_marker(w: str) -> bool:
    n = w.count(FRAME_CHAR)
    return n >= 2 and n >= len(w) / 2.0            # tolerates a damaged character or two


def parse_text(text: str) -> dict:
    """Split decoded text into header words, the pixel payload and the markers (word based)."""
    text = re.sub(r"[^A-Z0-9?" + re.escape(FRAME_CHAR) + r" ]", "", text.upper())
    info = {"text": text, "callsign": None, "res": None, "header_ok": False, "payload": "", "closed": False}
    words = []
    for w in text.split():
        if len(w) > 12 and re.search(re.escape(FRAME_CHAR) + "{4,}", w):      # marker glued to data
            words += [p for p in re.split("(" + re.escape(FRAME_CHAR) + "{4,})", w) if p]
        else:
            words.append(w)
    mk = [i for i, w in enumerate(words) if _is_marker(w)]
    if not mk:
        return info
    groups, cur = [], [mk[0]]
    for i in mk[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    head_words = words[: groups[0][0]]
    end = groups[-1][0] if len(groups) > 1 else None
    info["payload"] = "".join(words[groups[0][-1] + 1: end])
    info["closed"] = end is not None

    res_votes = []
    meta = [w for w in head_words[-2:] if re.fullmatch(r"RES[0-9?]{2,3}", w)]
    for w in meta:
        m = re.fullmatch(r"RES(\d{2,3})", w)
        if m and int(m.group(1)) in RESOLUTIONS:
            res_votes.append(int(m.group(1)))
    if meta:
        head_words = head_words[: len(head_words) - len(meta)]
    info["res"] = Counter(res_votes).most_common(1)[0][0] if res_votes else None
    cands = [w for w in head_words[-3:] if len(w) <= MAX_CALL]       # longer "words" are noise, not a call sign
    call = vote_word(cands)
    if call is not None and not re.search(r"[A-Z0-9]", call):
        call = None
    info["callsign"] = call
    info["header_ok"] = call is not None
    return info


class DecodeError(Exception):
    pass


def decode_pixels(info: dict, forced_n: int | None = None, allow_partial: bool = False):
    """Majority-vote the 5x repeated symbols. Returns (pixels(n,n), n, report_lines)."""
    payload = info["payload"]
    n = forced_n or info.get("res")
    notes = []
    if not n:
        err, n = min((abs(len(payload) - r * r * 5), r) for r in RESOLUTIONS)
        if err > max(25, len(payload) // 20):
            raise DecodeError("Resolution metadata missing and the data length matches no known resolution.")
        notes.append(f"Resolution inferred from data length: {n} x {n}")
    need = n * n * 5
    if len(payload) < need:
        if not allow_partial:
            raise DecodeError(f"Insufficient data: received {len(payload)} symbols, expected {need}. "
                              f"Check the resolution setting.")
        notes.append(f"Only {len(payload)} of {need} symbols were received; the rest is left black.")
        payload = payload + "?" * (need - len(payload))
    extra = len(payload) - need
    if extra > 0:
        notes.append(f"{extra} extra character(s) after the data were ignored.")
    table = np.full(256, 36, np.int64)
    for i, ch in enumerate(ALPHABET):
        table[ord(ch)] = i
    arr = table[np.frombuffer(payload[:need].encode("ascii", "replace"), np.uint8)].reshape(n * n, 5)
    rows = np.arange(n * n)
    counts = np.zeros((n * n, 37), np.int16)
    for k in range(5):
        counts[rows, arr[:, k]] += 1
    c36 = counts[:, :36]
    win = c36.argmax(1)
    best = c36.max(1)
    ties = (c36 == best[:, None]).sum(1)
    uncertain = (best < 3) | (ties > 1)
    corrected = (arr != win[:, None]).any(1) & (best > 0)
    px = np.where(best > 0, np.floor(win / 35.0 * 255 + 0.5), 0).astype(np.uint8)
    invalid = int((arr == 36).sum())
    rep = list(notes)
    rep.append(f"Resolution: {n} x {n} ({n * n} px)")
    rep.append(f"Groups with a mismatched character: {int(corrected.sum())}")
    rep.append(f"Uncertain groups (no clear majority): {int(uncertain.sum())}")
    if invalid:
        rep.append(f"Invalid / unreadable symbols: {invalid}")
    rep.append("Some pixels were rendered from a best guess." if uncertain.any()
               else "All groups returned a clear majority.")
    return px.reshape(n, n), n, rep


# --------------------------------------------------------------------------- #
#  Encoder: text -> Morse timing -> IQ file
# --------------------------------------------------------------------------- #
def morse_segments(text: str):
    """Alternating OFF/ON durations in Morse units (OFF first). dot=1 dash=3, gaps 1/3/7."""
    chars, lead, skipped, first = [], [], 0, True
    for word in text.upper().split():
        wfirst = True
        for ch in word:
            if ch not in PATTERN:
                skipped += 1
                continue
            lead.append(0 if first else (7 if wfirst else 3))
            chars.append(ch)
            first = wfirst = False
    if not chars:
        return None, skipped, 0
    pats = [PATTERN[c] for c in chars]
    lens = np.fromiter((len(p) for p in pats), np.int64, len(pats))
    segs = np.concatenate(pats)
    segs[np.cumsum(lens) - lens] = lead
    return segs, skipped, len(chars)


def pick_rate(wpm: float) -> int:
    for r in STD_RATES:
        if r >= wpm:
            return r
    return int(math.ceil(wpm / 1000.0) * 1000)


def export_plan(text: str, wpm: float, fmt: str, ramp_ms: float) -> dict:
    """Everything the export needs, plus the warnings/errors shown in the info box."""
    plan = {"warn": [], "err": [], "ok": False, "wpm": wpm, "fmt": fmt, "lpm": wpm * 5.0}
    limit = RF64_MAX_WPM if fmt == "rf64" else WAV_MAX_WPM
    if not text.strip():
        plan["err"].append("Nothing to send. Open an image first.")
        return plan
    if wpm < MIN_WPM:
        plan["err"].append(f"Speed must be at least {MIN_WPM:.0f} WPM.")
    if wpm > limit:
        extra = " Switch the format to RF64 for up to 150,000 WPM." if fmt == "wav" else ""
        plan["err"].append(f"{wpm:,.0f} WPM exceeds the {limit:,.0f} WPM limit for this format.{extra}")
    if plan["err"]:
        return plan
    segs, skipped, nchars = morse_segments(text)
    if segs is None:
        plan["err"].append("The output has no characters that can be sent as Morse.")
        return plan
    fs = BASE_RATE
    if wpm > fs:
        fs = pick_rate(wpm)
        plan["warn"].append(f"{wpm:,.0f} WPM exceeds the {BASE_RATE:,} Hz sample rate, so the sample rate "
                            f"is raised to {fs:,} Hz for this file.")
    spu = fs * 1.2 / wpm                                  # samples per Morse unit (PARIS timing)
    units = int(segs.sum())
    pad = int(round(pad_seconds(wpm) * fs))
    frames = 2 * pad + int(math.floor(units * spu + 0.5))
    bw = min_bandwidth_hz(wpm, ramp_ms, fs)
    plan.update(segs=segs, fs=fs, spu=spu, units=units, frames=frames, bytes=frames * 4, chars=nchars,
                skipped=skipped, ramp_ms=ramp_ms, duration=frames / fs, bandwidth=bw, pad=pad,
                marker=frame_str(wpm), marker_len=marker_len(wpm))
    if spu < 2.0:
        plan["warn"].append(f"A dot is only {spu:.2f} samples long at {fs:,} Hz: keying is right at the limit "
                            f"of the sample rate, so weak or noisy copy may fail. Lower the WPM for a more robust file.")
    if skipped:
        plan["warn"].append(f"{skipped} character(s) with no Morse equivalent will be skipped.")
    if fmt == "wav" and frames > WAV_MAX_FRAMES:
        need = math.ceil(fs * 1.2 * units / (WAV_MAX_FRAMES - 2 * pad) * 10) / 10
        plan["err"].append(f"The file would exceed the 4 GiB WAV size limit. Use RF64, or send at "
                           f"{need:,.1f} WPM or faster.")
    elif plan["bytes"] > 8 * 2 ** 30:
        plan["warn"].append("Very large file: make sure the destination drive has enough free space.")
    plan["ok"] = not plan["err"]
    return plan


def wav_header(fs: int, frames: int, rf64: bool) -> bytes:
    data = frames * 4
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 2, fs, fs * 4, 4, 16)
    if not rf64:
        return b"RIFF" + struct.pack("<I", 36 + data) + b"WAVE" + fmt + b"data" + struct.pack("<I", data)
    riff = 4 + (8 + 28) + (8 + 16) + (8 + data)
    ds64 = struct.pack("<4sIQQQI", b"ds64", 28, riff, data, frames, 0)
    return b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64 + fmt + b"data" + struct.pack("<I", 0xFFFFFFFF)


def boxcar(x: np.ndarray, m: int) -> np.ndarray:
    """Centred moving average of odd length m (same length; first/last m//2 samples are approximate)."""
    if m <= 1 or len(x) <= 2 * (m // 2) + 1:
        return x
    h = m // 2
    n = len(x)
    cp = np.concatenate(([0], np.cumsum(x, dtype=np.complex128 if np.iscomplexobj(x) else np.float64)))
    y = np.empty(n, dtype=cp.dtype)
    y[h:n - h] = (cp[2 * h + 1:] - cp[:n - 2 * h]) / m
    y[:h] = y[h]
    y[n - h:] = y[n - h - 1]
    return y


def write_iq_file(path, plan, tone_hz=0.0, progress=None, cancelled=lambda: False):
    """Stream the keyed carrier to disk in chunks (WAV or RF64, 16-bit stereo; L = I, R = Q)."""
    segs, fs, spu, pad = plan["segs"], plan["fs"], plan["spu"], plan["pad"]
    B = pad + np.floor(np.cumsum(segs, dtype=np.float64) * spu + 0.5).astype(np.int64)   # cumulative rounding
    total = int(B[-1]) + pad
    on_s, on_e = B[0::2], B[1::2]
    wt = min(plan["ramp_ms"] * fs / 1000.0, spu / 2.0)            # key-edge width, centred on each boundary
    m = 1 if wt < 2 else (int(wt / 2) | 1)
    mar = 2 * (m // 2)
    CH = 1 << 20
    amp = LEVEL * 32767.0
    with open(path, "wb") as f:
        f.write(wav_header(fs, total, plan["fmt"] == "rf64"))
        for a in range(0, total, CH):
            b = min(total, a + CH)
            lo, w = a - mar, (b - a) + 2 * mar
            j0 = int(np.searchsorted(on_e, lo, "right"))
            j1 = int(np.searchsorted(on_s, b + mar, "left"))
            key = np.zeros(w)
            if j1 > j0:
                s = np.clip(on_s[j0:j1] - lo, 0, w)
                e = np.clip(on_e[j0:j1] - lo, 0, w)
                d = np.bincount(s, minlength=w + 1)[: w + 1] - np.bincount(e, minlength=w + 1)[: w + 1]
                key = np.cumsum(d[:w]).astype(np.float64)
            env = boxcar(boxcar(key, m), m)[mar: mar + (b - a)] if m > 1 else key[mar: mar + (b - a)]
            out = np.empty((b - a, 2), "<i2")
            if tone_hz == 0:
                out[:, 0] = np.rint(env * amp)
                out[:, 1] = 0
            else:
                ph = np.mod(np.arange(a, b, dtype=np.float64) * (tone_hz / fs), 1.0) * (2 * np.pi)
                out[:, 0] = np.rint(env * amp * np.cos(ph))
                out[:, 1] = np.rint(env * amp * np.sin(ph))
            f.write(out.tobytes())
            if progress:
                progress(int(100 * b / total))
            if cancelled():
                raise InterruptedError("Export cancelled.")


# --------------------------------------------------------------------------- #
#  IQ sources (WAV / RF64 file with any bit depth, or an in-memory buffer)
# --------------------------------------------------------------------------- #
class ArrayIQ:
    def __init__(self, z: np.ndarray, fs: int):
        self.z, self.fs, self.n = z, int(fs), len(z)
        self.desc = "live buffer"

    def read(self, a: int, b: int) -> np.ndarray:
        return self.z[max(0, a):min(self.n, b)]

    def close(self):
        pass


class WavIQ:
    """Random-access complex64 reader for 2-channel WAV/RF64: PCM 8/16/24/32-bit, float 32/64-bit."""

    def __init__(self, path: str):
        self.path = path
        size = os.path.getsize(path)
        fmt = data = ds64 = None
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64", b"BW64") or head[8:12] != b"WAVE":
                raise ValueError("This is not a RIFF/RF64 WAVE file.")
            pos = 12
            while pos + 8 <= size:
                f.seek(pos)
                cid, csz = struct.unpack("<4sI", f.read(8))
                body = pos + 8
                if cid == b"ds64":
                    ds64 = struct.unpack("<QQQ", f.read(24))
                elif cid == b"fmt ":
                    d = f.read(min(csz, 40))
                    tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", d[:16])
                    if tag == 0xFFFE and len(d) >= 26:
                        tag = struct.unpack("<H", d[24:26])[0]
                    fmt = (tag, ch, rate, bits)
                elif cid == b"data":
                    dsz = csz
                    if csz == 0xFFFFFFFF and ds64:
                        dsz = ds64[1]
                    if csz == 0xFFFFFFFF or body + dsz > size:
                        dsz = size - body
                    data = (body, dsz)
                    break
                pos = body + csz + (csz & 1)
        if not fmt or not data:
            raise ValueError("The WAV file has no usable fmt/data chunk.")
        tag, ch, rate, bits = fmt
        if ch != 2:
            raise ValueError(f"The file has {ch} channel(s). An IQ file needs 2 channels (Left = I, Right = Q).")
        if tag == 1 and bits in (8, 16, 24, 32):
            self.kind = {8: "u8", 16: "i16", 24: "i24", 32: "i32"}[bits]
        elif tag == 3 and bits in (32, 64):
            self.kind = {32: "f32", 64: "f64"}[bits]
        else:
            raise ValueError(f"Unsupported sample format (tag {tag}, {bits}-bit).")
        self.fs, self.bits, self.tag = int(rate), bits, tag
        self.fb = 2 * bits // 8
        self.n = data[1] // self.fb
        if self.n < 16:
            raise ValueError("The file contains no audio data.")
        self.raw = np.memmap(path, mode="r", dtype=np.uint8, offset=data[0], shape=(self.n * self.fb,))
        self.desc = f"{bits}-bit {'float' if tag == 3 else 'PCM'}"

    def read(self, a: int, b: int) -> np.ndarray:
        a, b = max(0, a), min(self.n, b)
        seg = np.ascontiguousarray(self.raw[a * self.fb: b * self.fb])
        k = self.kind
        if k == "i16":
            x = seg.view("<i2").reshape(-1, 2).astype(np.float32) / 32768.0
        elif k == "f32":
            x = seg.view("<f4").reshape(-1, 2)
        elif k == "f64":
            x = seg.view("<f8").reshape(-1, 2).astype(np.float32)
        elif k == "i32":
            x = seg.view("<i4").reshape(-1, 2).astype(np.float32) / 2147483648.0
        elif k == "u8":
            x = (seg.reshape(-1, 2).astype(np.float32) - 128.0) / 128.0
        else:  # 24-bit
            t = seg.reshape(-1, 2, 3)
            v = (t[..., 0].astype(np.int32) | (t[..., 1].astype(np.int32) << 8)
                 | (t[..., 2].astype(np.int8).astype(np.int32) << 16))
            x = v.astype(np.float32) / 8388608.0
        return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)

    def close(self):
        self.raw = None


# --------------------------------------------------------------------------- #
#  Morse demodulator for IQ
# --------------------------------------------------------------------------- #
CH_READ = 1 << 22
L_CANDIDATES = (1, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049)


def mix(z, f0, fs, start, dc=0j):
    if dc != 0:
        z = z - np.complex64(dc)                       # remove an SDR DC spike
    if f0 == 0:
        return z.astype(np.complex64, copy=False)
    ph = np.mod(np.arange(start, start + len(z), dtype=np.float64) * (f0 / fs), 1.0) * (2 * np.pi)
    return z * (np.cos(ph) - 1j * np.sin(ph)).astype(np.complex64)


def spectrum_of(src, nblocks=24, nfft=8192):
    n = src.n
    nfft = min(nfft, 1 << max(8, int(math.log2(max(n, 256)))))
    nb = max(1, min(nblocks, n // nfft))
    win = np.hanning(nfft)
    P = np.zeros(nfft)
    for s in np.linspace(0, max(0, n - nfft), nb).astype(np.int64):
        z = src.read(int(s), int(s) + nfft)
        if len(z) < nfft:
            continue
        P += np.abs(np.fft.fft(z * win)) ** 2
    P = np.fft.fftshift(P)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / src.fs))
    return freqs, P


def find_carrier(src):
    """Strongest keyed line in the spectrum (a constant DC spike is rejected by a modulation test)."""
    freqs, P = spectrum_of(src)
    nfft = len(P)
    df = src.fs / nfft
    work = P.copy()
    sep = max(3, int(150 / df))
    cands = []
    for _ in range(4):
        i = int(np.argmax(work))
        if work[i] <= 0:
            break
        f = freqs[i]
        if 0 < i < nfft - 1:
            a, b, c = np.log10(P[i - 1] + 1e-30), np.log10(P[i] + 1e-30), np.log10(P[i + 1] + 1e-30)
            den = a - 2 * b + c
            if den != 0:
                f += 0.5 * (a - c) / den * df
        cands.append(float(f))
        work[max(0, i - sep): i + sep + 1] = 0
    if not cands:
        return 0.0, (freqs, P)
    starts = np.linspace(0, max(0, src.n - 32768), 8).astype(np.int64)

    def keyed(f):
        score = 0.0
        for st in starts:
            z = src.read(int(st), int(st) + 32768)
            if len(z) == 0:
                continue
            mag = np.abs(boxcar(mix(z, f, src.fs, int(st)), 8))
            mu = mag.mean()
            if mu > 0:
                score = max(score, float(mag.std() / mu))
        return score

    best, bscore = cands[0], -1.0
    for f in cands:                                    # candidates are ordered strongest first
        sc = keyed(f)
        if sc >= 0.35:
            return f, (freqs, P)
        if sc > bscore:
            best, bscore = f, sc
    return best, (freqs, P)


def activity_start(src, f0, blk=4096):
    pw = []
    for a in range(0, src.n, CH_READ):
        z = src.read(a, a + CH_READ)
        y = boxcar(mix(z, f0, src.fs, a), 33)
        m = len(y) // blk
        if m:
            pw.append((np.abs(y[: m * blk]) ** 2).reshape(m, blk).mean(1))
    if not pw:
        return 0
    pw = np.concatenate(pw)
    base, top = np.median(pw), np.percentile(pw, 98)
    if top <= base * 2.0:
        return 0
    idx = np.flatnonzero(pw > base + 0.25 * (top - base))
    return 0 if len(idx) == 0 else max(0, (int(idx[0]) - 1) * blk)


def run_lengths(env, thr):
    b = env > thr
    d = np.flatnonzero(b[1:] != b[:-1]) + 1
    return np.diff(np.concatenate(([0], d, [len(b)]))), bool(b[0])


_KGRID = np.array([1.0, 3.0, 7.0])


def fit_unit(runs, max_runs=200000):
    """Best samples-per-unit for the run lengths (clusters at 1, 3 and 7 units). -> (spu, inlier fraction)."""
    r = runs[:max_runs]
    if len(r) < 12:
        return None
    vals, cnt = np.unique(r, return_counts=True)
    vals = vals.astype(np.float64)
    cnt = cnt.astype(np.float64)
    hi = min(float(np.percentile(r, 60)) * 1.5 + 4, 60000.0)
    grid = np.exp(np.linspace(math.log(1.0), math.log(max(hi, 2.0)), 420))

    def cost(sv):
        t = sv[:, None, None] * _KGRID[None, None, :]
        err = np.abs(vals[None, :, None] - t)
        rel = (err / (t + 0.5)) ** 2
        kb = rel.argmin(2)
        j = np.minimum(np.take_along_axis(rel, kb[..., None], 2)[..., 0], 0.04)   # cap: silences can't dominate
        e = np.take_along_axis(err, kb[..., None], 2)[..., 0]
        tk = sv[:, None] * _KGRID[kb]
        inl = (e <= np.maximum(1.0, 0.25 * tk))
        return (j * cnt).sum(1) / cnt.sum(), (inl * cnt).sum(1) / cnt.sum()

    j, q = cost(grid)
    s0 = float(grid[int(np.argmin(j))])
    fine = np.linspace(s0 * 0.96, s0 * 1.04, 81)
    jf, qf = cost(fine)
    i = int(np.argmin(jf))
    return float(fine[i]), float(qf[i])


def estimate(src, live=False, f0_override=None):
    """Find carrier, start of activity, smoothing length, slicer threshold and speed."""
    fs = src.fs
    if f0_override is None:
        f0, spec = find_carrier(src)
    else:
        f0, spec = float(f0_override), spectrum_of(src)
    start = activity_start(src, f0)
    a0 = max(0, start - 2048)
    a1 = min(src.n, a0 + ((1 << 21) if live else (1 << 22)))
    zw = src.read(a0, a1)
    dc = complex(zw.mean()) if abs(f0) > 1000 else 0j
    y = mix(zw, f0, fs, a0, dc)
    e9 = np.abs(boxcar(y, 9))
    sub9 = e9[:: max(1, len(e9) // 200000)]
    gate = e9 > 0.5 * (np.percentile(sub9, 10) + np.percentile(sub9, 95))
    r = np.sum(y[1:] * np.conj(y[:-1]) * (gate[1:] & gate[:-1]))     # lag-1 phase slope = residual carrier error
    if abs(r) > 0:
        f0 += float(np.angle(r)) * fs / (2 * np.pi)
        y = mix(zw, f0, fs, a0, dc)
    results = []
    for L in L_CANDIDATES:
        if L > max(1, (a1 - a0) // 8):
            break
        env = np.abs(boxcar(y, L)).astype(np.float32)
        sub = env[:: max(1, len(env) // 400000)]
        lo, hi = np.percentile(sub, 10), np.percentile(sub, 95)
        if hi - lo <= 1e-9:
            continue
        thr = float(lo + 0.5 * (hi - lo))
        runs, _ = run_lengths(env, thr)
        fit = fit_unit(runs)
        if fit is None:
            continue
        results.append((fit[1], L, fit[0], thr))
        if L > 1 and fit[1] >= 0.9 and L > 0.75 * fit[0]:
            break
    if not results:
        return {"f0": f0, "start": start, "score": 0.0, "spec": spec, "fs": fs, "dc": dc}
    top = max(r[0] for r in results)
    ok = [r for r in results if r[0] >= top - 0.03 and r[1] <= max(1.0, r[2] / 2)] or [max(results)]
    score, L, s, thr = max(ok, key=lambda r: r[1])
    return {"f0": f0, "start": start, "score": score, "L": L, "s": s, "thr": thr, "spec": spec, "fs": fs,
            "wpm": fs * 1.2 / s, "dc": dc}


def extract_runs(src, f0, L, thr, a0=0, a1=None, cancel=None, dc=0j):
    a1 = src.n if a1 is None else min(a1, src.n)
    h = L // 2
    edges, prev, first_on = [], None, False
    for a in range(a0, a1, CH_READ):
        if cancel and cancel():
            raise InterruptedError("Cancelled.")
        b = min(a1, a + CH_READ)
        lo, hi = max(0, a - h), min(src.n, b + h)
        env = np.abs(boxcar(mix(src.read(lo, hi), f0, src.fs, lo, dc), L))[a - lo: b - lo]
        bb = env > thr
        if prev is None:
            first_on = bool(bb[0])
        elif bb[0] != prev:
            edges.append(np.array([a], dtype=np.int64))
        edges.append(np.flatnonzero(bb[1:] != bb[:-1]).astype(np.int64) + 1 + a)
        prev = bool(bb[-1])
    bounds = np.concatenate([[a0]] + edges + [[a1]])
    return np.diff(bounds), first_on


def runs_to_text(runs, first_on, s):
    """Classify runs into dots/dashes/gaps and look the characters up. '?' = unknown pattern."""
    if len(runs) < 2:
        return ""
    idx = np.arange(len(runs))
    on_idx = np.flatnonzero((idx % 2 == 0) == first_on)
    on_r = runs[on_idx].astype(np.float64)
    n = len(on_r)
    if n == 0:
        return ""
    nxt = on_idx + 1
    gap_r = np.where(nxt < len(runs), runs[np.minimum(nxt, len(runs) - 1)], 10 ** 9).astype(np.float64)
    dash = (np.abs(on_r - 3 * s) < np.abs(on_r - s)).astype(np.int64)
    g = np.abs(gap_r[:, None] - s * _KGRID[None, :]).argmin(1)         # 0: element, 1: letter, 2: word
    g[-1] = 2                                                          # the stream ends after the last element
    char_end = g >= 1
    gid = np.cumsum(char_end) - char_end
    starts = np.flatnonzero(np.concatenate(([True], char_end[:-1])))
    pos = np.minimum(np.arange(n) - starts[gid], 8)
    key = np.bincount(gid, weights=(dash + 1) * 3.0 ** pos).astype(np.int64)
    k = np.minimum(np.searchsorted(KEY_ARR, key), len(KEY_ARR) - 1)
    chars = np.where(KEY_ARR[k] == key, KEY_CHR[k], ord("?")).astype(np.uint8)
    wend = (g == 2)[char_end]
    out = np.zeros(2 * len(chars), np.uint8)
    out[0::2] = chars
    out[1::2] = np.where(wend[: len(chars)], 32, 0)
    return out[out != 0].tobytes().decode("ascii")


def despeckle(runs, minlen):
    """Merge glitch runs shorter than minlen into their neighbours (noise flicker around the slicer level)."""
    for _ in range(12):
        idx = np.flatnonzero(runs[1:-1] < minlen) + 1
        if len(idx) == 0:
            break
        sel = idx[np.concatenate(([True], np.diff(idx) > 2))]
        r2 = runs.copy()
        r2[sel - 1] += runs[sel] + runs[sel + 1]
        keep = np.ones(len(runs), bool)
        keep[sel] = False
        keep[sel + 1] = False
        runs = r2[keep]
    return runs


def demod_text(src, p, a0=None, a1=None, cancel=None):
    runs, first_on = extract_runs(src, p["f0"], p["L"], p["thr"], p["start"] if a0 is None else a0, a1, cancel,
                                  p.get("dc", 0j))
    return runs_to_text(despeckle(runs, 0.4 * p["s"]), first_on, p["s"])


def decode_source(src, emit=None, forced_n=None, f0_override=None, cancel=None, allow_partial=False):
    """Full decode: find the header first (call sign, resolution), then the whole picture."""
    emit = emit or (lambda *a: None)
    emit("stage", "Looking for the carrier and Morse speed...")
    p = estimate(src, f0_override=f0_override)
    emit("spectrum", p["spec"], p["f0"])
    if p["score"] < 0.45 or "s" not in p:
        raise DecodeError("No Morse signal was found (the keying could not be measured).")
    base = {"wpm": p["wpm"], "f0": p["f0"], "fs": src.fs}
    emit("stage", "Listening for the header...")
    head_end = min(src.n, p["start"] + int(240 * src.fs))
    head = parse_text(demod_text(src, p, p["start"], head_end, cancel))
    if head["header_ok"]:
        emit("header", dict(base, callsign=head["callsign"], res=head["res"]))
    else:
        emit("stage", "Header not seen yet; decoding everything...")
    emit("stage", "Decoding the picture...")
    text = head["text"] if head_end >= src.n else demod_text(src, p, p["start"], None, cancel)
    info = parse_text(text)
    if not info["header_ok"]:
        raise DecodeError("The start marker was not found. The signal may be too weak or "
                          "this is not a Bitmap Burst transmission.")
    if not head["header_ok"]:
        emit("header", dict(base, callsign=info["callsign"], res=info["res"]))
    px, n, rep = decode_pixels(info, forced_n, allow_partial)
    return dict(base, pixels=px, n=n, callsign=info["callsign"], report=rep, text=info["text"],
                complete=info["closed"] and len(info["payload"]) >= n * n * 5)


def psd_db(freqs, P, points=512):
    """Reduce a PSD to ~points max-pooled dB values for display."""
    k = max(1, len(P) // points)
    m = (len(P) // k) * k
    pp = P[:m].reshape(-1, k).max(1)
    ff = freqs[:m].reshape(-1, k).mean(1)
    db = 10 * np.log10(pp + 1e-20)
    return ff, db
