#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morse Code Bitmap Burst Encoder - Qt6 GUI
==========================================
A glossy, Aero / MMSSTV-styled front end for core_part.py.

Encode tab : pick an image, a callsign and a speed; export a baseband IQ file as WAV
             or RF64 (stereo, L = I, R = Q).
Decode tab : import an IQ file (any common bit depth) or listen live on a selected audio
             input (SDR software in raw / binaural IQ demod mode); the call sign shows up
             as soon as the header is heard, then the picture is decoded.

Run:  python3 gui.py      (pip install PySide6 numpy)
"""
from __future__ import annotations

import os
import shutil
import sys
import traceback

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer, QSize
from PySide6.QtCore import QUrl
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QAction, QIcon, QDesktopServices
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QLineEdit, QDoubleSpinBox, QFileDialog,
    QProgressBar, QPlainTextEdit, QRadioButton, QButtonGroup, QMessageBox, QToolBar, QStatusBar,
    QSizePolicy, QFrame, QStyleFactory,
)

import core_part as core

APP_TITLE = core.APP_NAME


# --------------------------------------------------------------------------- #
#  Theme: glossy / Aero-style QSS for a dark and a light palette
# --------------------------------------------------------------------------- #
def build_qss(dark: bool) -> str:
    if dark:
        bg, panel, panel2 = "#1b2430", "#222c3a", "#2c3848"
        text, subtext, edge = "#e7edf5", "#9fb0c3", "#3a4759"
        accent, accent2 = "#5aaeff", "#2a68c8"
        field_bg, field_bg2, tab_sel = "#141b26", "#1a2330", "#2c3848"
        gloss = "rgba(255,255,255,0.09)"
    else:
        bg, panel, panel2 = "#e9eff8", "#ffffff", "#f1f5fb"
        text, subtext, edge = "#1a2330", "#5a6b80", "#b9c7da"
        accent, accent2 = "#3a8cf0", "#1b5fc0"
        field_bg, field_bg2, tab_sel = "#ffffff", "#f7f9fc", "#ffffff"
        gloss = "rgba(255,255,255,0.65)"
    return f"""
    QMainWindow, QWidget {{ background: {bg}; color: {text}; font-size: 12px; }}
    QToolBar {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {panel2}, stop:0.5 {panel}, stop:1 {bg});
               border-bottom: 1px solid {edge}; spacing: 6px; padding: 4px; }}
    QStatusBar {{ background: {panel}; border-top: 1px solid {edge}; color: {subtext}; }}
    QTabWidget::pane {{ border: 1px solid {edge}; border-radius: 8px; background: {panel}; top: -1px; }}
    QTabBar::tab {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {panel2}, stop:0.5 {panel}, stop:0.51 {bg}, stop:1 {panel});
                   border: 1px solid {edge}; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px;
                   padding: 8px 22px; margin-right: 2px; color: {subtext}; font-weight: 600; }}
    QTabBar::tab:selected {{ background: {tab_sel}; color: {text}; border-bottom: 3px solid {accent}; }}
    QTabBar::tab:hover {{ color: {text}; }}
    QGroupBox {{ border: 1px solid {edge}; border-radius: 9px; margin-top: 14px; padding: 12px 8px 8px 8px;
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {panel2}, stop:0.12 {panel2}, stop:0.13 {panel}, stop:1 {panel});
                font-weight: 600; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {accent}; }}
    QPushButton {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {panel2}, stop:0.49 {panel}, stop:0.5 {bg}, stop:1 {panel2});
                  border: 1px solid {edge}; border-radius: 7px; padding: 6px 14px; color: {text}; font-weight: 600; }}
    QPushButton:hover {{ border: 1px solid {accent}; }}
    QPushButton:pressed {{ background: {panel2}; }}
    QPushButton:disabled {{ color: {subtext}; }}
    QPushButton#primary {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {accent}, stop:0.49 {accent2}, stop:0.5 {accent2}, stop:1 {accent});
                           color: white; border: 1px solid {accent2}; }}
    QPushButton#primary:hover {{ border: 1px solid white; }}
    QPushButton#primary:disabled {{ background: {edge}; color: {subtext}; border-color: {edge}; }}
    QLineEdit, QDoubleSpinBox, QComboBox, QPlainTextEdit {{ background: {field_bg}; border: 1px solid {edge}; border-radius: 6px;
                                                           padding: 4px 6px; color: {text}; selection-background-color: {accent}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView {{ background: {field_bg}; color: {text}; selection-background-color: {accent}; }}
    QPlainTextEdit {{ background: {field_bg2}; font-family: Consolas, "Courier New", monospace; }}
    QProgressBar {{ border: 1px solid {edge}; border-radius: 7px; background: {field_bg}; text-align: center; color: {text}; height: 16px; }}
    QProgressBar::chunk {{ border-radius: 6px; background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {accent}, stop:0.5 {accent2}, stop:1 {accent}); }}
    QRadioButton, QLabel {{ background: transparent; color: {text}; }}
    QRadioButton::indicator {{ width: 12px; height: 12px; border-radius: 7px; border: 1px solid {subtext}; background: {field_bg}; }}
    QRadioButton::indicator:checked {{ background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 white, stop:0.45 {accent}, stop:1 {accent2}); border: 1px solid {accent2}; }}
    QLabel#subtle {{ color: {subtext}; }}
    QLabel#preview {{ background: {field_bg}; border: 1px solid {edge}; border-radius: 8px; }}
    QLabel#gloss {{ background: {gloss}; }}
    """


PALETTE = {
    "dark": dict(bg="#141b26", grid="#223049", line="#5aaeff", marker="#ff7a4f", axis="#7d8da3"),
    "light": dict(bg="#ffffff", grid="#dbe4f0", line="#2f7dde", marker="#d9622c", axis="#5a6b80"),
}


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def pixels_to_qpixmap(px: np.ndarray, target: int = 280) -> QPixmap:
    n = px.shape[0]
    img = QImage(np.ascontiguousarray(px).tobytes(), n, n, n, QImage.Format_Grayscale8).copy()
    scale = max(1, target // max(1, n))
    return QPixmap.fromImage(img).scaled(n * scale, n * scale, Qt.KeepAspectRatio, Qt.FastTransformation)


def qimage_to_gray_array(img: QImage, n: int) -> np.ndarray:
    """Fit the whole image into an n x n square (letterboxed in white) and convert to 8-bit grayscale."""
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    scale = min(n / w, n / h)
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    scaled = img.scaled(sw, sh, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    bpl = scaled.bytesPerLine()
    arr = np.frombuffer(scaled.constBits(), np.uint8, sh * bpl).reshape(sh, bpl)[:, : sw * 3]
    rgb = arr.reshape(sh, sw, 3).astype(np.float32)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    canvas = np.full((n, n), 255.0, np.float32)
    oy, ox = (n - sh) // 2, (n - sw) // 2
    canvas[oy:oy + sh, ox:ox + sw] = gray
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def fmt_hz(hz: float) -> str:
    return f"{hz / 1000:,.2f} kHz" if hz >= 1000 else f"{hz:,.0f} Hz"


def fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:,.0f} B" if unit == "B" else f"{b:,.1f} {unit}"
        b /= 1024.0


def fmt_dur(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.2f} s"
    d, rem = divmod(int(s), 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    parts = ([f"{d}d"] if d else []) + ([f"{h}h"] if d or h else []) + [f"{m}m", f"{sec}s"]
    return " ".join(parts)


def bandwidth_text(bw: dict, note: str = "") -> str:
    return (f"Minimum: ~{fmt_hz(bw['min_hz'])}   Recommended: ~{fmt_hz(bw['comfortable_hz'])}   "
            f"Clean edges: ~{fmt_hz(bw['clean_hz'])}\n"
            f"Total passband centred on the tone (e.g. SSB filter). It is one on/off keyed tone, so no harmonics "
            f"need to pass.{note}")


class ElidedLabel(QLabel):
    """One-line label that never forces the layout wider: long text is cut with an ellipsis (full text in the tooltip)."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        self.setText(text)

    def setText(self, text):
        self._full = str(text)
        self.setToolTip(self._full if len(self._full) > 24 else "")
        self._apply()

    def _apply(self):
        w = max(0, self.width() - 2)
        QLabel.setText(self, self.fontMetrics().elidedText(self._full, Qt.ElideRight, w) if w > 0 else self._full)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._apply()


# --------------------------------------------------------------------------- #
#  Spectrum widget
# --------------------------------------------------------------------------- #
class SpectrumWidget(QWidget):
    def __init__(self, dark=True, parent=None):
        super().__init__(parent)
        self.dark, self.freqs, self.db, self.f0, self.bw = dark, None, None, None, None
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def set_theme(self, dark: bool):
        self.dark = dark
        self.update()

    def set_data(self, freqs, P, f0=None, bw=None):
        self.freqs, self.db = core.psd_db(freqs, P, 360)
        self.f0, self.bw = f0, bw
        self.update()

    def set_bandwidth(self, bw):
        self.bw = bw
        self.update()

    def paintEvent(self, ev):
        pal = PALETTE["dark" if self.dark else "light"]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect()
        p.fillRect(r, QColor(pal["bg"]))
        plot = r.adjusted(6, 6, -6, -16)
        p.setPen(QPen(QColor(pal["grid"]), 1))
        for i in range(5):
            y = int(plot.top() + plot.height() * i / 4)
            p.drawLine(plot.left(), y, plot.right(), y)
        if self.freqs is None or len(self.freqs) < 2:
            p.setPen(QColor(pal["axis"]))
            p.drawText(r, Qt.AlignCenter, "No spectrum yet")
            p.end()
            return
        fmin, fmax = float(self.freqs[0]), float(self.freqs[-1])
        dmin, dmax = float(np.percentile(self.db, 2)), float(self.db.max())
        if dmax - dmin < 1:
            dmax = dmin + 1
        span = max(1e-9, fmax - fmin)
        xf = lambda f: plot.left() + (f - fmin) / span * plot.width()
        yf = lambda d: plot.bottom() - max(0.0, min(1.0, (d - dmin) / (dmax - dmin))) * plot.height()
        if self.f0 is not None and self.bw and fmin <= self.f0 <= fmax:
            c = QColor(pal["marker"])
            c.setAlpha(45)
            x1, x2 = xf(max(fmin, self.f0 - self.bw / 2)), xf(min(fmax, self.f0 + self.bw / 2))
            p.fillRect(int(x1), plot.top(), max(1, int(x2 - x1)), plot.height(), c)
        pts = [(xf(f), yf(d)) for f, d in zip(self.freqs, self.db)]
        p.setPen(QPen(QColor(pal["line"]), 1.4))
        for i in range(1, len(pts)):
            p.drawLine(int(pts[i - 1][0]), int(pts[i - 1][1]), int(pts[i][0]), int(pts[i][1]))
        if self.f0 is not None and fmin <= self.f0 <= fmax:
            x = xf(self.f0)
            p.setPen(QPen(QColor(pal["marker"]), 1.6, Qt.DashLine))
            p.drawLine(int(x), plot.top(), int(x), plot.bottom())
            p.setPen(QColor(pal["marker"]))
            p.drawText(int(x) + 4, plot.top() + 12, f"{self.f0:,.0f} Hz")
        p.setPen(QColor(pal["axis"]))
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(plot.left(), r.bottom() - 3, f"{fmin:,.0f} Hz")
        p.drawText(plot.right() - 70, r.bottom() - 3, f"{fmax:,.0f} Hz")
        p.end()


# --------------------------------------------------------------------------- #
#  Background workers
# --------------------------------------------------------------------------- #
class ExportWorker(QThread):
    progress = Signal(int)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, path, plan, tone_hz):
        super().__init__()
        self.path, self.plan, self.tone_hz = path, plan, tone_hz
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            core.write_iq_file(self.path, self.plan, self.tone_hz, progress=self.progress.emit,
                               cancelled=lambda: self._cancel)
            self.done.emit(self.path)
        except InterruptedError:
            self.failed.emit("Export cancelled.")
        except Exception as e:
            self.failed.emit(str(e))


class DecodeWorker(QThread):
    """Runs core.decode_source on a WavIQ file or a live ArrayIQ snapshot."""
    stage = Signal(str)
    spectrum = Signal(object, object, object)
    header = Signal(dict)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, src_factory, forced_n=None, allow_partial=False, close_after=True):
        super().__init__()
        self.src_factory, self.forced_n = src_factory, forced_n
        self.allow_partial, self.close_after = allow_partial, close_after
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        src = None
        try:
            src = self.src_factory()

            def emit(kind, *a):
                if kind == "stage":
                    self.stage.emit(a[0])
                elif kind == "spectrum":
                    self.spectrum.emit(a[0][0], a[0][1], a[1])
                elif kind == "header":
                    self.header.emit(a[0])

            self.done.emit(core.decode_source(src, emit=emit, forced_n=self.forced_n,
                                              cancel=lambda: self._cancel, allow_partial=self.allow_partial))
        except InterruptedError:
            pass
        except core.DecodeError as e:
            self.failed.emit(str(e))
        except Exception:
            self.failed.emit(traceback.format_exc(limit=2))
        finally:
            if src is not None and self.close_after:
                try:
                    src.close()
                except Exception:
                    pass


class MicCapture(QObject):
    """Captures a selected audio input as stereo IQ (L = I, R = Q) into a capped buffer."""
    errorOccurred = Signal(str)
    MAX_SECONDS = 600

    def __init__(self):
        super().__init__()
        self.source = self.io = None
        self.fs = 48000
        self._fmt = None
        self.clear()

    @staticmethod
    def devices():
        return list(QMediaDevices.audioInputs())

    def start(self, device):
        self.stop()
        fmt = None
        for sf in (QAudioFormat.Int16, QAudioFormat.Float, QAudioFormat.Int32):
            f = QAudioFormat()
            f.setSampleRate(48000)
            f.setChannelCount(2)
            f.setSampleFormat(sf)
            if device.isFormatSupported(f):
                fmt = f
                break
        if fmt is None:
            f = device.preferredFormat()
            f.setChannelCount(2)
            if device.isFormatSupported(f):
                fmt = f
        if fmt is None:
            self.errorOccurred.emit("This input cannot deliver a 2-channel (stereo) stream. The decoder needs "
                                    "stereo IQ: Left = I, Right = Q (SDR software in raw / binaural IQ mode).")
            return False
        self.fs, self._fmt = fmt.sampleRate(), fmt
        self.clear()
        self.source = QAudioSource(device, fmt)
        self.io = self.source.start()
        if self.io is None:
            self.errorOccurred.emit("Could not open the selected audio input.")
            return False
        self.io.readyRead.connect(self._on_ready)
        return True

    def stop(self):
        if self.source is not None:
            try:
                self.source.stop()
            except Exception:
                pass
        self.source = self.io = None

    def _on_ready(self):
        if self.io is None:
            return
        data = bytes(self.io.readAll())
        sf = self._fmt.sampleFormat()
        if sf == QAudioFormat.Int16:
            x, scale = np.frombuffer(data[: len(data) // 4 * 4], "<i2"), 32768.0
        elif sf == QAudioFormat.Int32:
            x, scale = np.frombuffer(data[: len(data) // 8 * 8], "<i4"), 2147483648.0
        elif sf == QAudioFormat.Float:
            x, scale = np.frombuffer(data[: len(data) // 8 * 8], "<f4"), 1.0
        else:
            x, scale = np.frombuffer(data[: len(data) // 2 * 2], np.uint8).astype(np.float32) - 128.0, 128.0
        if len(x) < 2:
            return
        x = x.reshape(-1, 2).astype(np.float32) / scale
        self._chunks.append((x[:, 0] + 1j * x[:, 1]).astype(np.complex64))
        self._total += len(x)
        cap = int(self.MAX_SECONDS * self.fs)
        if self._total > cap:
            full = np.concatenate(self._chunks)[-cap:]
            self._chunks, self._total = [full], len(full)

    def snapshot(self):
        if self._cache is None or self._cache_n != len(self._chunks):
            self._cache = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.complex64)
            self._cache_n = len(self._chunks)
        return core.ArrayIQ(self._cache, self.fs)

    def clear(self):
        self._chunks, self._total, self._cache, self._cache_n = [], 0, None, -1

    @property
    def seconds(self):
        return self._total / self.fs if self.fs else 0.0


# --------------------------------------------------------------------------- #
#  Encode tab
# --------------------------------------------------------------------------- #
class EncodeTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.source_image = self.pixels = self.plan = self.worker = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        left, right = QVBoxLayout(), QVBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 2)

        gb = QGroupBox("Source Image")
        g = QGridLayout(gb)
        self.btn_open = QPushButton("Open Image\u2026")
        self.lbl_file = ElidedLabel("No image loaded.")
        self.lbl_file.setObjectName("subtle")
        g.addWidget(self.btn_open, 0, 0)
        g.addWidget(self.lbl_file, 0, 1)
        g.setColumnStretch(1, 1)
        g.addWidget(QLabel("Resolution"), 1, 0)
        self.cmb_res = QComboBox()
        for r in core.RESOLUTIONS:
            self.cmb_res.addItem(f"{r} x {r}", r)
        g.addWidget(self.cmb_res, 1, 1)
        g.addWidget(QLabel("Callsign"), 2, 0)
        self.ed_call = QLineEdit()
        self.ed_call.setPlaceholderText("N0CALL")
        self.ed_call.setMaxLength(16)
        g.addWidget(self.ed_call, 2, 1)
        left.addWidget(gb)

        gb2 = QGroupBox("Speed && File Format")
        g2 = QGridLayout(gb2)
        g2.addWidget(QLabel("Format"), 0, 0)
        hb = QVBoxLayout()
        self.rb_wav = QRadioButton("WAV (PCM, limit 45,000 WPM / 4 GiB)")
        self.rb_rf64 = QRadioButton("RF64 PCM (no 4 GiB limit, limit 150,000 WPM)")
        self.rb_wav.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_wav)
        grp.addButton(self.rb_rf64)
        hb.addWidget(self.rb_wav)
        hb.addWidget(self.rb_rf64)
        g2.addLayout(hb, 0, 1)
        g2.addWidget(QLabel("Speed"), 1, 0)
        self.sp_wpm = QDoubleSpinBox()
        self.sp_wpm.setDecimals(1)
        self.sp_wpm.setRange(core.MIN_WPM, 1_000_000)      # wide on purpose: out-of-range values get a warning
        self.sp_wpm.setValue(20.0)
        self.sp_wpm.setSuffix(" WPM")
        g2.addWidget(self.sp_wpm, 1, 1)
        self.lbl_lpm = QLabel()
        self.lbl_lpm.setObjectName("subtle")
        g2.addWidget(self.lbl_lpm, 2, 1)
        g2.addWidget(QLabel("Tone offset"), 3, 0)
        self.sp_tone = QDoubleSpinBox()
        self.sp_tone.setRange(-768000, 768000)
        self.sp_tone.setSuffix(" Hz")
        g2.addWidget(self.sp_tone, 3, 1)
        g2.addWidget(QLabel("Edge shaping"), 4, 0)
        self.sp_ramp = QDoubleSpinBox()
        self.sp_ramp.setRange(0, 50)
        self.sp_ramp.setValue(5.0)
        self.sp_ramp.setSuffix(" ms")
        g2.addWidget(self.sp_ramp, 4, 1)
        left.addWidget(gb2)

        gb3 = QGroupBox("Export Info")
        gi = QVBoxLayout(gb3)
        self.txt_info = QPlainTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMaximumHeight(140)
        gi.addWidget(self.txt_info)
        left.addWidget(gb3)

        gb4 = QGroupBox("Minimum Demod Bandwidth")
        g4 = QVBoxLayout(gb4)
        self.lbl_bw = QLabel("\u2014")
        self.lbl_bw.setWordWrap(True)
        g4.addWidget(self.lbl_bw)
        left.addWidget(gb4)

        self.progress = QProgressBar()
        left.addWidget(self.progress)
        self.btn_export = QPushButton("\u25B6  Export IQ File\u2026")
        self.btn_export.setObjectName("primary")
        self.btn_export.setEnabled(False)
        left.addWidget(self.btn_export)
        left.addStretch(1)

        gp = QGroupBox("Preview")
        pv = QVBoxLayout(gp)
        self.lbl_preview = QLabel()
        self.lbl_preview.setObjectName("preview")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(280, 280)
        pv.addWidget(self.lbl_preview)
        self.lbl_dims = QLabel("No image loaded.")
        self.lbl_dims.setObjectName("subtle")
        self.lbl_dims.setAlignment(Qt.AlignCenter)
        pv.addWidget(self.lbl_dims)
        right.addWidget(gp)
        right.addStretch(1)

        self.btn_open.clicked.connect(self.open_image)
        self.cmb_res.currentIndexChanged.connect(self.render_preview)
        self.ed_call.textChanged.connect(self.on_call_changed)
        for w in (self.rb_wav, self.rb_rf64):
            w.toggled.connect(self.update_plan)
        for w in (self.sp_wpm, self.sp_tone, self.sp_ramp):
            w.valueChanged.connect(self.update_plan)
        self.btn_export.clicked.connect(self.export)
        self.update_plan()

    def on_call_changed(self, text):
        clean = "".join(ch for ch in text.upper() if ch.isalnum())[:16]
        if clean != text:
            cur = self.ed_call.cursorPosition()
            self.ed_call.blockSignals(True)
            self.ed_call.setText(clean)
            self.ed_call.setCursorPosition(min(cur, len(clean)))
            self.ed_call.blockSignals(False)
        self.update_plan()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)")
        if not path:
            return
        img = QImage(path)
        if img.isNull():
            QMessageBox.warning(self, APP_TITLE, "Unable to read the selected image.")
            return
        self.source_image = img
        self.lbl_file.setText(os.path.basename(path))
        self.render_preview()

    def render_preview(self):
        if self.source_image is None:
            return
        n = self.cmb_res.currentData()
        self.pixels = qimage_to_gray_array(self.source_image, n)
        self.lbl_preview.setPixmap(pixels_to_qpixmap(self.pixels))
        self.lbl_dims.setText(f"{n} x {n} ({n * n:,} px)")
        self.update_plan()

    def current_text(self):
        if self.pixels is None:
            return ""
        return core.pixels_to_text(self.pixels, self.cmb_res.currentData(),
                                   self.ed_call.text().strip() or "N0CALL", self.sp_wpm.value())

    def update_plan(self):
        wpm = self.sp_wpm.value()
        self.lbl_lpm.setText(f"= {wpm * 5:,.0f} LPM (letters per minute)")
        fmt = "rf64" if self.rb_rf64.isChecked() else "wav"
        # bandwidth does not depend on the image, so show it even before one is loaded
        self.lbl_bw.setText(bandwidth_text(core.min_bandwidth_hz(wpm, self.sp_ramp.value()),
                                           f"\nTone is at {self.sp_tone.value():+,.0f} Hz from the file's centre."))
        plan = self.plan = core.export_plan(self.current_text(), wpm, fmt, self.sp_ramp.value())
        lines = []
        if plan.get("ok"):
            lines.append(f"Characters: {plan['chars']:,}    Sample rate: {plan['fs']:,} Hz")
            lines.append(f"Sync marker: {plan['marker_len']} x '{core.FRAME_CHAR}'    "
                         f"Lead-in/out silence: {plan['pad'] / plan['fs'] * 1000:,.0f} ms")
            lines.append(f"Duration: {fmt_dur(plan['duration'])}    File size: {fmt_bytes(plan['bytes'])}")
        lines += ["\u26A0 " + w for w in plan.get("warn", [])] + ["\u2717 " + e for e in plan.get("err", [])]
        self.txt_info.setPlainText("\n".join(lines))
        self.btn_export.setEnabled(bool(plan.get("ok")) and self.worker is None)

    def export(self):
        if not self.plan or not self.plan.get("ok"):
            return
        ext = "rf64" if self.rb_rf64.isChecked() else "wav"
        path, _ = QFileDialog.getSaveFileName(self, "Save IQ File", f"bitmap_burst.{ext}", "WAV / RF64 (*.wav *.rf64)")
        if not path:
            return
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        self.worker = ExportWorker(path, self.plan, self.sp_tone.value())
        self.worker.progress.connect(self.progress.setValue)
        self.worker.done.connect(self._export_done)
        self.worker.failed.connect(self._export_failed)
        self.mw.set_status("Exporting\u2026")
        self.worker.start()

    def _export_done(self, path):
        self.worker = None
        self.progress.setValue(100)
        self.mw.set_status(f"Saved {path}")
        self.update_plan()

    def _export_failed(self, msg):
        self.worker = None
        self.mw.set_status("Export failed.")
        self.update_plan()
        QMessageBox.warning(self, APP_TITLE, f"Export failed:\n{msg}")


# --------------------------------------------------------------------------- #
#  Decode tab
# --------------------------------------------------------------------------- #
class DecodeTab(QWidget):
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        self.file_path = None
        self.worker = None
        self.mic = MicCapture()
        self.mic.errorOccurred.connect(self._mic_error)
        self.mic_timer = QTimer(self)
        self.mic_timer.setInterval(1400)
        self.mic_timer.timeout.connect(self._mic_tick)
        self.mic_running = False
        self.last_pixels = None
        self._mic_devices_loaded = False
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        left, right = QVBoxLayout(), QVBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 2)

        gb = QGroupBox("Source")
        g = QGridLayout(gb)
        self.rb_file = QRadioButton("IQ file (WAV / RF64, stereo, any bit depth up to 32-bit float)")
        self.rb_mic = QRadioButton("Microphone / audio input (live, e.g. SDR software in raw binaural IQ mode)")
        self.rb_file.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_file)
        grp.addButton(self.rb_mic)
        g.addWidget(self.rb_file, 0, 0, 1, 2)
        g.addWidget(self.rb_mic, 1, 0, 1, 2)
        self.btn_browse = QPushButton("Browse\u2026")
        self.lbl_path = ElidedLabel("No file selected.")
        self.lbl_path.setObjectName("subtle")
        g.addWidget(self.btn_browse, 2, 0)
        g.addWidget(self.lbl_path, 2, 1)
        g.setColumnStretch(1, 1)
        g.addWidget(QLabel("Audio input"), 3, 0)
        self.cmb_mic = QComboBox()
        g.addWidget(self.cmb_mic, 3, 1)
        hb = QHBoxLayout()
        self.btn_decode = QPushButton("\u25B6  Decode File")
        self.btn_decode.setObjectName("primary")
        self.btn_mic_start = QPushButton("\u25CF  Start Listening")
        self.btn_mic_start.setObjectName("primary")
        self.btn_mic_clear = QPushButton("Clear Buffer")
        for b in (self.btn_decode, self.btn_mic_start, self.btn_mic_clear):
            hb.addWidget(b)
        g.addLayout(hb, 4, 0, 1, 2)
        left.addWidget(gb)

        go = QGroupBox("Options")
        gl = QGridLayout(go)
        gl.addWidget(QLabel("Resolution"), 0, 0)
        self.cmb_res = QComboBox()
        self.cmb_res.addItem("Auto (from header metadata)", None)
        for r in core.RESOLUTIONS:
            self.cmb_res.addItem(f"{r} x {r}", r)
        gl.addWidget(self.cmb_res, 0, 1)
        left.addWidget(go)

        gs = QGroupBox("Status")
        sv = QVBoxLayout(gs)
        self.lbl_stage = ElidedLabel("Waiting.")
        sv.addWidget(self.lbl_stage)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        sv.addWidget(self.progress)
        self.spectrum = SpectrumWidget(dark=self.mw.dark_mode)
        sv.addWidget(self.spectrum)
        left.addWidget(gs)

        gh = QGroupBox("Header")
        hl = QGridLayout(gh)
        hl.addWidget(QLabel("Callsign:"), 0, 0)
        self.lbl_call = ElidedLabel("\u2014")
        f = self.lbl_call.font()
        f.setPointSize(18)
        f.setBold(True)
        self.lbl_call.setFont(f)
        hl.addWidget(self.lbl_call, 0, 1)
        hl.addWidget(QLabel("Speed:"), 1, 0)
        self.lbl_wpm = ElidedLabel("\u2014")
        hl.addWidget(self.lbl_wpm, 1, 1)
        hl.addWidget(QLabel("Resolution:"), 2, 0)
        self.lbl_res = ElidedLabel("\u2014")
        hl.addWidget(self.lbl_res, 2, 1)
        hl.addWidget(QLabel("Needed bandwidth:"), 3, 0)
        self.lbl_bw = ElidedLabel("\u2014")
        hl.addWidget(self.lbl_bw, 3, 1)
        hl.setColumnStretch(1, 1)
        left.addWidget(gh)
        left.addStretch(1)

        gp = QGroupBox("Decoded Image")
        pv = QVBoxLayout(gp)
        self.lbl_preview = QLabel()
        self.lbl_preview.setObjectName("preview")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(280, 280)
        pv.addWidget(self.lbl_preview)
        self.btn_save_png = QPushButton("Save PNG\u2026")
        self.btn_save_png.setEnabled(False)
        pv.addWidget(self.btn_save_png)
        right.addWidget(gp)
        gr = QGroupBox("Report")
        rv = QVBoxLayout(gr)
        self.txt_report = QPlainTextEdit()
        self.txt_report.setReadOnly(True)
        rv.addWidget(self.txt_report)
        right.addWidget(gr)

        self.rb_file.toggled.connect(self._source_changed)
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_decode.clicked.connect(self.decode_file)
        self.btn_mic_start.clicked.connect(self.toggle_mic)
        self.btn_mic_clear.clicked.connect(self.clear_mic_buffer)
        self.btn_save_png.clicked.connect(self.save_png)
        self._source_changed()

    def set_theme(self, dark):
        self.spectrum.set_theme(dark)

    def _source_changed(self):
        is_file = self.rb_file.isChecked()
        self.btn_browse.setEnabled(is_file)
        self.lbl_path.setEnabled(is_file)
        self.btn_decode.setEnabled(is_file and self.file_path is not None and self.worker is None)
        self.cmb_mic.setEnabled(not is_file)
        self.btn_mic_start.setEnabled(not is_file)
        self.btn_mic_clear.setEnabled(not is_file)
        if is_file and self.mic_running:
            self.toggle_mic()
        if not is_file and not self._mic_devices_loaded:
            self.refresh_mic_devices()

    def refresh_mic_devices(self):
        """Enumerated lazily, so a missing audio backend can't affect the Encode tab or file decoding."""
        try:
            devices = self.mic.devices()
        except Exception as e:
            self.lbl_stage.setText(f"Could not list audio inputs: {e}")
            return
        self.cmb_mic.clear()
        for d in devices:
            self.cmb_mic.addItem(d.description(), d)
        self._mic_devices_loaded = True
        if not devices:
            self.lbl_stage.setText("No audio input devices were found.")

    # -- file decode --
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open IQ File", "", "WAV / RF64 (*.wav *.rf64 *.w64)")
        if path:
            self.file_path = path
            self.lbl_path.setText(os.path.basename(path))
            self.btn_decode.setEnabled(self.worker is None)

    def decode_file(self):
        if not self.file_path or self.worker is not None:
            return
        path = self.file_path
        try:
            probe = core.WavIQ(path)
            probe.close()
        except Exception as e:
            QMessageBox.warning(self, APP_TITLE, f"Could not open file:\n{e}")
            return
        self._reset_result_panels()
        self.btn_decode.setEnabled(False)
        self.progress.setVisible(True)
        self.worker = DecodeWorker(lambda: core.WavIQ(path), forced_n=self.cmb_res.currentData())
        self._wire(self.worker)
        self.worker.done.connect(self._decode_done)
        self.worker.failed.connect(self._decode_failed)
        self.mw.set_status("Decoding\u2026")
        self.worker.start()

    def _wire(self, w):
        w.stage.connect(self.lbl_stage.setText)
        w.spectrum.connect(lambda fr, P, f0: self.spectrum.set_data(fr, P, f0, self.spectrum.bw))
        w.header.connect(self._on_header)

    def _on_header(self, info):
        self.lbl_call.setText((info.get("callsign") or "\u2014")[: core.MAX_CALL])
        wpm = info["wpm"]
        self.lbl_wpm.setText(f"{wpm:,.0f} WPM ({wpm * 5:,.0f} LPM), tone {info['f0']:+,.0f} Hz, fs {info['fs']:,} Hz")
        self.lbl_res.setText(f"{info['res']} x {info['res']}" if info.get("res") else "unknown (will infer)")
        bw = core.min_bandwidth_hz(wpm)
        self.lbl_bw.setText(f"min ~{fmt_hz(bw['min_hz'])}, recommended ~{fmt_hz(bw['comfortable_hz'])} "
                            f"(centred on the tone)")
        self.spectrum.set_bandwidth(bw["comfortable_hz"])

    def _show_result(self, res):
        self.last_pixels = res["pixels"]
        self.lbl_preview.setPixmap(pixels_to_qpixmap(res["pixels"]))
        self.btn_save_png.setEnabled(True)
        self._on_header(dict(res, res=res["n"]))
        self.txt_report.setPlainText("\n".join(res.get("report", [])))

    def _decode_done(self, res):
        self.worker = None
        self.progress.setVisible(False)
        self._source_changed()
        self.lbl_stage.setText("Done.")
        self._show_result(res)
        self.mw.set_status(f"Decoded {res['n']} x {res['n']} from {res.get('callsign')}.")

    def _decode_failed(self, msg):
        self.worker = None
        self.progress.setVisible(False)
        self._source_changed()
        self.lbl_stage.setText("Failed.")
        self.mw.set_status("Decode failed.")
        QMessageBox.warning(self, APP_TITLE, msg)

    def _reset_result_panels(self):
        for lb in (self.lbl_call, self.lbl_wpm, self.lbl_res, self.lbl_bw):
            lb.setText("\u2014")
        self.txt_report.clear()
        self.lbl_preview.clear()
        self.btn_save_png.setEnabled(False)
        self.last_pixels = None
        self.spectrum.bw = None

    def save_png(self):
        if self.last_pixels is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save PNG", "decoded.png", "PNG (*.png)")
        if path:
            n = self.last_pixels.shape[0]
            QImage(np.ascontiguousarray(self.last_pixels).tobytes(), n, n, n, QImage.Format_Grayscale8).save(path, "PNG")
            self.mw.set_status(f"Saved {path}")

    # -- live microphone / audio input --
    def toggle_mic(self):
        if self.mic_running:
            self.mic.stop()
            self.mic_timer.stop()
            self.mic_running = False
            self.btn_mic_start.setText("\u25CF  Start Listening")
            self.lbl_stage.setText("Stopped.")
            return
        if not self._mic_devices_loaded:
            self.refresh_mic_devices()
        if self.cmb_mic.count() == 0:
            QMessageBox.warning(self, APP_TITLE, "No audio input devices were found.")
            return
        self._reset_result_panels()
        if not self.mic.start(self.cmb_mic.currentData()):
            return
        self.mic_running = True
        self.btn_mic_start.setText("\u25A0  Stop Listening")
        self.lbl_stage.setText("Listening\u2026 waiting for the header.")
        self.mic_timer.start()

    def clear_mic_buffer(self):
        self.mic.clear()
        self._reset_result_panels()
        self.lbl_stage.setText("Listening\u2026 buffer cleared." if self.mic_running else "Buffer cleared.")

    def _mic_error(self, msg):
        self.mic_running = False
        self.mic_timer.stop()
        self.btn_mic_start.setText("\u25CF  Start Listening")
        QMessageBox.warning(self, APP_TITLE, msg)

    def _mic_tick(self):
        if not self.mic_running or self.worker is not None or self.mic.seconds < 1.0:
            return
        snap = self.mic.snapshot()
        self.worker = DecodeWorker(lambda s=snap: s, forced_n=self.cmb_res.currentData(),
                                   allow_partial=True, close_after=False)
        self._wire(self.worker)
        self.worker.done.connect(self._mic_done)
        self.worker.failed.connect(self._mic_failed)
        self.worker.start()

    def _mic_done(self, res):
        self.worker = None
        self._show_result(res)
        if res.get("complete"):
            self.lbl_stage.setText(f"Picture complete from {res.get('callsign')}. Listening continues; "
                                   f"use Clear Buffer for the next picture.")
        else:
            self.lbl_stage.setText(f"Receiving from {res.get('callsign')}\u2026 picture filling in.")

    def _mic_failed(self, msg):
        self.worker = None
        if self.mic_running:
            self.lbl_stage.setText(f"Listening\u2026 ({self.mic.seconds:.0f} s buffered, no header yet)")


# --------------------------------------------------------------------------- #
#  Main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dark_mode = True
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 820)
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        tb.addWidget(QLabel(f"  <b>{APP_TITLE}</b>  "))
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.act_theme = QAction("", self)
        self.act_theme.triggered.connect(self.toggle_theme)
        tb.addAction(self.act_theme)
        self.tabs = QTabWidget()
        self.encode_tab = EncodeTab(self)
        self.decode_tab = DecodeTab(self)
        self.tabs.addTab(self.encode_tab, "Encode")
        self.tabs.addTab(self.decode_tab, "Decode")
        self.setCentralWidget(self.tabs)
        self.setStatusBar(QStatusBar())
        self.set_status("Ready.")
        self.apply_theme()

    def set_status(self, text):
        self.statusBar().showMessage(text, 8000)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme()

    def apply_theme(self):
        self.setStyleSheet(build_qss(self.dark_mode))
        self.act_theme.setText("\u2600  Light Mode" if self.dark_mode else "\u263D  Dark Mode")
        self.decode_tab.set_theme(self.dark_mode)

    def closeEvent(self, ev):
        try:
            self.decode_tab.mic.stop()
        except Exception:
            pass
        super().closeEvent(ev)


FFMPEG_URL = "https://ffmpeg.org/download.html"


def app_dir() -> str:
    return os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))


def resource_path(name: str) -> str:
    return os.path.join(getattr(sys, "_MEIPASS", app_dir()), name)


def find_ffmpeg():
    """ffmpeg on PATH, or an ffmpeg(.exe) sitting next to the program / in the working directory."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for d in (app_dir(), os.getcwd()):
        for nm in ("ffmpeg.exe", "ffmpeg"):
            p = os.path.join(d, nm)
            if os.path.isfile(p):
                return p
    return None


def show_ffmpeg_missing() -> None:
    box = QMessageBox()
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("FFmpeg not found")
    box.setTextFormat(Qt.RichText)
    box.setTextInteractionFlags(Qt.TextBrowserInteraction)
    box.setText(f"<b>FFmpeg was not found on this computer.</b><br><br>"
                f"{APP_TITLE} needs FFmpeg. Download it from:<br>"
                f'<a href="{FFMPEG_URL}">{FFMPEG_URL}</a><br><br>'
                f"Install it (or put ffmpeg.exe next to this program, or add it to your PATH), then start the "
                f"program again.<br>The program will close when you close this window.")
    btn_open = box.addButton("Open Download Page", QMessageBox.ActionRole)
    box.addButton("Close", QMessageBox.RejectRole)
    box.exec()
    if box.clickedButton() is btn_open:
        QDesktopServices.openUrl(QUrl(FFMPEG_URL))


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setApplicationName(APP_TITLE)
    icon = resource_path("icon.png")
    if os.path.isfile(icon):
        app.setWindowIcon(QIcon(icon))
    if find_ffmpeg() is None:
        show_ffmpeg_missing()
        sys.exit(0)                                   # popup closed -> the app closes
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
