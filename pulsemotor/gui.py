"""
Phase 3 GUI: live rotor visualization + dashboard (PySide6 / Qt6).

Architecture:
  * SimWorker     -- runs the drive-cycle physics in a QThread, paced so the
                     spin-up is visible; publishes a latest SimFrame the UI polls.
  * RotorCanvas   -- QPainter-drawn wheel: magnets at the correct angular spacing,
                     stator coil across the air gap with an energised glow, hub,
                     live RPM readout, rotating at the simulated omega.
  * GaugePanel    -- numeric dashboard (RPM, I_pk, kickback V, L, per-event
                     trigger/spike energies, powers, pulse rate).
  * RollingGraph  -- lightweight QPainter sparkline (RPM, per-event spike energy).
  * ControlPanel  -- source voltage / dwell / turns / load sliders + pause/reset;
                     live-reconfigures the motor (no thread restart).

Runs on a real display via `python run_gui.py`, or headless under the offscreen
platform for screenshot verification.
"""
from __future__ import annotations

import math
import sys
import time
from collections import deque
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal, QPointF, QElapsedTimer
from PySide6.QtGui import (QPainter, QColor, QPen, QBrush, QFont, QRadialGradient,
                           QPolygonF)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                               QVBoxLayout, QGridLayout, QLabel, QSlider, QPushButton,
                               QGroupBox, QSizePolicy, QFrame)

from .coil import Coil
from .rotor import Rotor
from .circuit import Circuit
from .sim import Motor

# reference parameterisation (self-starting, motoring region)
REF = dict(source_voltage=12.0, dwell=0.30, turns=900, air_gap=0.002,
           fire_phase=0.65, load_torque=0.0, drag_coeff=6e-7, friction=3e-5,
           coulomb=1e-4, rotor_mass=0.030)


def build_motor(**over) -> Motor:
    p = {**REF, **over}
    coil = Coil(turns=int(p["turns"]), awg=23, wire_material="copper_enamelled",
                winding_length=0.060, core_material="mild_steel",
                core_diameter=0.010, core_length=0.100, core_closed=False)
    rotor = Rotor(wheel_diameter=0.150, magnet_radius=0.070, rotor_mass=p["rotor_mass"],
                  n_magnets=8, magnet_remanence=1.32, magnet_length=0.012,
                  magnet_area=1.0e-4, magnet_mass=0.020, air_gap=p["air_gap"],
                  inductance_swing=0.35, fire_phase=p["fire_phase"],
                  friction_coeff=p["friction"], drag_coeff=p["drag_coeff"],
                  coulomb_friction=p["coulomb"])
    circuit = Circuit(source_voltage=p["source_voltage"], dwell_fraction=p["dwell"],
                      vce_sat=0.20, turn_off_time=50e-9, charge_battery_voltage=12.0,
                      diode_vf=0.70, c_parasitic=25e-12)
    return Motor(coil, rotor, circuit, load_torque=p["load_torque"])


# --------------------------------------------------------------------------- worker
@dataclass
class SimFrame:
    t: float = 0.0
    rpm: float = 0.0
    omega: float = 0.0
    theta: float = 0.0
    i_peak: float = 0.0
    v_kick: float = 0.0
    l_mH: float = 0.0
    e_in_mJ: float = 0.0
    e_kick_mJ: float = 0.0
    e_rec_mJ: float = 0.0
    p_in_W: float = 0.0
    p_kick_W: float = 0.0
    rate_Hz: float = 0.0
    energized: bool = False
    settled: bool = False


class SimWorker(QObject):
    """Period-by-period drive-cycle loop in a worker thread."""

    def __init__(self, motor: Motor, initial_omega: float = 8.0,
                 time_scale: float = 1.0):
        super().__init__()
        self.motor = motor
        self.initial_omega = initial_omega
        self.time_scale = time_scale          # sim-seconds per wall-second
        self._running = False
        self._paused = False
        self.latest = SimFrame(omega=initial_omega)
        self.t = 0.0
        self._omega_prev = initial_omega
        self._settle_run = 0
        self.settled = False
        self._wall_ref = time.perf_counter()
        self._sim_ref = 0.0

    def go(self) -> None:
        self.motor.rotor.omega = self.initial_omega
        self.motor.rotor.theta = 0.0
        self.t = 0.0
        self._wall_ref = time.perf_counter()
        self._sim_ref = 0.0
        self._running = True

    def stop(self) -> None:
        self._running = False

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        # re-baseline the clock so resumed sim-time stays continuous
        self._wall_ref = time.perf_counter()
        self._sim_ref = self.t

    def set_time_scale(self, scale: float) -> None:
        self.time_scale = max(scale, 1e-3)
        self._wall_ref = time.perf_counter()
        self._sim_ref = self.t

    def reconfigure(self, motor: Motor, keep_speed: bool = True) -> None:
        prev = self.motor.rotor.omega
        self.motor = motor
        self.motor.rotor.omega = prev if (keep_speed and prev > 0.5) else self.initial_omega
        self.t = 0.0
        self._settle_run = 0
        self.settled = False
        self._wall_ref = time.perf_counter()
        self._sim_ref = 0.0

    def run(self) -> None:
        while self._running:
            if self._paused:
                QThread.msleep(20)
                self._wall_ref = time.perf_counter()
                self._sim_ref = self.t
                continue
            m = self.motor
            coil, rotor, circuit = m.coil, m.rotor, m.circuit
            load = m.load_torque
            omega = rotor.omega
            if omega > 0.5:
                period = rotor.magnet_step / omega
                pulse = circuit.pulse(coil, rotor, rotor.fire_phase * rotor.magnet_step,
                                      omega, on_steps=600, off_steps=300)
                t_avg = pulse.e_mech / rotor.magnet_step          # work -> avg torque
            else:
                # stalled (drive can't sustain motion, e.g. very low voltage):
                # coast down under losses only, no pulse (avoid divide-by-zero)
                period = 1e-3
                pulse = None
                t_avg = 0.0
            n_m = max(1, int(math.ceil(period / 2e-5)))
            dt_m = period / n_m
            for _ in range(n_m):
                rotor.mechanical_step(t_avg, load, dt_m)

            self.t += period
            rate = rotor.n_magnets * rotor.omega / (2.0 * math.pi)
            psi = (rotor.theta % rotor.magnet_step) / rotor.magnet_step
            if pulse is not None:
                self.latest = SimFrame(
                    t=self.t, rpm=rotor.rpm, omega=rotor.omega, theta=rotor.theta,
                    i_peak=pulse.i_peak, v_kick=pulse.v_kickback_didt,
                    l_mH=coil.inductance(pulse.i_peak) * 1e3,
                    e_in_mJ=pulse.e_input * 1e3, e_kick_mJ=pulse.e_kickback_total * 1e3,
                    e_rec_mJ=pulse.e_kickback_recovered * 1e3,
                    p_in_W=pulse.e_input * rate, p_kick_W=pulse.e_kickback_total * rate,
                    rate_Hz=rate, energized=(0.45 < psi < 0.95), settled=self.settled,
                )
            else:
                self.latest = SimFrame(t=self.t, rpm=rotor.rpm, omega=rotor.omega,
                                       theta=rotor.theta, rate_Hz=rate, settled=self.settled)

            if self._omega_prev > 0 and abs(rotor.omega - self._omega_prev) / self._omega_prev < 7e-4:
                self._settle_run += 1
                if self._settle_run > 60:
                    self.settled = True
            else:
                self._settle_run = 0
            self._omega_prev = rotor.omega

            # pace: keep sim-time (self.t) tracking time_scale * wall-elapsed
            target_sim = self._sim_ref + self.time_scale * (time.perf_counter() - self._wall_ref)
            if self.t > target_sim:
                wait = (self.t - target_sim) / self.time_scale
                QThread.msleep(min(int(wait * 1000), 50))


# --------------------------------------------------------------------------- rotor canvas
class RotorCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(440, 440)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.n_magnets = 8
        self.theta = 0.0
        self.energized = False
        self.rpm = 0.0

    def set_state(self, theta: float, n_magnets: int, energized: bool, rpm: float) -> None:
        self.theta = theta % (2.0 * math.pi)
        self.n_magnets = n_magnets
        self.energized = energized
        self.rpm = rpm
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        R = min(w, h) * 0.40                       # wheel radius [px]
        Rmag = R * 0.92                            # magnet-centre radius
        mag_len = R * 0.18
        mag_wid = R * 0.11

        # background
        p.fillRect(self.rect(), QColor(18, 20, 26))

        # stator coil across the air gap (top of wheel)
        gap_top = cy - R - R * 0.16
        coil_energized = self.energized
        core = QPolygonF([QPointF(cx - 10, gap_top), QPointF(cx + 10, gap_top),
                          QPointF(cx + 10, cy - R + 2), QPointF(cx - 10, cy - R + 2)])
        p.setPen(QPen(QColor(120, 120, 130), 1))
        p.setBrush(QBrush(QColor(70, 72, 80)))
        p.drawPolygon(core)
        # windings
        wc = QColor(220, 110, 60) if coil_energized else QColor(150, 95, 55)
        p.setBrush(QBrush(wc))
        p.setPen(Qt.NoPen)
        span = max(cy - R - gap_top - 10, 6.0)
        for k in range(7):
            wy = gap_top + 5 + k * (span / 7)
            p.drawRoundedRect(_rectf(cx - 20, wy - 4, 40, 7), 3, 3)
        if coil_energized:
            glow = QRadialGradient(cx, cy - R * 0.5, R * 0.9)
            glow.setColorAt(0.0, QColor(255, 140, 60, 90))
            glow.setColorAt(1.0, QColor(255, 140, 60, 0))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(cx, cy), R * 1.02, R * 1.02)

        # wheel rim + hub
        p.setPen(QPen(QColor(90, 94, 104), 3))
        p.setBrush(QBrush(QColor(34, 37, 44)))
        p.drawEllipse(QPointF(cx, cy), R, R)
        p.setPen(QPen(QColor(60, 64, 72), 2))
        p.setBrush(QBrush(QColor(50, 54, 62)))
        p.drawEllipse(QPointF(cx, cy), R * 0.28, R * 0.28)

        # magnets
        for i in range(self.n_magnets):
            ang = self.theta + i * 2.0 * math.pi / self.n_magnets - math.pi / 2.0
            mx = cx + Rmag * math.cos(ang)
            my = cy + Rmag * math.sin(ang)
            p.save()
            p.translate(mx, my)
            p.rotate(math.degrees(ang) + 90)
            # north (red) outer, south (blue) inner
            p.setPen(QPen(QColor(20, 20, 20), 1))
            p.setBrush(QBrush(QColor(214, 64, 64)))
            p.drawRoundedRect(_rectf(-mag_wid / 2, -mag_len / 2, mag_wid, mag_len / 2), 3, 3)
            p.setBrush(QBrush(QColor(64, 110, 214)))
            p.drawRoundedRect(_rectf(-mag_wid / 2, 0, mag_wid, mag_len / 2), 3, 3)
            p.restore()

        # RPM readout
        p.setPen(QColor(230, 232, 238))
        p.setFont(QFont("DejaVu Sans", 16, QFont.Bold))
        p.drawText(_rectf(cx - R * 0.28, cy - 14, R * 0.56, 28),
                   Qt.AlignCenter, f"{self.rpm:,.0f} RPM")

        p.end()


def _rectf(x, y, w, h):
    from PySide6.QtCore import QRectF
    return QRectF(x, y, w, h)


# --------------------------------------------------------------------------- gauges
class GaugePanel(QGroupBox):
    def __init__(self):
        super().__init__("Live dashboard")
        self._labels = {}
        grid = QGridLayout()
        rows = [
            ("Sim time", "t", "{:.2f}", "s"),
            ("RPM", "rpm", "{:,.0f}", ""),
            ("Pulse rate", "rate", "{:,.0f}", "Hz"),
            ("Peak current  I_pk", "ipk", "{:.3f}", "A"),
            ("Coil inductance  L", "L", "{:.2f}", "mH"),
            ("Kickback peak V", "vkick", "{:.3g}", "V"),
            ("", None, None, None),
            ("Spike TRIGGER energy / event", "ein", "{:.3f}", "mJ"),
            ("Spike OUTPUT energy / event", "ekick", "{:.3f}", "mJ"),
            ("  -> recovered / event", "erec", "{:.3f}", "mJ"),
            ("", None, None, None),
            ("Input power", "pin", "{:.3f}", "W"),
            ("Spike output power", "pkick", "{:.3f}", "W"),
        ]
        for r, (title, key, fmt, unit) in enumerate(rows):
            if key is None:
                continue
            t = QLabel(title); t.setStyleSheet("color:#aab;")
            v = QLabel("—"); v.setStyleSheet("color:#eee; font-weight:bold;")
            u = QLabel(unit); u.setStyleSheet("color:#889;")
            grid.addWidget(t, r, 0)
            grid.addWidget(v, r, 1, alignment=Qt.AlignRight)
            grid.addWidget(u, r, 2)
            self._labels[key] = (v, fmt)
        self.setLayout(grid)
        self.setStyleSheet("QGroupBox{color:#cde;font-weight:bold;border:1px solid #334;border-radius:6px;margin-top:10px;padding:8px;}")

    def set_values(self, f: SimFrame) -> None:
        vals = {"t": f.t, "rpm": f.rpm, "rate": f.rate_Hz, "ipk": f.i_peak, "L": f.l_mH,
                "vkick": f.v_kick, "ein": f.e_in_mJ, "ekick": f.e_kick_mJ,
                "erec": f.e_rec_mJ, "pin": f.p_in_W, "pkick": f.p_kick_W}
        for key, (lbl, fmt) in self._labels.items():
            lbl.setText(fmt.format(vals[key]))


# --------------------------------------------------------------------------- rolling graph
class RollingGraph(QWidget):
    def __init__(self, title: str, series: list, ylab: str = "", maxlen: int = 240):
        super().__init__()
        self.title = title
        self.ylab = ylab
        self.series = series                       # [(name, color)]
        self.data = {name: deque(maxlen=maxlen) for name, _ in series}
        self.setMinimumHeight(140)

    def append(self, mapping: dict) -> None:
        for name, _ in self.series:
            self.data[name].append(mapping.get(name, 0.0))
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(20, 22, 28))
        p.setPen(QColor(170, 176, 188))
        p.setFont(QFont("DejaVu Sans", 9, QFont.Bold))
        p.drawText(8, 14, self.title)
        margin_l, margin_b, margin_t = 38, 18, 20
        plot_w = w - margin_l - 8
        plot_h = h - margin_b - margin_t
        if plot_w <= 0 or plot_h <= 0:
            p.end(); return
        # determine y-range
        allv = [v for name, _ in self.series for v in self.data[name]]
        if not allv:
            p.end(); return
        ymax = max(allv)
        if ymax <= 0:
            ymax = 1.0
        ymax *= 1.1
        # grid + y labels
        p.setPen(QPen(QColor(48, 52, 60), 1, Qt.DotLine))
        p.setFont(QFont("DejaVu Sans", 7))
        for frac in (0.0, 0.5, 1.0):
            yy = margin_t + plot_h * (1 - frac)
            p.drawLine(margin_l, yy, margin_l + plot_w, yy)
            p.setPen(QColor(110, 116, 128))
            p.drawText(2, yy + 8, f"{ymax*frac:.1g}")
            p.setPen(QPen(QColor(48, 52, 60), 1, Qt.DotLine))
        # lines
        for name, color in self.series:
            d = self.data[name]
            if len(d) < 2:
                continue
            p.setPen(QPen(QColor(*color), 2))
            n = len(d)
            pts = []
            for i, v in enumerate(d):
                x = margin_l + plot_w * i / (n - 1)
                y = margin_t + plot_h * (1 - min(v / ymax, 1.0))
                pts.append(QPointF(x, y))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        # legend
        p.setFont(QFont("DejaVu Sans", 8))
        lx = margin_l + 4
        for name, color in self.series:
            p.setPen(QPen(QColor(*color), 3))
            p.drawLine(lx, margin_t + 6, lx + 14, margin_t + 6)
            p.setPen(QColor(200, 206, 218))
            p.drawText(lx + 18, margin_t + 9, name)
            lx += 18 + 8 * (len(name) + 1) + 24
        p.end()


# --------------------------------------------------------------------------- controls
class ControlPanel(QGroupBox):
    changed = Signal(dict)            # motor-parameter changes (rebuilds the motor)
    speed_changed = Signal(float)     # simulation time-scale

    SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]

    def __init__(self):
        super().__init__("Controls")
        self._state = dict(REF)
        grid = QGridLayout()
        self._sliders = {}
        self._displays = {}

        # source voltage -- logarithmic 1 V .. 1000 V (slider 0..1000 -> 10^(3*pos/1000))
        grid.addWidget(QLabel("Source voltage (log)"), 0, 0)
        vsl = QSlider(Qt.Horizontal); vsl.setMinimum(0); vsl.setMaximum(1000)
        vsl.setValue(int(round(1000 * math.log10(REF["source_voltage"]) / 3.0)))
        vdisp = QLabel()
        vsl.valueChanged.connect(lambda v: self._on_voltage(v, vdisp))
        grid.addWidget(vsl, 0, 1); grid.addWidget(vdisp, 0, 2)
        self._sliders["source_voltage"] = vsl
        self._on_voltage(vsl.value(), vdisp)

        specs = [
            ("dwell", "Dwell fraction", 0.10, 0.45, 0.01, ""),
            ("turns", "Coil turns", 400, 1600, 100, ""),
            ("load_torque", "Load torque", 0.0, 0.02, 0.0005, "N·m"),
        ]
        for r, (key, label, lo, hi, step, unit) in enumerate(specs, start=1):
            grid.addWidget(QLabel(label), r, 0)
            sl = QSlider(Qt.Horizontal)
            nsteps = int(round((hi - lo) / step))
            sl.setMinimum(0); sl.setMaximum(nsteps)
            sl.setValue(int(round((REF[key] - lo) / step)))
            disp = QLabel()
            sl.valueChanged.connect(lambda v, k=key, lo=lo, st=step, d=disp, u=unit:
                                    (d.setText(f"{lo+v*st:g} {u}"), self._on(k, lo + v * st)))
            grid.addWidget(sl, r, 1); grid.addWidget(disp, r, 2)
            self._sliders[key] = sl
            disp.setText(f"{REF[key]:g} {unit}")

        # simulation speed (time-scale)
        grid.addWidget(QLabel("Sim speed"), 4, 0)
        ssl = QSlider(Qt.Horizontal); ssl.setMinimum(0); ssl.setMaximum(len(self.SPEEDS) - 1)
        ssl.setValue(self.SPEEDS.index(1.0))
        sdisp = QLabel("1.0x")
        ssl.valueChanged.connect(lambda v: (sdisp.setText(f"{self.SPEEDS[v]:g}x"),
                                            self.speed_changed.emit(self.SPEEDS[v])))
        grid.addWidget(ssl, 4, 1); grid.addWidget(sdisp, 4, 2)

        self.pause_btn = QPushButton("Pause")
        self.reset_btn = QPushButton("Reset")
        grid.addWidget(self.pause_btn, 5, 0)
        grid.addWidget(self.reset_btn, 5, 1, 1, 2)
        self.setLayout(grid)
        self.setStyleSheet("QGroupBox{color:#cde;font-weight:bold;border:1px solid #334;"
                           "border-radius:6px;margin-top:10px;padding:8px;} QLabel{color:#bbc;}")

    def _on_voltage(self, pos, disp):
        v = 10.0 ** (3.0 * pos / 1000.0)
        self._state["source_voltage"] = v
        disp.setText(f"{v:.2f} V")
        self.changed.emit(dict(self._state))

    def _on(self, key, val):
        self._state[key] = val
        self.changed.emit(dict(self._state))


# --------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Switched-Reluctance Pulse Motor — live simulation")
        self.resize(1180, 760)
        self.setStyleSheet("background-color:#121419;")

        self.worker = SimWorker(build_motor())
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.canvas = RotorCanvas()
        self.gauges = GaugePanel()
        self.graph_rpm = RollingGraph("Rotor speed [RPM]", [("rpm", (90, 170, 250))])
        self.graph_e = RollingGraph("Per-event energy [mJ]",
                                    [("trigger in", (240, 170, 90)),
                                     ("spike out", (110, 210, 130)),
                                     ("recovered", (180, 130, 230))])
        self.controls = ControlPanel()
        self.controls.changed.connect(self._reconfigure)
        self.controls.speed_changed.connect(self.worker.set_time_scale)
        self.controls.pause_btn.clicked.connect(self._toggle_pause)
        self.controls.reset_btn.clicked.connect(self._reset)
        self._paused = False

        # layout
        left = QVBoxLayout()
        left.addWidget(self.canvas, stretch=1)
        left.addWidget(self.controls)
        right = QVBoxLayout()
        right.addWidget(self.gauges)
        right.addWidget(self.graph_rpm, stretch=1)
        right.addWidget(self.graph_e, stretch=1)
        central = QWidget()
        h = QHBoxLayout(central)
        h.addLayout(left, stretch=5)
        v = QVBoxLayout()
        line = QFrame(); line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color:#334;")
        h.addWidget(line)
        h.addLayout(right, stretch=4)
        self.setCentralWidget(central)

        # display timer drives the UI at ~33 Hz
        self._clock = QElapsedTimer(); self._clock.start()
        self._last_ns = self._clock.nsecsElapsed()
        self._vis_theta = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self.worker.go()
        self.thread.start()

    def _tick(self):
        now = self._clock.nsecsElapsed()
        dt = (now - self._last_ns) * 1e-9
        self._last_ns = now
        f = self.worker.latest
        ts = self.worker.time_scale
        if not self._paused:
            # sim advances at time_scale * wall-clock, so the visual rotor does too
            self._vis_theta = (self._vis_theta + f.omega * dt * ts) % (2.0 * math.pi)
            self.gauges.set_values(f)
            self.graph_rpm.append({"rpm": f.rpm})
            self.graph_e.append({"trigger in": f.e_in_mJ, "spike out": f.e_kick_mJ,
                                 "recovered": f.e_rec_mJ})
        self.canvas.set_state(self._vis_theta, self.worker.motor.rotor.n_magnets,
                              f.energized, f.rpm)

    def _reconfigure(self, state):
        self.worker.reconfigure(build_motor(**state), keep_speed=True)

    def _toggle_pause(self):
        self._paused = not self._paused
        self.worker.set_paused(self._paused)
        self.controls.pause_btn.setText("Resume" if self._paused else "Pause")

    def _reset(self):
        m = build_motor(**self.controls._state)
        self.worker.reconfigure(m, keep_speed=False)

    def closeEvent(self, ev):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(2000)
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
