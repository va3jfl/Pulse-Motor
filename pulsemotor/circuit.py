"""
Circuit & transient "flyback" (kickback) calculator.

Solves a single drive pulse of the switched topology and reports the full energy
breakdown, with the *total inductive kickback energy* vs the *supply input
energy* as the headline pair.

Physical model (all SI, all faithful -- nothing is capped on the output side):

ON-time (transistor conducting, dwell angle):
    Flux linkage lambda = L(theta) * lambda_base(i), where lambda_base comes from
    the saturating core model.  The exact circuit equation is
        V_s - Vce(sat) = R i + d(lambda)/dt
                       = R i + L_inc(i,theta) di/dt + (d lambda/d theta) omega
    with L_inc = d(lambda)/di (floors at the air-core value, never negative).
    Mechanical work comes from the co-energy:  T = d W'/d theta,  W'(i) = integral
    lambda di.

OFF-time (transistor off):
    Current cannot change instantly; the stored field energy is released.
        * total released energy = W_field(i_pk) = integral_0^{Ipk} s L_inc(s) ds
          (= 1/2 L I^2 exactly for an unsaturated core)
        * peak voltage, three complementary estimates, none capped:
              - V = -L di/dt      with di/dt ~ Ipk / t_off   (forced collapse)
              - LC ring peak      V = Ipk sqrt(L / C_par)    (unclamped)
              - clamped level     V = V_batt + V_f           (recovery path)
        * recovery into the charging battery is integrated along the real path;
          diode and winding-R losses are reported as explicit separate terms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .coil import Coil
from .rotor import Rotor


@dataclass
class Circuit:
    # --- source / drive ---
    source_voltage: float = 12.0          # V_s [V]
    trigger_threshold: float = 0.8        # trigger-winding EMF that fires the switch [V]
    dwell_fraction: float = 0.30          # on-time as a fraction of the magnet period

    # --- switch (bipolar / MOSFET modelled as a saturated drop) ---
    vce_sat: float = 0.20                 # switch on-state voltage drop [V]
    turn_off_time: float = 50.0e-9        # t_off: spike "sharpness" dt [s]

    # --- recovery path ---
    charge_battery_voltage: float = 12.0  # V_batt the clamp/recovery rail [V]
    diode_vf: float = 0.70                # rectifier forward drop [V]

    # --- parasitics (set the unclamped spike height) ---
    c_parasitic: float = 25.0e-12         # winding + switch capacitance [F]

    # --- integration resolution ---
    on_steps: int = 4000
    off_steps: int = 4000

    # ---------------------------------------------------------------- helpers
    def _L_inc(self, coil: Coil, rotor: Rotor, i: float, theta: float) -> float:
        """Differential inductance d(lambda)/di [H] of the coupled coil+rotor."""
        return rotor.inductance(coil.differential_inductance(i), theta)

    def _dlambda_dtheta(self, coil: Coil, rotor: Rotor, i: float, theta: float) -> float:
        """d(lambda)/d theta [Wb-turn/rad] -- the motional (rotational) coupling."""
        return rotor.dL_dtheta(coil.flux_linkage(i), theta)

    def _torque(self, coil: Coil, rotor: Rotor, i: float, theta: float) -> float:
        """Electromagnetic torque [N*m] = d W'/d theta (co-energy derivative)."""
        return rotor.dL_dtheta(coil.coenergy_base(i), theta)

    # ----------------------------------------------------------------- pulse
    def pulse(self, coil: Coil, rotor: Rotor, theta_fire: float,
              omega: float, on_steps: int | None = None,
              off_steps: int | None = None) -> "PulseResult":
        """Solve one drive pulse at firing angle `theta_fire` and speed `omega`."""
        R = coil.resistance_dc
        Vs, Vce = self.source_voltage, self.vce_sat

        period = rotor.magnet_step / omega if omega > 0 else float("inf")
        t_on = self.dwell_fraction * period
        if math.isfinite(period):
            t_on = min(t_on, 0.95 * period)

        n = self.on_steps if on_steps is None else on_steps
        dt = t_on / n
        i = 0.0
        theta = float(theta_fire)
        e_in = e_cu = e_mech = e_sw = 0.0

        def f(i_: float, th_: float):
            linc = self._L_inc(coil, rotor, i_, th_)
            dlam = self._dlambda_dtheta(coil, rotor, i_, th_)
            di_ = (Vs - Vce - R * i_ - omega * dlam) / linc
            return di_, omega

        for _ in range(n):
            i1, th1 = i, theta
            k1i, k1t = f(i1, th1)
            i2, th2 = i + 0.5 * dt * k1i, theta + 0.5 * dt * k1t
            k2i, k2t = f(i2, th2)
            i3, th3 = i + 0.5 * dt * k2i, theta + 0.5 * dt * k2t
            k3i, k3t = f(i3, th3)
            i4, th4 = i + dt * k3i, theta + dt * k3t
            k4i, k4t = f(i4, th4)
            i += (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
            theta += (dt / 6.0) * (k1t + 2.0 * k2t + 2.0 * k3t + k4t)
            i = max(i, 0.0)
            # 4th-order (RK4-stage / Simpson) energy quadrature -> tight closure
            w = dt / 6.0
            e_in += Vs * w * (i1 + 2.0 * i2 + 2.0 * i3 + i4)
            e_cu += R * w * (i1 * i1 + 2.0 * i2 * i2 + 2.0 * i3 * i3 + i4 * i4)
            e_sw += Vce * w * (i1 + 2.0 * i2 + 2.0 * i3 + i4)
            e_mech += w * omega * (
                self._torque(coil, rotor, i1, th1) + 2.0 * self._torque(coil, rotor, i2, th2)
                + 2.0 * self._torque(coil, rotor, i3, th3) + self._torque(coil, rotor, i4, th4))

        i_peak = max(i, 0.0)
        theta_off = theta

        # total inductive kickback energy = field energy at turn-off (no cap)
        e_field = rotor.inductance(coil.field_energy_base(i_peak), theta_off)
        L_pk = rotor.inductance(coil.inductance(i_peak), theta_off)     # chord, for display

        # peak flyback voltage (three estimates, none capped)
        v_didt = L_pk * i_peak / self.turn_off_time if self.turn_off_time > 0 else float("inf")
        v_lc = i_peak * math.sqrt(L_pk / self.c_parasitic) if self.c_parasitic > 0 else float("inf")
        v_clamped = self.charge_battery_voltage + self.diode_vf

        # OFF-time recovery into the charging battery (real path: R + Vf + V_batt)
        e_batt = e_diode = e_r_decay = 0.0
        t_decay = 0.0
        if i_peak > 0:
            Vb, Vf = self.charge_battery_voltage, self.diode_vf
            i_off = i_peak
            n_off = self.off_steps if off_steps is None else off_steps
            dt_off = max((L_pk * i_peak / max(v_clamped, 1e-3)) / n_off, 1e-12)
            steps = 0
            while i_off > 1e-6 and steps < 200000:
                linc = self._L_inc(coil, rotor, i_off, theta_off)
                didt = -(Vb + Vf + i_off * R) / linc
                i_next = i_off + didt * dt_off
                if i_next <= 0:
                    frac = i_off / max(i_off - i_next, 1e-30)
                    seg = frac * dt_off
                    i_mid = 0.5 * i_off
                    e_batt += Vb * i_mid * seg
                    e_diode += Vf * i_mid * seg
                    e_r_decay += i_mid * i_mid * R * seg
                    t_decay += seg
                    break
                i_mid = 0.5 * (i_off + i_next)
                e_batt += Vb * i_mid * dt_off
                e_diode += Vf * i_mid * dt_off
                e_r_decay += i_mid * i_mid * R * dt_off
                t_decay += dt_off
                i_off = i_next
                steps += 1

        return PulseResult(
            i_peak=i_peak, theta_off=theta_off, t_on=t_on, t_decay=t_decay,
            L_at_peak=L_pk,
            e_input=e_in, e_copper=e_cu, e_mech=e_mech, e_switch_loss=e_sw,
            e_kickback_total=e_field,
            e_kickback_recovered=e_batt,
            e_diode_loss=e_diode, e_r_decay_loss=e_r_decay,
            v_kickback_didt=v_didt, v_kickback_lc=v_lc, v_kickback_clamped=v_clamped,
            dt_sharp=self.turn_off_time, pulse_period=period,
        )


@dataclass
class PulseResult:
    # currents / timing
    i_peak: float                 # [A] peak winding current at switch-off
    theta_off: float              # [rad] rotor angle at switch-off
    t_on: float                   # [s] conduction time
    t_decay: float                # [s] flyback current decay time
    L_at_peak: float              # [H] chord inductance at I_peak
    pulse_period: float           # [s] magnet-crossing period at this speed
    dt_sharp: float               # [s] spike sharpness (switch turn-off time)
    # energies (per pulse) [J]
    e_input: float                # drawn from the supply during on-time
    e_copper: float               # I^2 R heat in the winding (on-time)
    e_mech: float                 # mechanical work delivered (on-time)
    e_switch_loss: float          # transistor Vce(sat) drop loss (on-time)
    e_kickback_total: float       # TOTAL inductive kickback energy = field energy at Ipk
    e_kickback_recovered: float   # delivered to the charging battery
    e_diode_loss: float           # diode forward-drop loss during recovery
    e_r_decay_loss: float         # winding-R loss during flyback decay
    # peak flyback voltages [V] -- three complementary estimates, none capped
    v_kickback_didt: float        # V = L di/dt  (forced collapse in t_off)
    v_kickback_lc: float          # LC ring peak (unclamped, parasitic C)
    v_kickback_clamped: float     # clamp level = V_batt + V_f

    @property
    def ratio_total_to_input(self) -> float:
        return self.e_kickback_total / self.e_input if self.e_input > 0 else float("nan")

    @property
    def ratio_recovered_to_input(self) -> float:
        return self.e_kickback_recovered / self.e_input if self.e_input > 0 else float("nan")

    def power_at(self, pulse_rate_hz: float) -> dict:
        """Average powers [W] at a given pulse repetition rate."""
        return {
            "P_input_W": self.e_input * pulse_rate_hz,
            "P_kickback_total_W": self.e_kickback_total * pulse_rate_hz,
            "P_kickback_recovered_W": self.e_kickback_recovered * pulse_rate_hz,
            "P_mech_W": self.e_mech * pulse_rate_hz,
            "P_copper_W": self.e_copper * pulse_rate_hz,
            "P_diode_W": self.e_diode_loss * pulse_rate_hz,
            "P_r_decay_W": self.e_r_decay_loss * pulse_rate_hz,
        }
