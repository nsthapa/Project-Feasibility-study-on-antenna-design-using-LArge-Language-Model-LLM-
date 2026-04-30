"""
============================================================
 BN304 Final Assignment — Script 3 of 6
 RF Energy Harvesting — Power Conversion Efficiency Analysis
 Antenna: 2.4 GHz Patch + HSMS-2860 Schottky Rectifier
 Run:     python rf_harvesting.py
 Output:  rf_harvesting.png
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def pce_model(p_dBm, pce_max=65.0, k=0.15, p_half=-15.0):
    """
    Logistic (sigmoid) PCE model calibrated to HSMS-2860 at 2.4 GHz.

    Parameters
    ----------
    p_dBm   : ndarray  Input RF power in dBm
    pce_max : float    Saturation efficiency in %  (default 65)
    k       : float    Steepness factor in 1/dBm   (default 0.15)
    p_half  : float    Power where PCE = pce_max/2 (default -15 dBm)

    Returns
    -------
    pce : ndarray  Power conversion efficiency in %
    """
    return pce_max / (1.0 + np.exp(-k * (p_dBm - p_half)))


def output_voltage(p_dBm, pce_percent, R_load=1000.0):
    """
    DC output voltage across the load resistor.

    Parameters
    ----------
    p_dBm       : ndarray  Input RF power in dBm
    pce_percent : ndarray  Efficiency in %
    R_load      : float    Load resistance in Ohm

    Returns
    -------
    V_dc : ndarray  DC output voltage in Volts
    """
    p_watts = 10.0 ** ((p_dBm - 30.0) / 10.0)       # dBm → Watts
    p_dc    = (pce_percent / 100.0) * p_watts
    return np.sqrt(np.maximum(p_dc * R_load, 0.0))


# ── Input power sweep: -35 to +15 dBm ───────────────────────────────────────
p_in = np.linspace(-35.0, 15.0, 500)

pce_1k   = pce_model(p_in)
pce_2k2  = pce_model(p_in, pce_max=60.0, k=0.14, p_half=-18.0)

v_out_1k  = output_voltage(p_in, pce_1k,  R_load=1000.0)
v_out_2k2 = output_voltage(p_in, pce_2k2, R_load=2200.0)

# ── Print summary table ──────────────────────────────────────────────────────
print("=" * 60)
print("  RF Harvesting Summary Table  (R_load = 1 kΩ)")
print("=" * 60)
print(f"{'P_in (dBm)':>12} {'PCE (%)':>10} {'V_dc (mV)':>12} {'P_dc (µW)':>12}")
print("-" * 50)
for p in [-30, -25, -20, -15, -10, -5, 0, 5, 10]:
    pv  = pce_model(np.array([float(p)]))[0]
    vv  = output_voltage(np.array([float(p)]), np.array([pv]), 1000.0)[0]
    pw  = 10.0 ** ((p - 30.0) / 10.0)
    pdu = pv / 100.0 * pw * 1e6
    print(f"{p:>12.0f} {pv:>10.1f} {vv*1000:>12.1f} {pdu:>12.3f}")
print("=" * 60)

# ── Ambient WiFi power range markers ─────────────────────────────────────────
ambient_min = -20.0   # dBm  (~3–5 m from AP)
ambient_max = -10.0   # dBm  (~1–2 m from AP)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: PCE vs input power
ax1.plot(p_in, pce_1k,  'b-',  linewidth=2.0, label='R_load = 1 kΩ')
ax1.plot(p_in, pce_2k2, 'r--', linewidth=2.0, label='R_load = 2.2 kΩ')
ax1.axvspan(ambient_min, ambient_max, alpha=0.15, color='green',
            label='Typical indoor WiFi ambient power')
ax1.set_ylabel('Power Conversion Efficiency (%)', fontsize=11)
ax1.set_title(
    'RF Energy Harvesting — 2.4 GHz Rectenna\n'
    '(HSMS-2860 Schottky Diode  |  FR4 Patch Antenna)',
    fontsize=12)
ax1.legend(fontsize=10)
ax1.set_ylim([0, 75])
ax1.grid(True, alpha=0.3)

# Bottom: Output voltage vs input power
ax2.plot(p_in, v_out_1k  * 1000.0, 'b-',  linewidth=2.0,
         label='R_load = 1 kΩ')
ax2.plot(p_in, v_out_2k2 * 1000.0, 'r--', linewidth=2.0,
         label='R_load = 2.2 kΩ')
ax2.axvspan(ambient_min, ambient_max, alpha=0.15, color='green',
            label='Typical indoor WiFi ambient power')
ax2.axhline(y=300.0, color='orange', linestyle=':', linewidth=1.5,
            label='~300 mV min. for boosted MCU supply')
ax2.set_xlabel('Input RF Power (dBm)', fontsize=11)
ax2.set_ylabel('DC Output Voltage (mV)', fontsize=11)
ax2.legend(fontsize=10)
ax2.set_ylim([0, 4500])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rf_harvesting.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: rf_harvesting.png")
