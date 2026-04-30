"""
============================================================
 BN304 Final Assignment — Script 1 of 6
 LTE Patch Antenna — S11 Return Loss Analysis
 Antenna: Microstrip Patch at 2.6 GHz on FR4
 Run:     python lte_s11.py
 Output:  lte_s11.png  (saved in same folder)
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_s11(freq, f0, R_in, Q, Z0=50.0):
    """
    Compute S11 using a parallel RLC resonator model.

    Parameters
    ----------
    freq : ndarray  Frequency array in Hz
    f0   : float    Resonant frequency in Hz
    R_in : float    Input resistance at resonance in Ohm
    Q    : float    Quality factor (dimensionless)
    Z0   : float    Reference impedance in Ohm (default 50)

    Returns
    -------
    S11_dB : ndarray  S11 values in dB
    """
    u = Q * (freq / f0 - f0 / freq)
    Zin = R_in / (1.0 + 1j * u)
    Gamma = (Zin - Z0) / (Zin + Z0)
    S11_dB = 20.0 * np.log10(np.abs(Gamma) + 1e-15)
    return S11_dB


# ── Frequency sweep 2.0 – 3.2 GHz ──────────────────────────────────────────
freq = np.linspace(2.0e9, 3.2e9, 2000)

# LTE patch: R_in = 52 Ω (slight mismatch), Q = 22 (typical FR4 at 2.6 GHz)
f0_lte = 2.6e9
s11_lte = compute_s11(freq, f0_lte, R_in=52.0, Q=22.0)

# ── Print key metrics ───────────────────────────────────────────────────────
mask = s11_lte < -10.0
if np.any(mask):
    f_below = freq[mask]
    bw_hz   = f_below[-1] - f_below[0]
    print("=" * 50)
    print("  LTE Patch Antenna — S11 Results")
    print("=" * 50)
    print(f"  Resonant frequency    : {f0_lte / 1e9:.3f} GHz")
    print(f"  S11 minimum           : {np.min(s11_lte):.1f} dB")
    print(f"  -10 dB lower edge     : {f_below[0] / 1e9:.4f} GHz")
    print(f"  -10 dB upper edge     : {f_below[-1] / 1e9:.4f} GHz")
    print(f"  -10 dB bandwidth      : {bw_hz / 1e6:.1f} MHz")
    print(f"  Fractional bandwidth  : {100 * bw_hz / f0_lte:.2f} %")
    print("=" * 50)
else:
    print("WARNING: S11 never drops below -10 dB. Check parameters.")

# ── LTE Band 7 downlink region ──────────────────────────────────────────────
f_dl_low  = 2.620e9
f_dl_high = 2.690e9

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(freq / 1e9, s11_lte, color='steelblue', linewidth=2.0,
        label='LTE Patch — 2.6 GHz (FR4, h = 1.6 mm)')
ax.axhline(y=-10.0, color='red', linestyle='--', linewidth=1.5,
           label='-10 dB threshold  (VSWR < 2)')
ax.axvspan(f_dl_low / 1e9, f_dl_high / 1e9, alpha=0.12, color='green',
           label='LTE Band 7 Downlink (2620 – 2690 MHz)')
ax.axvline(x=f0_lte / 1e9, color='gray', linestyle=':', linewidth=1.2,
           label=f'Resonance: {f0_lte / 1e9:.2f} GHz')

ax.annotate(f'Min S11 = {np.min(s11_lte):.1f} dB',
            xy=(f0_lte / 1e9, np.min(s11_lte)),
            xytext=(f0_lte / 1e9 + 0.15, np.min(s11_lte) + 6),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10)

ax.set_xlabel('Frequency (GHz)', fontsize=12)
ax.set_ylabel('S11 (dB)', fontsize=12)
ax.set_title('Return Loss (S11) — LTE Indoor Patch Antenna at 2.6 GHz',
             fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim([2.0, 3.2])
ax.set_ylim([-42, 5])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lte_s11.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: lte_s11.png")
