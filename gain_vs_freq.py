"""
============================================================
 BN304 Final Assignment — Script 6 of 6
 Gain vs Frequency — Comparison of All Three Antenna Designs
 Run:     python gain_vs_freq.py
 Output:  gain_vs_freq.png
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def gain_vs_freq(freq, f0, G_peak, Q, loss_tangent=0.02):
    """
    Approximate gain vs frequency for a resonant antenna.
    Accounts for mismatch loss off-resonance and substrate dielectric loss.

    Parameters
    ----------
    freq         : ndarray  Frequency array in Hz
    f0           : float    Resonant frequency in Hz
    G_peak       : float    Peak gain at resonance in dBi
    Q            : float    Antenna quality factor
    loss_tangent : float    Substrate loss tangent

    Returns
    -------
    gain_dBi : ndarray
    """
    u         = Q * (freq / f0 - f0 / freq)
    Gamma_sq  = u ** 2 / (1.0 + u ** 2)     # fraction of power reflected
    mismatch  = 1.0 - Gamma_sq               # accepted power fraction
    eta_rad   = 1.0 / (1.0 + loss_tangent * Q)
    G_lin     = (10.0 ** (G_peak / 10.0)) * mismatch * eta_rad
    return 10.0 * np.log10(G_lin + 1e-10)


# ── Antenna definitions ──────────────────────────────────────────────────────
antennas = [
    {
        'name':   'LTE Patch (2.6 GHz, FR4)',
        'f0':     2.6e9,
        'G_peak': 6.5,
        'Q':      22.0,
        'tan_d':  0.020,
        'color':  'steelblue',
        'ls':     '-',
        'fmin':   2.0e9,
        'fmax':   3.2e9,
    },
    {
        'name':   'RF Harvesting Patch (2.4 GHz, FR4)',
        'f0':     2.4e9,
        'G_peak': 6.2,
        'Q':      20.0,
        'tan_d':  0.020,
        'color':  'darkorange',
        'ls':     '--',
        'fmin':   2.0e9,
        'fmax':   3.0e9,
    },
    {
        'name':   'Near-field Loop (500 MHz)',
        'f0':     0.5e9,
        'G_peak': 1.8,
        'Q':      18.0,
        'tan_d':  0.005,
        'color':  'seagreen',
        'ls':     '-.',
        'fmin':   0.2e9,
        'fmax':   0.8e9,
    },
]

# ── Print summary ────────────────────────────────────────────────────────────
print("=" * 60)
print("  Gain Summary — All Three Antenna Designs")
print("=" * 60)
print(f"{'Antenna':<44} {'f0 (GHz)':>8} {'Gain (dBi)':>10}")
print("-" * 64)
for a in antennas:
    print(f"  {a['name']:<42} {a['f0']/1e9:>8.2f} {a['G_peak']:>10.1f}")
print("=" * 60)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Patch antennas (2 – 3 GHz)
ax0 = axes[0]
for a in antennas[:2]:
    freq = np.linspace(a['fmin'], a['fmax'], 1000)
    g    = gain_vs_freq(freq, a['f0'], a['G_peak'], a['Q'], a['tan_d'])
    ax0.plot(freq / 1e9, g, color=a['color'], linestyle=a['ls'],
             linewidth=2.0, label=a['name'])

ax0.axhline(y=0, color='gray', linestyle=':', linewidth=1.0,
            label='0 dBi reference')
ax0.set_xlabel('Frequency (GHz)', fontsize=11)
ax0.set_ylabel('Gain (dBi)', fontsize=11)
ax0.set_title('Patch Antenna Gain vs Frequency  (2 – 3 GHz)', fontsize=12)
ax0.legend(fontsize=10)
ax0.set_ylim([-5, 10])
ax0.grid(True, alpha=0.3)

# Right: Near-field loop (200 – 800 MHz)
ax1  = axes[1]
a    = antennas[2]
freq = np.linspace(a['fmin'], a['fmax'], 1000)
g    = gain_vs_freq(freq, a['f0'], a['G_peak'], a['Q'], a['tan_d'])
ax1.plot(freq / 1e6, g, color=a['color'], linestyle=a['ls'],
         linewidth=2.0, label=a['name'])
ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1.0,
            label='0 dBi reference')
ax1.set_xlabel('Frequency (MHz)', fontsize=11)
ax1.set_ylabel('Gain (dBi)', fontsize=11)
ax1.set_title('Loop Antenna Gain vs Frequency  (200 – 800 MHz)', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_ylim([-5, 5])
ax1.grid(True, alpha=0.3)

plt.suptitle('Gain vs Frequency — All Three Antenna Designs',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('gain_vs_freq.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: gain_vs_freq.png")
