"""
============================================================
 BN304 Final Assignment — Script 2 of 6
 LTE Patch Antenna — Radiation Pattern (E-plane & H-plane)
 Based on the two-radiating-slot model (Balanis, Ch. 14)
 Run:     python radiation_pattern.py
 Output:  radiation_pattern.png
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def patch_eplane(theta, k0, L):
    """
    E-plane pattern:
    Array factor of two radiating slots separated by L,
    multiplied by the obliquity factor cos(theta).
    """
    X  = k0 * L / 2.0 * np.sin(theta)
    af = np.where(np.abs(X) < 1e-10, 1.0, np.cos(X))
    return np.abs(np.cos(theta) * af)


def patch_hplane(theta, k0, W):
    """
    H-plane pattern:
    Single slot sinc factor (k0*W direction) times cos(theta).
    """
    X        = k0 * W / 2.0 * np.sin(theta)
    sinc_val = np.where(np.abs(X) < 1e-10, 1.0, np.sin(X) / X)
    return np.abs(np.cos(theta) * sinc_val)


def hpbw_deg(pattern_norm, theta_rad):
    """Return half-power beamwidth in degrees."""
    idx = np.where(pattern_norm >= 0.707)[0]
    if len(idx) > 1:
        return np.degrees(theta_rad[idx[-1]] - theta_rad[idx[0]])
    return float('nan')


# ── Antenna parameters ───────────────────────────────────────────────────────
f0         = 2.6e9
c          = 3.0e8
lam        = c / f0          # wavelength = 0.1154 m
k0         = 2.0 * np.pi / lam
W          = 35.1e-3         # patch width  (m)
L          = 27.1e-3         # patch length (m)
G_max_dBi = 6.5              # peak gain from CST simulation (dBi)

# ── Pattern calculation ──────────────────────────────────────────────────────
theta = np.linspace(-np.pi / 2, np.pi / 2, 2000)

ep = patch_eplane(theta, k0, L)
hp = patch_hplane(theta, k0, W)

ep_norm = ep / np.max(ep)
hp_norm = hp / np.max(hp)

ep_dBi = 20.0 * np.log10(ep_norm + 1e-6) + G_max_dBi
hp_dBi = 20.0 * np.log10(hp_norm + 1e-6) + G_max_dBi

# ── Print metrics ────────────────────────────────────────────────────────────
hpbw_e = hpbw_deg(ep_norm, theta)
hpbw_h = hpbw_deg(hp_norm, theta)
print("=" * 42)
print("  Radiation Pattern Results")
print("=" * 42)
print(f"  E-plane HPBW : {hpbw_e:.1f} degrees")
print(f"  H-plane HPBW : {hpbw_h:.1f} degrees")
print(f"  Peak gain    : {G_max_dBi} dBi")
print("=" * 42)

# ── Full 360-degree patterns for polar plot ──────────────────────────────────
theta_full = np.linspace(-np.pi, np.pi, 3600)

ep_full = patch_eplane(theta_full, k0, L)
hp_full = patch_hplane(theta_full, k0, W)
ep_full = ep_full / np.max(ep_full)
hp_full = hp_full / np.max(hp_full)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 6))

# Left panel: Cartesian
ax1 = fig.add_subplot(1, 2, 1)
theta_deg = np.degrees(theta)
ax1.plot(theta_deg, ep_dBi, 'b-',  linewidth=2.0,
         label=f'E-plane  (HPBW = {hpbw_e:.0f}°)')
ax1.plot(theta_deg, hp_dBi, 'r--', linewidth=2.0,
         label=f'H-plane  (HPBW = {hpbw_h:.0f}°)')
ax1.axhline(y=G_max_dBi - 3, color='gray', linestyle=':', linewidth=1.2,
            label=f'Half-power level ({G_max_dBi - 3:.1f} dBi)')
ax1.set_xlabel('Elevation Angle θ (degrees)', fontsize=11)
ax1.set_ylabel('Gain (dBi)', fontsize=11)
ax1.set_title('Radiation Pattern — Cartesian', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_xlim([-90, 90])
ax1.set_ylim([-20, 10])
ax1.grid(True, alpha=0.3)

# Right panel: Polar
ax2 = fig.add_subplot(1, 2, 2, projection='polar')
ax2.plot(theta_full, ep_full, 'b-',  linewidth=1.8, label='E-plane')
ax2.plot(theta_full, hp_full, 'r--', linewidth=1.8, label='H-plane')
ax2.set_title('Radiation Pattern — Polar\n(LTE Patch 2.6 GHz)',
              fontsize=11, pad=15)
ax2.legend(loc='lower right', fontsize=9)
ax2.set_theta_zero_location('N')
ax2.set_theta_direction(-1)

plt.tight_layout()
plt.savefig('radiation_pattern.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: radiation_pattern.png")
