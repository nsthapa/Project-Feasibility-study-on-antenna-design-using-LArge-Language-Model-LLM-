"""
============================================================
 BN304 Final Assignment — Script 4 of 6
 Patch Antenna Dimension Optimisation using SciPy
 Substrate: FR4  (εr = 4.4, h = 1.6 mm)
 Target frequency: 2.6 GHz
 Run:     python patch_optimization.py
 Output:  patch_optimization.png
============================================================
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ── Transmission-line model helper functions ─────────────────────────────────

def eps_eff(W, er=4.4, h=1.6e-3):
    """Effective dielectric constant (Hammerstad formula)."""
    return (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)


def delta_L(W, er=4.4, h=1.6e-3):
    """Fringing-field length extension ΔL."""
    ee  = eps_eff(W, er, h)
    WH  = W / h
    num = (ee + 0.3) * (WH + 0.264)
    den = (ee - 0.258) * (WH + 0.8)
    return 0.412 * h * (num / den)


def resonant_freq(W, L, er=4.4, h=1.6e-3):
    """Resonant frequency from the transmission-line model."""
    ee = eps_eff(W, er, h)
    dL = delta_L(W, er, h)
    return 3.0e8 / (2.0 * (L + 2.0 * dL) * np.sqrt(ee))


def s11_at_target(L, W=35.1e-3, f_target=2.6e9,
                  R_in=52.0, Q=22.0, Z0=50.0):
    """
    S11 at f_target using parallel RLC resonator centred at f_res(W, L).
    Returns S11 in dB (more negative = better match).
    """
    if L <= 0:
        return 0.0
    f_res = resonant_freq(W, L)
    u     = Q * (f_target / f_res - f_res / f_target)
    Zin   = R_in / (1.0 + 1j * u)
    Gamma = (Zin - Z0) / (Zin + Z0)
    return 20.0 * np.log10(np.abs(Gamma) + 1e-15)


# ── Fixed parameters ─────────────────────────────────────────────────────────
W_fixed  = 35.1e-3     # m  — width fixed from analytical formula
f_target = 2.6e9       # Hz

# ── Parametric sweep of L (22 – 33 mm) ──────────────────────────────────────
L_sweep   = np.linspace(22e-3, 33e-3, 300)
s11_sweep = np.array([s11_at_target(L, W_fixed, f_target) for L in L_sweep])
f_sweep   = np.array([resonant_freq(W_fixed, L) / 1e9 for L in L_sweep])

# ── SciPy Nelder-Mead optimisation ──────────────────────────────────────────
def objective(x):
    L_val = x[0]
    if L_val < 10e-3 or L_val > 45e-3:
        return 0.0          # out-of-range penalty (bad S11 → 0 dB)
    return s11_at_target(L_val, W_fixed, f_target)

result   = minimize(objective, [28e-3], method='Nelder-Mead',
                    options={'xatol': 1e-7, 'fatol': 1e-4, 'maxiter': 10000})
L_opt    = result.x[0]
s11_opt  = result.fun
f_opt    = resonant_freq(W_fixed, L_opt) / 1e9
ee_val   = eps_eff(W_fixed)
dL_val   = delta_L(W_fixed)
L_theory = 3e8 / (2 * f_target * np.sqrt(ee_val)) - 2 * dL_val

# ── Print results ────────────────────────────────────────────────────────────
print("=" * 54)
print("  Patch Length Optimisation Results")
print("=" * 54)
print(f"  Target frequency        : {f_target / 1e9:.3f} GHz")
print(f"  Fixed patch width W     : {W_fixed * 1e3:.2f} mm")
print(f"  Effective permittivity  : {ee_val:.3f}")
print(f"  ΔL (fringing, each end) : {dL_val * 1e3:.3f} mm")
print(f"  Analytical optimum L    : {L_theory * 1e3:.2f} mm")
print(f"  SciPy optimum L         : {L_opt * 1e3:.2f} mm")
print(f"  Resonant frequency      : {f_opt:.4f} GHz")
print(f"  Best S11 at 2.6 GHz     : {s11_opt:.1f} dB")
print(f"  Optimiser converged     : {result.success}")
print("=" * 54)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(L_sweep * 1e3, s11_sweep, color='steelblue', linewidth=2.0)
ax1.axvline(x=L_opt * 1e3, color='red', linestyle='--', linewidth=1.5,
            label=f'Optimal L = {L_opt*1e3:.2f} mm  →  S11 = {s11_opt:.1f} dB')
ax1.axhline(y=-10.0, color='gray', linestyle=':', linewidth=1.2,
            label='-10 dB threshold')
ax1.set_ylabel('S11 at 2.6 GHz (dB)', fontsize=11)
ax1.set_title('Patch Length Optimisation — LTE 2.6 GHz\n'
              '(FR4, h = 1.6 mm, W = 35.1 mm)', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_ylim([-38, 2])
ax1.grid(True, alpha=0.3)

ax2.plot(L_sweep * 1e3, f_sweep, color='darkorange', linewidth=2.0)
ax2.axhline(y=f_target / 1e9, color='red', linestyle='--', linewidth=1.5,
            label=f'Target: {f_target/1e9:.2f} GHz')
ax2.axvline(x=L_opt * 1e3, color='red', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Patch Length L (mm)', fontsize=11)
ax2.set_ylabel('Resonant Frequency (GHz)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('patch_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: patch_optimization.png")
