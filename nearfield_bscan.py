"""
============================================================
 BN304 Final Assignment — Script 5 of 6
 Near-field RF Sensing — Synthetic B-scan Simulation
 Sensor frequency: 500 MHz
 Targets: steel pipe at 0.30 m depth, PVC cable at 0.50 m depth
 Run:     python nearfield_bscan.py
 Output:  nearfield_bscan.png
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian_pulse(t, t_centre, f_centre, n_cycles=2.0):
    """
    Modulated Gaussian pulse (Ricker-style wavelet).

    Parameters
    ----------
    t        : ndarray  Time axis in seconds
    t_centre : float    Centre time of the pulse in seconds
    f_centre : float    Carrier frequency in Hz
    n_cycles : float    Number of cycles in the Gaussian envelope

    Returns
    -------
    pulse : ndarray
    """
    sigma    = n_cycles / (2.0 * np.pi * f_centre)
    envelope = np.exp(-0.5 * ((t - t_centre) / sigma) ** 2)
    carrier  = np.cos(2.0 * np.pi * f_centre * (t - t_centre))
    return envelope * carrier


def bscan_simulate(x_pos, t_axis, targets,
                   soil_eps=4.0, f_centre=500e6, noise_level=0.04):
    """
    Generate a synthetic B-scan radargram.

    Parameters
    ----------
    x_pos       : ndarray  Horizontal scan positions in metres
    t_axis      : ndarray  Two-way travel time axis in seconds
    targets     : list     Each element: (depth_m, x_m, amplitude)
                           Negative amplitude = metallic (phase reversal)
    soil_eps    : float    Soil relative permittivity
    f_centre    : float    Centre frequency in Hz
    noise_level : float    Gaussian noise amplitude

    Returns
    -------
    bscan : ndarray  Shape (len(t_axis), len(x_pos))
    """
    v     = 3.0e8 / np.sqrt(soil_eps)    # propagation velocity in soil
    bscan = np.zeros((len(t_axis), len(x_pos)))
    rng   = np.random.default_rng(seed=42)

    for xi, x in enumerate(x_pos):
        trace = np.zeros(len(t_axis))

        # Direct wave — air coupling at the surface (slight delay)
        trace += 1.0 * gaussian_pulse(t_axis, 0.8e-9, f_centre)

        # Reflection from each buried target
        for (depth, x_target, amp) in targets:
            r         = np.sqrt(depth ** 2 + (x - x_target) ** 2)
            t_reflect = 2.0 * r / v
            att       = np.exp(-0.5 * 2.0 * r)   # geometric + soil attenuation
            a_eff     = amp * att / (r + 0.01)
            trace    += a_eff * gaussian_pulse(t_axis, t_reflect, f_centre)

        # Add random noise
        trace += noise_level * rng.standard_normal(len(t_axis))
        bscan[:, xi] = trace

    return bscan


# ── Simulation parameters ────────────────────────────────────────────────────
f_c      = 500e6                        # 500 MHz centre frequency
soil_eps = 4.0                          # dry sandy soil
v_soil   = 3.0e8 / np.sqrt(soil_eps)   # 1.5 × 10^8 m/s

# Time axis: 0 to 15 ns, 50 ps time step
t_axis = np.arange(0, 15e-9, 0.05e-9)

# Horizontal scan: -0.8 m to +0.8 m in 5 cm steps (33 positions)
x_pos = np.linspace(-0.8, 0.8, 65)

# Buried targets: (depth_m, x_position_m, amplitude)
#   Negative amplitude → metallic target (phase reversal)
#   Positive amplitude → dielectric target (same phase)
targets = [
    (0.30,  0.00, -3.0),   # Steel pipe  — 0.30 m depth, centred at x = 0
    (0.50,  0.25,  1.8),   # PVC cable   — 0.50 m depth, at x = 0.25 m
]

# ── Run simulation ───────────────────────────────────────────────────────────
bscan = bscan_simulate(x_pos, t_axis, targets, soil_eps, f_c)

# ── Expected travel times ────────────────────────────────────────────────────
t1_ns = 2.0 * targets[0][0] / v_soil * 1e9
t2_ns = 2.0 * targets[1][0] / v_soil * 1e9

print("=" * 52)
print("  Near-field B-scan Simulation Results")
print("=" * 52)
print(f"  Centre frequency          : {f_c/1e6:.0f} MHz")
print(f"  Soil velocity             : {v_soil:.2e} m/s")
print(f"  Wavelength in soil        : {v_soil/f_c*100:.1f} cm")
print(f"  Steel pipe apex time      : {t1_ns:.1f} ns  (depth={targets[0][0]} m)")
print(f"  PVC cable apex time       : {t2_ns:.1f} ns  (depth={targets[1][0]} m)")
print("=" * 52)

# ── Plot ─────────────────────────────────────────────────────────────────────
t_ns   = t_axis * 1e9
extent = [x_pos[0], x_pos[-1], t_ns[-1], t_ns[0]]
vmax   = 0.6 * np.max(np.abs(bscan))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: B-scan image
ax = axes[0]
im = ax.imshow(bscan, extent=extent, aspect='auto', cmap='seismic',
               vmin=-vmax, vmax=vmax, interpolation='bilinear')
plt.colorbar(im, ax=ax, label='Signal Amplitude (a.u.)')

# Overlay theoretical hyperbolas
x_hyp = np.linspace(-0.8, 0.8, 500)
for depth, x0, amp in targets:
    r     = np.sqrt(depth ** 2 + (x_hyp - x0) ** 2)
    t_hyp = 2.0 * r / v_soil * 1e9
    style = 'w--' if amp < 0 else 'c--'
    lbl   = (f'Steel pipe  d = {depth} m' if amp < 0
             else f'PVC cable   d = {depth} m')
    ax.plot(x_hyp, t_hyp, style, linewidth=1.2, label=lbl, alpha=0.85)

ax.set_xlabel('Antenna Position (m)', fontsize=11)
ax.set_ylabel('Two-way Travel Time (ns)', fontsize=11)
ax.set_title('Synthetic B-scan — 500 MHz Near-field RF Sensor\n'
             'Steel pipe 0.30 m  |  PVC cable 0.50 m', fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim([t_ns[-1], 0.5])

# Right: A-scan trace at x = 0
ax2           = axes[1]
idx_centre    = np.argmin(np.abs(x_pos - 0.0))
trace_centre  = bscan[:, idx_centre]
ax2.plot(trace_centre, t_ns, color='steelblue', linewidth=1.5)
ax2.axhline(y=t1_ns, color='red', linestyle='--',
            label=f'Steel pipe reflection  ({t1_ns:.1f} ns)')
ax2.invert_yaxis()
ax2.set_xlabel('Amplitude (a.u.)', fontsize=11)
ax2.set_ylabel('Two-way Travel Time (ns)', fontsize=11)
ax2.set_title('A-scan at x = 0 m\n(Vertical trace through B-scan)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nearfield_bscan.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved: nearfield_bscan.png")
