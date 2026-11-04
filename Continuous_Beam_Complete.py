# -*- coding: utf-8 -*-
"""Project 2: Three-span continuous beam (OpenSeesPy).

Based on the original Continuous_beam.py model.
Units: N, mm, N/mm^2, N/mm, and N.mm.
Run in Spyder after installing openseespy, numpy, matplotlib.
Outputs are saved in a folder named continuous_beam_results beside this script.

Sign convention for plots:
  vertical deflection: positive UP
  shear: positive on the left face in the usual SFD convention
  bending moment: positive sagging, negative hogging
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import openseespy.opensees as ops

# ---------- 1. Input values (N, mm) ----------
E = 200_000.0
B = np.array([300.0, 300.0, 300.0])
HEIGHTS = np.array([500.0, 572.36, 630.0])
LENGTHS = np.array([3_000.0, 3_000.0, 8_000.0])
W_DOWN = np.array([50.0, 0.0, 25.0])  # N/mm; 1 N/mm = 1 kN/m
SUPPORT_X = np.r_[0.0, np.cumsum(LENGTHS)]  # mm
N_PLOT = 501

try:
    OUTPUT_DIR = Path(__file__).resolve().parent / 'continuous_beam_results'
except NameError:  # permits running code interactively
    OUTPUT_DIR = Path.cwd() / 'continuous_beam_results'
OUTPUT_DIR.mkdir(exist_ok=True)


def run_analysis(depth_3_factor=1.0):
    """Build and solve the beam, returning reactions, end forces and nodal DOFs."""
    heights = HEIGHTS.copy()
    heights[2] *= depth_3_factor
    areas = B * heights
    inertias = B * heights**3 / 12.0

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    for node_id, x in enumerate(SUPPORT_X, start=1):
        ops.node(node_id, float(x), 0.0)

    # Node 1: fixed; Nodes 2 & 3: rollers; Node 4: pin
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 0)
    ops.fix(3, 0, 1, 0)
    ops.fix(4, 1, 1, 0)

    ops.geomTransf('Linear', 1)
    for e in range(3):
        ops.element('elasticBeamColumn', e + 1, e + 1, e + 2,
                    float(areas[e]), E, float(inertias[e]), 1)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for e, w in enumerate(W_DOWN, start=1):
        if w != 0:
            ops.eleLoad('-ele', e, '-type', '-beamUniform', -float(w))

    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    if ops.analyze(1) != 0:
        raise RuntimeError('OpenSees analysis did not converge.')

    ops.reactions()
    reactions = np.array([ops.nodeReaction(i, 2) for i in range(1, 5)])
    fixed_moment = ops.nodeReaction(1, 3)
    end_forces = [np.array(ops.eleForce(i), dtype=float) for i in range(1, 4)]
    # Each nodal displacement is [horizontal mm, vertical mm, rotation rad].
    nodal_dofs = np.array([ops.nodeDisp(i) for i in range(1, 5)])
    return dict(heights=heights, inertias=inertias, reactions=reactions,
                fixed_moment=fixed_moment, end_forces=end_forces,
                nodal_dofs=nodal_dofs)


def get_diagrams(result):
    """Compute exact within-element shear/moment and vertical deflection curves.

    Deflection uses cubic Hermite interpolation of OpenSees nodal DOFs PLUS
    the exact uniformly distributed load particular solution. The extra term
    is essential: zero vertical displacement at the support nodes does NOT
    imply zero deflection between supports.
    """
    spans = []
    for e in range(3):
        L = LENGTHS[e]
        w = W_DOWN[e]
        I = result['inertias'][e]
        f = result['end_forces'][e]
        x = np.linspace(0.0, L, N_PLOT)  # local mm
        x_global = SUPPORT_X[e] + x

        # f = [Fxi, Fyi, Mzi, Fxj, Fyj, Mzj] (N, N.mm)
        # Section signs: M(0) = -Mzi and V(0) = Fyi.
        shear = f[1] - w * x
        moment = -f[2] + f[1] * x - 0.5 * w * x**2

        # Elastic Euler-Bernoulli beam deflection using nodal DOFs.
        xi = x / L
        v_i = result['nodal_dofs'][e, 1]
        theta_i = result['nodal_dofs'][e, 2]
        v_j = result['nodal_dofs'][e + 1, 1]
        theta_j = result['nodal_dofs'][e + 1, 2]
        n1 = 1 - 3*xi**2 + 2*xi**3
        n2 = L * (xi - 2*xi**2 + xi**3)
        n3 = 3*xi**2 - 2*xi**3
        n4 = L * (-xi**2 + xi**3)
        v_from_nodes = n1*v_i + n2*theta_i + n3*v_j + n4*theta_j
        # q is positive upward; downward distributed load is q = -w.
        v_due_to_load = -w * x**2 * (L - x)**2 / (24 * E * I)
        deflection = v_from_nodes + v_due_to_load

        spans.append(dict(x=x, x_global=x_global, shear=shear,
                          moment=moment, deflection=deflection,
                          start_shear=f[1], start_moment=-f[2]))
    return spans


def critical_moments(result):
    """Evaluate endpoints plus all interior V=0 stations, span by span."""
    candidates = []
    for e in range(3):
        L = LENGTHS[e]
        w = W_DOWN[e]
        f = result['end_forces'][e]
        locations = [0.0, L]
        if abs(w) > 1e-14:
            x_zero_shear = f[1] / w
            if 0.0 < x_zero_shear < L:
                locations.append(x_zero_shear)
        for x in locations:
            m = -f[2] + f[1]*x - w*x*x/2
            candidates.append((SUPPORT_X[e] + x, m, e+1))
    highest = max(candidates, key=lambda c: c[1])
    lowest = min(candidates, key=lambda c: c[1])
    absolute = max(candidates, key=lambda c: abs(c[1]))
    return highest, lowest, absolute


def save_sfd(spans, reactions):
    fig, ax = plt.subplots(figsize=(11, 5))
    for e, span in enumerate(spans, start=1):
        ax.plot(span['x_global']/1000, span['shear']/1000,
                linewidth=2, label=f'Element {e}')
    # At supports, applied reactions produce vertical shear jumps.
    # Elements on either side are plotted separately to preserve each jump.
    before_after = [(0.0, spans[0]['shear'][0]),
                    (spans[0]['shear'][-1], spans[1]['shear'][0]),
                    (spans[1]['shear'][-1], spans[2]['shear'][0]),
                    (spans[2]['shear'][-1], 0.0)]
    for x, (v_before, v_after), reaction in zip(SUPPORT_X/1000,
                                                before_after, reactions/1000):
        ax.plot([x, x], [v_before/1000, v_after/1000],
                'k--', linewidth=1.2)
        ax.axvline(x, color='gray', linestyle=':', linewidth=0.7, alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.9)
    ax.set(title='Continuous Beam — Shear Force Diagram',
           xlabel='Distance from Node 1 (m)', ylabel='Shear force (kN)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'Shear_Force_Diagram.png', dpi=300)
    plt.close(fig)


def save_bmd(spans, highest, lowest):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for e, span in enumerate(spans, start=1):
        ax.plot(span['x_global']/1000, span['moment']/1e6,
                linewidth=2, label=f'Element {e}')
    for x in SUPPORT_X / 1000:
        ax.axvline(x, color='gray', linestyle=':', linewidth=0.8, alpha=0.65)
    for point, color, offset in [(highest, 'red', (10, 13)),
                                 (lowest, 'blue', (10, -22))]:
        x_m, m_nmm, _ = point
        ax.scatter(x_m/1000, m_nmm/1e6, color=color, s=55, zorder=5)
        ax.annotate(f'{m_nmm/1e6:+.2f} kN.m at {x_m/1000:.3f} m',
                    (x_m/1000, m_nmm/1e6), xytext=offset,
                    textcoords='offset points', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.9)
    ax.set(title='Continuous Beam — Bending Moment Diagram (Sagging +)',
           xlabel='Distance from Node 1 (m)', ylabel='Bending moment (kN.m)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.margins(y=0.19)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'Bending_Moment_Diagram.png', dpi=300)
    plt.close(fig)


def save_deflection(spans):
    fig, ax = plt.subplots(figsize=(11, 5))
    for e, span in enumerate(spans, start=1):
        ax.plot(span['x_global']/1000, span['deflection'],
                linewidth=2, label=f'Element {e}')
    for x in SUPPORT_X/1000:
        ax.axvline(x, color='gray', linestyle=':', linewidth=0.8, alpha=0.65)
    ax.axhline(0, color='black', linewidth=0.9)
    ax.set(title='Continuous Beam — Vertical Deflection (Upward +)',
           xlabel='Distance from Node 1 (m)', ylabel='Vertical displacement (mm)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'Deflected_Shape.png', dpi=300)
    plt.close(fig)


def max_abs_deflection(spans):
    """Find largest sampled |vertical displacement| and its position."""
    points = [(s['x_global'][i], s['deflection'][i])
              for s in spans for i in range(len(s['x']))]
    return max(points, key=lambda pair: abs(pair[1]))


def save_parametric_study(results):
    factors = np.array([0.8, 1.0, 1.2])
    max_deflections = np.array([abs(r[1][1]) for r in results])
    max_absolute_moments = np.array([abs(r[2][1])/1e6 for r in results])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(factors, max_deflections, 'o-', linewidth=2)
    ax.set(title='Parametric Study — Element 3 Depth',
           xlabel='Element 3 depth / original depth',
           ylabel='Maximum absolute beam deflection (mm)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'Parametric_Study_Deflection.png', dpi=300)
    plt.close(fig)

    print('\nPARAMETRIC STUDY: changing only Element 3 depth')
    print('Depth factor | Depth (mm) | Max |defl.| (mm) | Max |M| (kN.m)')
    for factor, deflection, moment in zip(factors, max_deflections,
                                          max_absolute_moments):
        print(f'{factor:11.2f} | {HEIGHTS[2]*factor:10.2f} | '
              f'{deflection:16.5f} | {moment:14.3f}')


# ---------- 2. Original beam analysis ----------
base = run_analysis()
spans = get_diagrams(base)
highest, lowest, absolute = critical_moments(base)
x_deflection, y_deflection = max_abs_deflection(spans)

# ---------- 3. Verification ----------
total_load = np.dot(W_DOWN, LENGTHS)  # N
vertical_error = sum(base['reactions']) - total_load
# All positive moments counterclockwise; resultant of each UDL at span centre.
load_moment_about_node1 = sum(
    W_DOWN[e] * LENGTHS[e] * (SUPPORT_X[e] + LENGTHS[e]/2)
    for e in range(3)
)
moment_error = (base['fixed_moment']
                + np.dot(base['reactions'], SUPPORT_X)
                - load_moment_about_node1)

print('=== ORIGINAL CONTINUOUS BEAM ===')
for i, reaction in enumerate(base['reactions'], start=1):
    print(f'Node {i} vertical reaction: {reaction/1000:+.4f} kN')
print(f'Node 1 reaction moment: {base["fixed_moment"]/1e6:+.4f} kN.m')
print(f'Sum Fy residual: {vertical_error:.6f} N')
print(f'Sum M at Node 1 residual: {moment_error:.6f} N.mm')
print(f'Max positive moment: {highest[1]/1e6:+.4f} kN.m '
      f'at x = {highest[0]/1000:.4f} m (Element {highest[2]})')
print(f'Most negative moment: {lowest[1]/1e6:+.4f} kN.m '
      f'at x = {lowest[0]/1000:.4f} m (Element {lowest[2]})')
print(f'Max absolute moment: {abs(absolute[1])/1e6:.4f} kN.m '
      f'at x = {absolute[0]/1000:.4f} m')
print(f'Max sampled absolute deflection: {abs(y_deflection):.5f} mm '
      f'at x = {x_deflection/1000:.4f} m '
      f'(signed v = {y_deflection:+.5f} mm)')

assert abs(vertical_error) < 1e-5, 'Vertical force equilibrium failed'
assert abs(moment_error) < 1e-2, 'Moment equilibrium failed'
for e in [0, 1]:
    left_m = spans[e]['moment'][-1]
    right_m = spans[e+1]['moment'][0]
    assert abs(left_m - right_m) < 1e-2, 'BMD continuity failed'
assert abs(spans[2]['moment'][-1]) < 1e-2, 'Pinned end moment not zero'

# ---------- 4. Save original-model plots ----------
save_sfd(spans, base['reactions'])
save_bmd(spans, highest, lowest)
save_deflection(spans)

# ---------- 5. Simple parametric study ----------
study = []
for factor in [0.8, 1.0, 1.2]:
    result = run_analysis(depth_3_factor=factor)
    diagram = get_diagrams(result)
    _, _, largest_m = critical_moments(result)
    study.append((factor, max_abs_deflection(diagram), largest_m))
save_parametric_study(study)

print(f'\nAll four figures saved to:\n{OUTPUT_DIR}')
