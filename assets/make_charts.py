"""Draw the README charts (light and dark variants) from the Public scores recorded during the challenge.

Usage: python assets/make_charts.py      -> writes assets/*.png
Colours follow a validated palette: one hue (blue) for one-series and before/after encodings,
with the ordinal shades checked against each surface (light #fcfcfb, dark #1a1a19).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).parent
FONT = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']

THEMES = {
    'light': dict(bg='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  line='#2a78d6', before='#86b6ef', after='#1c5cab'),
    'dark': dict(bg='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 line='#3987e5', before='#184f95', after='#6da7ec'),
}

# Mean Public score over the ten systems at each milestone. The last point is an expectation, not a result.
PROGRESS = [
    ('Starter model,\n1 system', 0.039),
    ('First pass,\nall 10 systems', 0.496),
    ('Tuned direct\nmodels (rank 45)', 0.525),
    ('GRU state-space\nmodels', 0.686),
    ('+ Reservoir inflow\ncurve (rank 21)', 0.735),
    ('Final result\n(4th place)', 0.757),
]
LEADER = 0.803  # score of rank 1 on the last published board

# (system, first all-ten submission on the Public board, Final result on the hidden Final episodes)
SYSTEMS = [
    ('Ad auction', 0.5485, 0.8647), ('Traffic', 0.5800, 0.8267), ('Reservoir', 0.5664, 0.7844),
    ('Wildlife', 0.4803, 0.7819), ('Power grid', 0.5803, 0.7535), ('Epidemic', 0.3526, 0.7511),
    ('Supply chain', 0.3512, 0.7362), ('Market', 0.5081, 0.7041), ('Hospital queue', 0.5406, 0.6972),
    ('Social contagion', 0.4526, 0.6655),
]


def base(theme, figsize):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = FONT
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(theme['bg']); ax.set_facecolor(theme['bg'])
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(theme['axis'])
    ax.tick_params(colors=theme['muted'], length=0, labelsize=9)
    return fig, ax


def progress(name, t):
    fig, ax = base(t, (8.6, 4.6))
    xs = list(range(len(PROGRESS))); ys = [v for _, v in PROGRESS]
    ax.set_ylim(0, 0.92); ax.set_xlim(-0.45, len(xs) - 0.4)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8]); ax.yaxis.grid(True, color=t['grid'], linewidth=0.8); ax.set_axisbelow(True)
    ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in PROGRESS], color=t['ink2'], fontsize=8.5)
    ax.axhline(LEADER, color=t['muted'], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(-0.42, LEADER + 0.012, f'Rank 1 on the last published board: {LEADER:.3f}', color=t['muted'], fontsize=8.5, va='bottom')
    ax.plot(xs, ys, color=t['line'], linewidth=2.4, solid_capstyle='round', zorder=3)
    ax.plot(xs, ys, 'o', color=t['line'], markersize=8.5, markeredgecolor=t['bg'], markeredgewidth=2, zorder=4)
    ax.annotate(f'{ys[0]:.3f}', (xs[0], ys[0]), xytext=(-12, 0), textcoords='offset points', ha='right', va='center',
                color=t['ink'], fontsize=10.5, fontweight='bold')
    for i in (1, 3):
        ax.annotate(f'{ys[i]:.3f}', (xs[i], ys[i]), xytext=(0, 11), textcoords='offset points', ha='center', va='bottom',
                    color=t['ink'], fontsize=10.5, fontweight='bold')
    ax.annotate(f'{ys[-1]:.3f}', (xs[-1], ys[-1]), xytext=(0, -13), textcoords='offset points', ha='center', va='top',
                color=t['ink'], fontsize=10.5, fontweight='bold')
    fig.text(0.075, 0.955, 'Mean score over the ten systems', color=t['ink'], fontsize=12.5, fontweight='bold', ha='left', va='top')
    fig.text(0.075, 0.905, 'Public scores through the build, then the Final result on the hidden Final episodes (0.7565)', color=t['ink2'], fontsize=9.5, ha='left', va='top')
    fig.subplots_adjust(left=0.075, right=0.975, top=0.83, bottom=0.17)
    fig.savefig(OUT / f'progress_{name}.png', facecolor=t['bg']); plt.close(fig)


def systems(name, t):
    fig, ax = base(t, (8.6, 5.4))
    n = len(SYSTEMS)
    ax.set_xlim(0.3, 0.95); ax.set_ylim(n - 0.4, -1.1)
    ax.set_yticks(range(n)); ax.set_yticklabels([s for s, _, _ in SYSTEMS], color=t['ink2'], fontsize=9.5)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]); ax.xaxis.grid(True, color=t['grid'], linewidth=0.8); ax.set_axisbelow(True)
    ax.spines['bottom'].set_visible(False)
    for i, (_, a, b) in enumerate(SYSTEMS):
        ax.plot([a, b], [i, i], color=t['before'], linewidth=2.6, solid_capstyle='butt', zorder=2)
        ax.plot([a], [i], 'o', color=t['before'], markersize=9, markeredgecolor=t['bg'], markeredgewidth=2, zorder=3)
        ax.plot([b], [i], 'o', color=t['after'], markersize=9, markeredgecolor=t['bg'], markeredgewidth=2, zorder=4)
        ax.text(1.015, i, f'+{b - a:.2f}', transform=ax.get_yaxis_transform(), color=t['ink2'], fontsize=9.5, va='center', ha='left')
    ax.text(1.015, -0.85, 'gain', transform=ax.get_yaxis_transform(), color=t['muted'], fontsize=8.5, va='center', ha='left')
    # direct labels on the first row plus a legend, so identity never relies on colour alone
    _, a0, b0 = SYSTEMS[0]
    ax.annotate('first pass', (a0, 0), xytext=(-10, 0), textcoords='offset points', ha='right', va='center', color=t['ink2'], fontsize=8.5)
    ax.annotate('final', (b0, 0), xytext=(11, 0), textcoords='offset points', ha='left', va='center', color=t['ink2'], fontsize=8.5)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=t['before'], markeredgecolor=t['bg'], markersize=9, label='First pass over all ten systems'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=t['after'], markeredgecolor=t['bg'], markersize=9, label='Final result')]
    leg = ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=8.5, handletextpad=0.3, columnspacing=1.6, borderaxespad=0.2)
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])
    fig.text(0.02, 0.965, 'Score for each system: first Public pass and Final result', color=t['ink'], fontsize=12.5, fontweight='bold', ha='left', va='top')
    fig.subplots_adjust(left=0.19, right=0.91, top=0.83, bottom=0.06)
    fig.savefig(OUT / f'systems_{name}.png', facecolor=t['bg']); plt.close(fig)


if __name__ == '__main__':
    for nm in ('light',):  # the README uses the light variants; add 'dark' here to render both
        progress(nm, THEMES[nm]); systems(nm, THEMES[nm])
    print('wrote', ', '.join(sorted(p.name for p in OUT.glob('*.png'))))
