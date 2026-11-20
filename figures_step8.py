"""Figures for step 8 (official logger), in the BME Suborbitals visual style:
near-black background, white type, crimson accent, periwinkle data, Noto Sans."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in glob.glob(os.path.expanduser("~/.fonts/NotoSans-*.ttf")):
    fm.fontManager.addfont(f)

BG, PANEL, INK, MUTED, GRID = "#000000", "#111113", "#ffffff", "#b3b3b3", "#2b2b2b"
CRIMSON, BURGUNDY, PERI, GOOD = "#c7244b", "#8e202b", "#c7cfff", "#3ddc84"
COL = {"true": CRIMSON, "isa": "#f0a35e", "logger": PERI}
LABEL = {"true": "Aims for true 3,000 m", "isa": "Library formula, mixed into the filter", "logger": "Aims for the logger's 3,000 m"}

plt.rcParams.update({
    "font.family": "Noto Sans", "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": PANEL,
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK,
})


def title(fig, kicker, text):
    fig.text(0.03, 0.955, kicker.upper(), color=CRIMSON, fontsize=9.5, fontweight="black", va="top")
    fig.text(0.03, 0.905, text, color=INK, fontsize=14.5, fontweight="bold", va="top")


def main():
    d = json.load(open("results/official_logger.json"))
    os.makedirs("figures", exist_ok=True)

    # 1) logged miss vs ground temperature, nominal rocket
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axvspan(15, 25, color=BURGUNDY, alpha=0.25, lw=0)
    ax.text(20, 0.97, "Typical October\ndaytime, Central Portugal", transform=ax.get_xaxis_transform(),
            ha="center", va="top", color=MUTED, fontsize=9)
    ax.axhline(0, color=INK, lw=1)
    for m in ("true", "isa", "logger"):
        pts = sorted((r["T"], r["logged"] - 3000) for r in d["sweep"] if r["mode"] == m and r["T"] <= 30)
        T, miss = zip(*pts)
        ax.plot(T, miss, "-o", color=COL[m], lw=2.6, ms=5)
        dy = {"true": 0, "isa": 9, "logger": -9}[m]
        ax.annotate(LABEL[m], (T[-1], miss[-1]), xytext=(10, dy), textcoords="offset points", color=COL[m],
                    va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(4, 42)
    ax.set_xticks(range(5, 35, 5))
    ax.set_xlabel("Ground temperature on launch day [°C]")
    ax.set_ylabel("Logged apogee minus 3,000 m [m]")
    title(fig, "Same rocket, same airbrake, only the altitude units differ",
          "Scored apogee drifts about 10 m per °C unless the airbrake aims for the logger's altitude")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.12)
    fig.savefig("figures/step8_logged_vs_temperature.png", dpi=150)
    plt.close(fig)

    # 2) Monte Carlo: distribution of logged miss per mode
    reach = [r for r in d["mc"] if r["closed_logged"] >= 3000]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    rng = np.random.default_rng(1)
    for i, m in enumerate(("true", "isa", "logger")):
        miss = np.array([r[m + "_logged"] - 3000 for r in reach])
        s = d["summary"][m]
        y = 2 - i
        ax.axhspan(y - 0.32, y + 0.32, xmin=0, xmax=1, color=PANEL, lw=0)
        ax.plot([s["p2_5"], s["p97_5"]], [y, y], color=COL[m], lw=10, alpha=0.25, solid_capstyle="butt")
        ax.scatter(miss, y + rng.uniform(-0.22, 0.22, len(miss)), s=9, color=COL[m], alpha=0.75, lw=0)
        ax.plot([s["mean"]] * 2, [y - 0.3, y + 0.3], color=INK, lw=2)
        ax.text(1.01, y, f"1σ {s['sd']:.0f} m\n95%: {s['p2_5']:+.0f} to {s['p97_5']:+.0f} m",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9.5, color=COL[m])
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks([2, 1, 0], [LABEL["true"], LABEL["isa"], LABEL["logger"]])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Logged apogee minus 3,000 m [m]  (bar: 95% of flights, white tick: mean)")
    title(fig, f"{d['n_reach']} simulated EuRoC flights: weather, wind, motor and build errors",
          "Aiming for the logger's altitude centres the scored apogee on target")
    fig.subplots_adjust(left=0.30, right=0.80, top=0.78, bottom=0.15)
    fig.savefig("figures/step8_monte_carlo.png", dpi=150)
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
