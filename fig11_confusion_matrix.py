import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mdpi_figure_style import SQ, BLUE, INK, save

cm = np.array([[20906, 10424, 132],
               [14983, 177428, 21826],
               [250, 22252, 137351]])
rown = cm / cm.sum(axis=1, keepdims=True)
cmap = LinearSegmentedColormap.from_list("mdpi_blue", ["#ffffff", BLUE])

fig, ax = plt.subplots(figsize=(SQ * 1.18, SQ))
ax.grid(False)
im = ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]:,}\n({100 * rown[i, j]:.1f}%)", ha="center", va="center",
                color="white" if rown[i, j] > 0.75 else INK)
labels = ["Hypo", "Normo", "Hyper"]
ax.set_xticks(range(3), labels)
ax.set_yticks(range(3), labels)
ax.tick_params(length=0)
ax.set(xlabel="Predicted class", ylabel="True class")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Row-normalized rate")
cb.outline.set_linewidth(0.6)
save(fig, "fig11_confusion_45min", outdir=sys.argv[1] if len(sys.argv) > 1 else ".")
print(np.round(100 * rown, 1))
