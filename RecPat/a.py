from matplotlib import pyplot as mp
import numpy as np
from scipy import stats

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

for mu, sig in [(0, 1), (2.3, 1.5)]:
    mp.plot(gaussian(np.linspace(-3, 6, 120), mu, sig))

x = np.linspace(mu - 5 * sig, mu + 5 * sig, 1000)
colors = ['c', 'r', 'b', 'g', ]
colors = colors + list(reversed(colors))

iq = stats.norm(mu, sig)
for i, color in zip(range(-4, 4), colors):
    low = mu + i * sig
    high = mu + (i + 1) * sig
    px = x[np.logical_and(x >= low, x <= high)]
    mp.fill_between(
        px,
        iq.pdf(px),
        color=color,
        alpha=0.5,
        linewidth=0,
    )

mp.tight_layout()    

mp.show()
