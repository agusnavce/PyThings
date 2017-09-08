from matplotlib import pyplot as mp
import numpy as np
from scipy import stats
from scipy.stats import norm

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#for mu, sig in [(0, 1), (2.3, 1.5)]:


 #   mp.plot(gaussian(np.linspace(-3, 6, 120), mu, sig)

mu,sig = (0, 1)
x = np.linspace(mu - 5 * sig, mu + 5 * sig, 1000)
colors = ['c', 'r', 'b', 'g', ]
colors = colors + list(reversed(colors))

iq = stats.norm(mu, sig)
low = -4
high = 1.5
px = x[np.logical_and(x >= low, x <= high)]
mp.fill_between(
        px,
        iq.pdf(px),
        color="blue",
        alpha=0.5,
        linewidth=0,
    )
    

mu,sig = (3, 1)
x = np.linspace(mu - 5 * sig, mu + 5 * sig, 1000)
iq = stats.norm(mu, sig)

low = 1.5
high = 6
px = x[np.logical_and(x >= low, x <= high)]
mp.fill_between(
        px,
        iq.pdf(px),
        color="red",
        alpha=0.5,
        linewidth=0,
    )    

mu,sig = (0, 1)
x = np.linspace(mu - 5 * sig, mu + 5 * sig, 1000)
iq = stats.norm(mu, sig)
low = 1.5
high = 4
px = x[np.logical_and(x >= low, x <= high)]
mp.fill_between(
        px,
        iq.pdf(px),
        color="green",
        alpha=0.5,
        linewidth=0,
    )
    
    
mu,sig = (3, 1)
x = np.linspace(mu - 5 * sig, mu + 5 * sig, 1000)
iq = stats.norm(mu, sig)

low = -1
high = 1.5
px = x[np.logical_and(x >= low, x <= high)]
mp.fill_between(
        px,
        iq.pdf(px),
        color="orange",
        alpha=0.5,
        linewidth=0,
    ) 
mp.tight_layout()    


mp.savefig("gauss.png")
