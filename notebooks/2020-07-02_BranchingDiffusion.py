# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import matplotlib.pyplot as plt
import numpy as np


# +
class Individual:
    def __init__(self, parent, birth_time, birth_location):
        self.parent = parent
        self.birth_time = birth_time
        self.birth_location = birth_location
        self.death_time = None
        self.death_location = None
        self.children = []
        self.latest_time = None

    def __str__(self):
        return f"({self.birth_time}--{self.death_time} : \
                [{','.join([str(child) for child in self.children])}])"

    def die(self, death_time, death_location):
        self.death_time = death_time
        self.death_location = death_location
        self.update_latest_time(death_time)

    def reproduce(self, n_children, time, location):
        new_children = [Individual(self, time, location) for i in range(n_children)]
        self.children += new_children
        self.die(time, location)
        return new_children

    def update_latest_time(self, time):
        if not self.latest_time or time > self.latest_time:
            self.latest_time = time
            if self.parent:
                self.parent.update_latest_time(time)

    def location_at(self, time):
        if time < self.birth_time:
            return np.nan

        if self.death_time:
            mu = self.birth_location + (self.death_location - self.birth_location) * (
                time - self.birth_time
            ) / (self.death_time - self.birth_time)
            sigma = np.sqrt(
                (self.death_time - time)
                * (time - self.birth_time)
                / (self.death_time - self.birth_time)
            )
        else:
            mu = self.birth_location
            sigma = np.sqrt(time - self.birth_time)
        return np.random.normal(mu, sigma)

    def descendants_at(self, time):
        if time < self.birth_time:
            return []
        elif time <= self.death_time:
            return [self]
        elif time <= self.latest_time:
            return sum([child.descendants_at(time) for child in self.children], [])
        else:
            return []

    def resample_locations(self):
        if self.death_time:
            tmp = self.death_time
            self.death_time = None
            self.death_location = self.location_at(tmp)
            self.death_time = tmp
            for child in self.children:
                child.birth_location = self.death_location
                child.resample_locations()

    def plot(self, ax, **kwargs):
        ax.plot(
            [self.birth_time, self.death_time],
            [self.birth_location, self.death_location],
            lw=0.5,
            color="k",
            **kwargs,
        )
        for child in self.children:
            child.plot(ax, **kwargs)


class Population:
    def __init__(self):
        self.ancestor = Individual(None, 0.0, 0.0)
        self.living = [self.ancestor]
        self.popsize = 1
        self.cumsize = 1
        self.maxsize = 1

    def __str__(self):
        return str(self.ancestor)

    def death(self, i, t):
        dying = self.living.pop(i)
        loc = dying.location_at(t)
        dying.die(t, loc)
        self.popsize -= 1

    def birth(self, i, n, t):
        parent = self.living.pop(i)
        loc = parent.location_at(t)
        self.living += parent.reproduce(n, t, loc)
        self.popsize += n - 1
        self.cumsize += n
        self.maxsize = max(self.maxsize, self.popsize)

    def locations_at(self, t):
        return [indiv.location_at(t) for indiv in self.ancestor.descendants_at(t)]

    def resample_locations(self):
        self.ancestor.resample_locations()

    def plot(self, ax, **kwargs):
        self.ancestor.plot(ax, **kwargs)

    def get_latest_time(self):
        return self.ancestor.latest_time


def simulate(s=0.1, max_steps=1000):
    population = Population()
    p_death = (1 + s) / 2
    t = 0.0
    for step in range(max_steps):
        t += np.random.exponential() / population.popsize
        i = np.random.randint(population.popsize)
        if np.random.rand() < p_death:
            population.death(i, t)
        else:
            population.birth(i, 2, t)
        if population.popsize == 0:
            break
    else:
        print(
            f"Still alive after {max_steps} steps (t={t}). \
            Killing remaining individuals..."
        )
        for indiv in population.living:
            indiv.die(t)
    return population


# -

# ## Testing the code

# +
a = Individual(None, 0.0, 0.0)
t = 0.5
x = 1.0
a.reproduce(2, t, x)
a.die(t, x)

a.children[0].die(1.0, 1.2)
a.children[1].die(2.0, -0.4)

print(a)

print(a.descendants_at(0.0))
print(a.descendants_at(0.25))
print(a.descendants_at(0.51))
print(a.descendants_at(1.25))
print(a.descendants_at(2.51))

t = 0.99
print([c.location_at(t) for c in a.descendants_at(t)])
# -

ax = plt.subplot(111)
a.plot(ax)

for i in range(5):
    ax = plt.subplot(5, 1, i + 1)
    a.resample_locations()
    a.plot(ax)

pop = Population()
print(pop)
pop.birth(0, 2, 0.25)
print(pop)
pop.death(0, 0.5)
pop.death(0, 1.5)
print(pop)

ax = plt.subplot(111)
pop.plot(ax)

# +
a = Individual(None, 0.0, 0.0)
t0 = 1.0
x0 = 1.0
a.die(t0, x0)

ax = plt.subplot()
a.plot(ax)
plt.show()

for t in [0.01, 0.1, 0.5, 0.9, 0.99]:
    X = [a.location_at(t) for i in range(10000)]
    plt.hist(X, bins=np.linspace(-1, 2, 101), histtype="step")
# -

# ## Example simulations

np.random.seed(100)
npops = 1000
pops = [simulate(s=0.025, max_steps=10000) for i in range(npops)]

maxtimes = [pop.get_latest_time() for pop in pops]
maxsizes = [pop.maxsize for pop in pops]
plt.scatter(maxtimes, maxsizes, alpha=0.2)
m = max(maxtimes)
plt.plot([0, m], [0, m / 2], c="k")
plt.xlim([0, 150])
plt.ylim([0, 75])

cumsizes = [pop.cumsize for pop in pops]
plt.scatter(maxsizes, cumsizes, alpha=0.5)
m = max(maxsizes)
plt.yscale("log")
plt.xscale("log")
plt.plot([1, m], [1, m ** 2], c="k")

plt.hist(maxsizes, bins=np.arange(100), histtype="step")
plt.yscale("log")

times = [20, 30]
to_plot = 5
plotted = 0
for pop in pops:
    if pop.maxsize < 40:
        continue

    ax = plt.subplot(121)
    pop.plot(ax)
    ax.set_ylim([-20, 20])
    ax.vlines(times, -20, 20, linestyle="dashed", colors=["C0", "C1"])

    ax = plt.subplot(122)
    for t in times:
        plt.hist(
            pop.locations_at(t),
            histtype="stepfilled",
            orientation="horizontal",
            bins=np.linspace(-20, 20, 41),
            alpha=0.75,
        )
    ax.set_ylim([-20, 20])
    plt.show()
    plotted += 1
    if plotted >= to_plot:
        break

times = [20, 30]
for pop in pops:
    if pop.maxsize >= 10:
        for i in range(4):
            pop.resample_locations()

            ax = plt.subplot(121)
            pop.plot(ax)
            ax.set_ylim([-20, 20])
            ax.vlines(times, -20, 20, linestyle="dashed", colors=["C0", "C1"])

            ax = plt.subplot(122)
            for t in times:
                plt.hist(
                    pop.locations_at(t),
                    histtype="stepfilled",
                    orientation="horizontal",
                    bins=np.linspace(-20, 20, 41),
                    alpha=0.75,
                )
            ax.set_ylim([-20, 20])
            plt.show()
        break

for t in [1, 5, 10, 50]:
    locs = []
    for pop in pops:
        locs += pop.locations_at(t)
    print(t, len(locs), np.var(locs))
    plt.hist(locs, bins=np.linspace(-10, 10, 21), histtype="step")

for t in [1, 5, 10, 50]:
    samples = np.empty(npops)
    for i, pop in enumerate(pops):
        w = np.sum(np.exp(-((np.array(pop.locations_at(t)) - 2.0) ** 2) / 2))
        samples[i] = w
    plt.hist(samples, bins=np.arange(0.1, 5, 0.1))
    plt.show()


# +
def sample(locs, x, sigma):
    return np.sum(np.exp(-(((locs - x) / sigma) ** 2) / 2)) / (
        np.sqrt(2 * np.pi) * sigma
    )


nsamples = 100

sigma1 = 4
sigma2 = 0.25
samples1 = np.zeros((nsamples, npops))
samples2 = np.zeros((nsamples, npops))

ts = np.random.exponential(10, size=nsamples)
xs = np.random.normal(0, 10, size=nsamples)
weights = np.zeros((nsamples, npops))

for i in range(nsamples):
    for j, pop in enumerate(pops):
        locs = np.array(pop.locations_at(ts[i]))

        w1 = sample(locs, xs[i], sigma1)
        samples1[i, j] = w1

        w2 = sample(locs, xs[i], sigma2)
        samples2[i, j] = w2

    weights[i, :] = np.exp(ts[i] / 10) * np.exp((xs[i] / 10) ** 2 / 2)

# +
plt.hist(
    samples1.flatten(),
    weights=weights.flatten(),
    bins=np.arange(0.0, 10, 0.05),
    log=True,
)
plt.show()

plt.hist(
    samples2.flatten(),
    weights=weights.flatten(),
    bins=np.arange(0.0, 10, 0.05),
    log=True,
)
plt.show()

# +
m1 = np.mean(samples1.flatten() * weights.flatten())
m2 = np.mean(samples2.flatten() * weights.flatten())
v1 = np.mean(samples1.flatten() ** 2 * weights.flatten()) - m1 ** 2
v2 = np.mean(samples2.flatten() ** 2 * weights.flatten()) - m2 ** 2

scale1 = np.sqrt(v1 / m1)
scale2 = np.sqrt(v2 / m2)

k1 = m1 / scale1
k2 = m2 / scale2
print(m1, m2)
print(v1, v2)
print(scale1, scale2)
print(k1, k2)

# +
bins = np.arange(0.0, 10, 0.05)

z1 = np.random.gamma(k1, scale=scale1, size=100000)
plt.hist(
    samples1.flatten(), weights=weights.flatten(), bins=bins, log=True, density=True
)
plt.hist(z1, bins=bins, log=True, histtype="step", density=True)
plt.show()

z2 = np.random.gamma(k2, scale=scale2, size=100000)
plt.hist(
    samples2.flatten(), weights=weights.flatten(), bins=bins, log=True, density=True
)
plt.hist(z2, bins=bins, log=True, histtype="step", density=True)
# -

# ## Brownian bridge

np.random.seed(101)
n = 100
t = np.linspace(0, 1, n + 1)
dt = 1 / n
dW = np.zeros(n + 1)
dW[1:] = np.random.normal(size=n) * np.sqrt(dt)
W = np.cumsum(dW)
plt.plot(t, W, drawstyle="steps-post")

B = W - t * W[-1]
plt.plot(t, B, drawstyle="steps-post")
plt.hlines(0, 0, 1, linestyle="dashed")

# +
left = 1.0
right = 2.0

B2 = W - t * W[-1] + left + t * (right - left)
plt.plot(t, B2, drawstyle="steps-post")
plt.plot(t, left + t * (right - left), 1, linestyle="dashed", color="k")
# -

# ## Profiling

np.random.seed(100)
npops = 1000
# %prun [simulate(s=0.025, max_steps=10000) for i in range(npops)]

# +
# %%prun

nsamples = 100

sigma1 = 4
sigma2 = 0.25
samples1 = np.zeros((nsamples, npops))
samples2 = np.zeros((nsamples, npops))

ts = np.random.exponential(10, size=nsamples)
xs = np.random.normal(0, 10, size=nsamples)
weights = np.zeros((nsamples, npops))

for i in range(nsamples):
    for j, pop in enumerate(pops):
        locs = np.array(pop.locations_at(ts[i]))

        w1 = sample(locs, xs[i], sigma1)
        samples1[i, j] = w1

        w2 = sample(locs, xs[i], sigma2)
        samples2[i, j] = w2

    weights[i, :] = np.exp(ts[i] / 10) * np.exp((xs[i] / 10) ** 2 / 2)
# -
