import numpy as np

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
        return f"({self.birth_time}--{self.death_time} : [{','.join([str(child) for child in self.children])}])"

    def die(self, death_time, death_location):
        self.death_time = death_time
        self.death_location = death_location
        self.update_latest_time(death_time)

    def reproduce(self, n_children, time, location):
        new_children = [
            Individual(self, time, location) for i in range(n_children)
        ]
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
            mu = self.birth_location + (
                self.death_location - self.birth_location
            ) * (time - self.birth_time) / (self.death_time - self.birth_time)
            sigma = np.sqrt(
                (self.death_time - time) * (time - self.birth_time) /
                (self.death_time - self.birth_time))
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
            return sum([child.descendants_at(time) for child in self.children],
                       [])
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
        ax.plot([self.birth_time, self.death_time],
                [self.birth_location, self.death_location],
                lw=0.5,
                color='k',
                **kwargs)
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
        return [
            indiv.location_at(t) for indiv in self.ancestor.descendants_at(t)
        ]

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
            f'Still alive after {max_steps} steps (t={t}). Killing remaining individuals...'
        )
        for indiv in population.living:
            indiv.die(t)
    return population
