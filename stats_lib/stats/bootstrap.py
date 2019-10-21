import numpy as np


class Bootstrapper():
    def __init__(self, N, num_replicates=5000, seed=42):
        np.random.seed(seed)

        self.N = N
        self.num_replicates = num_replicates
        self.replicates = np.random.choice(self.N,
                                           size=(self.num_replicates, self.N),
                                           replace=True)

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter < self.num_replicates:
            replicate = self.replicates[self.counter]
            self.counter += 1
            return replicate
        else:
            raise StopIteration

    def __len__(self):
        return self.num_replicates
