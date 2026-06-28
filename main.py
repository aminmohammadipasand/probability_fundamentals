import time
import math
C = 2 ** 31 - 1

class XorShiftEngine:
    def __init__(self, seed=time.time_ns()):
        self.seed = seed % C
    
    def set_seed(self, seed):
        self.seed = seed % C
    
    def next_random(self):
        x = self.seed
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17)
        x ^= (x << 5) & 0xFFFFFFFF
        self.seed = x
        return x / 0xFFFFFFFF

class PoissonDistribution:
    def __init__(self, lam, random_engine):
        self.lam = lam
        self.engine = random_engine
        
    def generate_sample(self):
        l = math.exp(-self.lam)
        k = 0
        p = 1.0
        
        while p > l:
            k += 1
            u = self.engine.next_random() 
            p *= u
            
        return k - 1

class ExponentialDistribution:
    def __init__(self, lam, random_engine):
        self.lam = lam
        self.engine = random_engine

    def generate_sample(self):
        u = self.engine.next_random()       
        while u == 0:
            u = self.engine.next_float()
        
        return -math.log(u) / self.lam

import math

class NormalDistribution:
    def __init__(self, mu, sigma, random_engine):
        self.mu = mu
        self.sigma = sigma
        self.engine = random_engine
        self._next_gauge = None

    def generate_sample(self):
        if self._next_gauge is not None:
            z = self._next_gauge
            self._next_gauge = None
            return self.mu + z * self.sigma

        u1 = self.engine.next_float()
        u2 = self.engine.next_float()

        while u1 == 0:
            u1 = self.engine.next_float()

        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2

        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)

        self._next_gauge = z1

        return self.mu + z0 * self.sigma
