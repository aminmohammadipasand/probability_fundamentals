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
