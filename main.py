import time
import math
C = 2 ** 31 - 1

class LCGEngine:
    def __init__(self, seed = None):
        if seed is None:
            seed = time.time_ns()
        self.seed = seed % C
        if self.seed == 0:
            self.seed = 1

    def set_seed(self, seed):
        self.seed = seed % C
        if self.seed == 0:
            self.seed = 1

    def next_random(self):
        self.seed = (48271 * self.seed) % C
        return self.seed / C

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
    
class BernoulliDistribution:
    def __init__(self, p, random_engine):
        self.p = p
        self.engine = random_engine

    def generate_sample(self):
        return 1 if self.engine.next_random() < self.p else 0
    
class BinomialDistribution:
    def __init__(self, n, p, random_engine):
        self.n = n
        self.p = p
        self.engine = random_engine
        self.bernoulli = BernoulliDistribution(p, random_engine)

    def generate_sample(self):
        return sum(self.bernoulli.generate_sample() for i in range(self.n))



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

class LimitTheoremsEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def normal_pdf(x):
        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (x ** 2))

    @staticmethod
    def normal_cdf_standard(z):
        lower_bound = -10.0
        if z < lower_bound:
            return 0.0
        if z > 10.0:
            return 1.0
        
        steps = 10000
        h = (z - lower_bound) / steps
        
        area = 0.5 * (LimitTheoremsEvaluator.normal_pdf(lower_bound)  + LimitTheoremsEvaluator.normal_pdf(z))
        for i in range(1, steps):
            x = lower_bound + i * h
            area += LimitTheoremsEvaluator.normal_pdf(x)
        
        return area * h

    def get_normal_probability(self, a, b, mu, sigma):
        z_a = (a - mu) / sigma
        z_b = (b - mu) / sigma
        
        return self.normal_cdf_standard(z_b) - self.normal_cdf_standard(z_a)

    def evaluate_binomial_to_normal(self, binomial_samples, n, p, a, b):
        M = len(binomial_samples)
        
        success_count = sum(1 for x in binomial_samples if a <= x <= b)
        empirical_prob = success_count / M
        
        mu = n * p
        sigma = math.sqrt(n * p * (1 - p))
        
        a_corrected = a - 0.5
        b_corrected = b + 0.5
        
        theoretical_prob = self.get_normal_probability(a_corrected, b_corrected, mu, sigma)
        
        error_pct = (abs(empirical_prob - theoretical_prob) / theoretical_prob) * 100
        
        return empirical_prob, theoretical_prob, error_pct

    def evaluate_poisson_to_normal(self, poisson_samples, lam, a, b):
        """سنجش تقریب پوآسون به نرمال با اعمال تصحیح پیوستگی"""
        M = len(poisson_samples)
        
        success_count = sum(1 for x in poisson_samples if a <= x <= b)
        empirical_prob = success_count / M
        
        mu = lam
        sigma = math.sqrt(lam)
        
        a_corrected = a - 0.5
        b_corrected = b + 0.5
        
        theoretical_prob = self.get_normal_probability(a_corrected, b_corrected, mu, sigma)
        
        error_pct = (abs(empirical_prob - theoretical_prob) / theoretical_prob) * 100
        
        return empirical_prob, theoretical_prob, error_pct


if __name__ == "__main__":
    #comparison in faze 0:
    lcg = LCGEngine()
    start_time = time.time()
    for i in range(1000000):
        lcg.next_random()
    lcg_time = time.time() - start_time
    
    xorshift = XorShiftEngine()
    start_time = time.time()
    for i in range(1000000):
        xorshift.next_random()
    xorshift_time = time.time() - start_time
    
    if lcg_time < xorshift_time:
        print("lcg is faster than xorshift")
    else:
        print("xorshift is faster than lcg")


    lcg = LCGEngine()
    engine = XorShiftEngine()
    evaluator = LimitTheoremsEvaluator()
    pois = PoissonDistribution(40, engine)
    print("\n Piosson to normal error computing:")
    poisson_data = [pois.generate_sample() for _ in range(10 ** 4)]
    lam = 40
    a_poi, b_poi = 30, 40
    emp_p_poi, theo_p_poi, err_poi = evaluator.evaluate_poisson_to_normal(poisson_data, lam, a_poi, b_poi)
    print(f"range: [{a_poi}, {b_poi}]")
    print(f"empirical probability: {emp_p_poi:.5f}")
    print(f"theoretical pobability: {theo_p_poi:.5f}")
    print(f"error precetage: {err_poi:.2f}%")
