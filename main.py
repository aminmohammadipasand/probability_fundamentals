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
    
def compare_engines():
    lcg = LCGEngine()
    start_time = time.time()
    for i in range(1000000):
        lcg.next_random()
    lcg_time = time.time() - start_time
    
    xorshift = XorShiftEngine()
    start_time = time.time()
    for i in range(100000):
        xorshift.next_random()
    xorshift_time = time.time() - start_time   

    if lcg_time < xorshift_time:
        print("lcg is faster than xorshift")
    else:
        print("xorshift is faster than lcg")
    
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

class GeometricDistribution:
    def __init__(self, p, random_engine):
        self.p = p
        self.engine = random_engine

    def generate_sample(self):
        k = 1
        while self.engine.next_random() >= self.p:
            k += 1
        return k

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
    
def run_statistical_analysis(samples, theo_mean, theo_var):

    M = len(samples)

    emp_mean = sum(samples) / M

    emp_var = sum((x - emp_mean) ** 2 for x in samples) / (M - 1) if M > 1 else 0

    mean_error = (abs(theo_mean - emp_mean) / theo_mean) * 100 if theo_mean != 0 else abs(emp_mean) * 100
    var_error = (abs(theo_var - emp_var) / theo_var) * 100 if theo_var != 0 else abs(emp_var) * 100

    print("results: ")
    print(f"emp_mean: {emp_mean:.5f}  |  theo_mean: {theo_mean:.5f}  |  error: {mean_error:.2f}%")
    print(f"emp_var: {emp_var:.5f}  |  theo_var: {theo_var:.5f}  |  error: {var_error:.2f}%")

def verify_memoryless_property(samples, s, t, is_discrete = False):

    greater_s = [x for x in samples if x > s]
    if not greater_s:
        print("No samples greater than s.")
        return
        
    prob_cond_empirical = sum(1 for x in greater_s if x > s + t) / len(greater_s)
    
    prob_t_empirical = sum(1 for x in samples if x > t) / len(samples)
    
    diff = abs(prob_cond_empirical - prob_t_empirical)
    
    print("Khasiate bi hafezegi: ")
    print(f"P(X > {s} + {t} | X > {s}) ≈ P(X > {t})")
    print(f"P(X > s+t | X > s): {prob_cond_empirical:.5f}")
    print(f"P(X > t): {prob_t_empirical:.5f}")
    print(f"deifference: {diff:.5f}")
    if diff < 0.05:
         print("Done.")
    else:
         print("You have to increase M or change s and t values.")

def main_menu():
    engine = XorShiftEngine() 
    evaluator = LimitTheoremsEvaluator()
    
    while True:
        print("═"*50)
        print("BOP project")
        print("═"*50)
        print("1. faze 0")
        print("2. faze 1 and faze 2")
        print("3. Bi hafezegi")
        print("4. Ghazayeh taghrib hady be normal")
        print("5. exit")
        print("═"*50)
        
        choice = input("Enter what you wanna do:").strip()
        
        if choice == '1':
            compare_engines()
            
        elif choice == '2':
            print("Distributions:")
            print("1.(Bernoulli)")
            print("2.(Binomial)")
            print("3.(Geometric)")
            print("4.(Poisson)")
            print("5.(Exponential)")
            print("6.(Normal)")
            dist_choice = input("Enter your choice (1-6):").strip()
            
            try:
                M = int(input("Enter the number of samples to generate (M): "))
            except ValueError:
                print("enter an integer for M")
                continue
                
            if dist_choice == '1':
                p = float(input("p (0,1): "))
                dist = BernoulliDistribution(p, engine)
                samples = [dist.generate_sample() for i in range(M)]
                run_statistical_analysis(samples, p, p * (1 - p))
                
            elif dist_choice == '2':
                n = int(input("n: "))
                p = float(input("p (0,1): "))
                dist = BinomialDistribution(n, p, engine)
                samples = [dist.generate_sample() for i in range(M)]
                run_statistical_analysis(samples, n * p, n * p * (1 - p))
                
            elif dist_choice == '3':
                p = float(input(" p (0,1): "))
                dist = GeometricDistribution(p, engine)
                samples = [dist.generate_sample() for i in range(M)]
                theo_mean = 1 / p
                theo_var = (1 - p) / (p ** 2)
                run_statistical_analysis(samples, theo_mean, theo_var)
                
            elif dist_choice == '4':
                lam = float(input("λ: "))
                dist = PoissonDistribution(lam, engine)
                samples = [dist.generate_sample() for i in range(M)]
                run_statistical_analysis(samples, lam, lam)
                
            elif dist_choice == '5':
                lam = float(input("λ: "))
                dist = ExponentialDistribution(lam, engine)
                samples = [dist.generate_sample() for i in range(M)]
                run_statistical_analysis(samples, 1 / lam, 1 / (lam ** 2))
                
            elif dist_choice == '6':
                mu = float(input("mu: "))
                sigma = float(input("sigma: "))
                dist = NormalDistribution(mu, sigma, engine)
                samples = [dist.generate_sample() for i in range(M)]
                run_statistical_analysis(samples, mu, sigma ** 2)
            else:
                print("wrong choice")
                
        elif choice == '3':
            print("Bi hafezegi: ")
            print("1. geometric")
            print("2. exponential")
            m_choice = input("enter your choice (1 or 2): ").strip()
            
            p_lam = float(input("enter p or lambda "))
            s = float(input("s: "))
            t = float(input("t: "))
            M = 100000
            
            if m_choice == '1':
                dist = GeometricDistribution(p_lam, engine)
                samples = [dist.generate_sample() for i in range(M)]
                verify_memoryless_property(samples, s, t, is_discrete=True)
            elif m_choice == '2':
                dist = ExponentialDistribution(p_lam, engine)
                samples = [dist.generate_sample() for i in range(M)]
                verify_memoryless_property(samples, s, t, is_discrete=False)
            else:
                print("wrong choice")
                
        elif choice == '4':
            print("Taghrib haddi be normal:")
            print("1. taghrib binomial be normal (n >= 50)")
            print("2. taghrib poisson be normal (lambda >= 30)")
            approx_choice = input("enter your choice (1 or 2): ").strip()
            
            M = 100000
            a = float(input("a: "))
            b = float(input("b: "))
            
            if approx_choice == '1':
                n = int(input("n: "))
                p = float(input("p: "))
                if n < 50:
                    print("n must be >= 50")
                dist = BinomialDistribution(n, p, engine)
                samples = [dist.generate_sample() for i in range(M)]
                emp_p, theo_p, err = evaluator.evaluate_binomial_to_normal(samples, n, p, a, b)
                
            elif approx_choice == '2':
                lam = float(input("lambda: "))
                if lam < 30:
                    print("lambda must be >= 30 ")
                dist = PoissonDistribution(lam, engine)
                samples = [dist.generate_sample() for i in range(M)]
                emp_p, theo_p, err = evaluator.evaluate_poisson_to_normal(samples, lam, a, b)
            else:
                print("wrong choice")
                continue
                
            print("results: ")
            print(f"[{a}, {b}]")
            print(f" emp_p: {emp_p:.5f}")
            print(f" theoretically : {theo_p:.5f}")
            print(f"percentage error: {err:.2f}%")
            
        elif choice == '5':
            print("Tnx to you for using this program.")
            break
        else:
            print("wrong choice")


if __name__ == "__main__":
    main_menu()