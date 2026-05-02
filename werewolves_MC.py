import numpy as np
import matplotlib.pyplot as plt


M = 4  # should be 2


class MonteCarloSolver:
    def __init__(self, T_0, delta_0) -> None:
        self.T_0 = T_0
        self.delta_0 = delta_0

    def p(self, delta, t: int) -> float:
        return (0.5 * delta + 0.5 * self.T_0 - t - 1) / (self.T_0 - 2 * t - 1)

    def monte_carlo(self, n_samples, n_time_steps):
        t_max = self.T_0 // M
        delta = np.zeros((n_samples, n_time_steps + 1))
        delta[:, 0] = self.delta_0

        for t in range(t_max):
            p = self.p(delta[:, t], t)
            u = np.random.uniform(0, 1, size=n_samples)
            delta[:, t + 1] = delta[:, t] + np.where(u < p, -2, 0)
        
        return delta[:, -1]


class ItoMonteCarloSolver:
    def __init__(self, T_0, delta_0) -> None:
        self.T_0 = T_0
        self.delta_0 = delta_0

    def p(self, delta, t: float) -> float:
        return (0.5 * delta + 0.5 * self.T_0 - t - 1) / (self.T_0 - 2 * t - 1)

    def mu(self, delta, t: float) -> float:
        return -2 * self.p(delta, t)
    
    def sigma(self, delta, t: float) -> float:
        p = self.p(delta, t)
        return np.sqrt(4 * (p * (1 - p)))

    def monte_carlo(self, n_samples, n_time_steps):
        t_max = self.T_0 // M
        dt = t_max / n_time_steps
        delta = np.zeros((n_samples, n_time_steps + 1))
        delta[:, 0] = self.delta_0

        for i in range(n_time_steps):
            t = i * dt
            dW = np.random.normal(0, np.sqrt(dt), size=n_samples)

            delta[:, i + 1] = delta[:, i] + self.mu(delta[:, i], t) * dt + self.sigma(delta[:, i], t) * dW
        
        return delta[:, -1]


class MonteCarloExperiment:
    def __init__(self) -> None:
        pass

    def run(self):
        T_0 = 16000
        delta_0 = 15600
        n_samples = 100000
        n_time_steps = 8000

        delta_samples = MonteCarloSolver(T_0, delta_0).monte_carlo(n_samples, n_time_steps)
        
        print("mean =", np.mean(delta_samples))
        print("variance =", np.var(delta_samples))

        delta_samples_Ito = ItoMonteCarloSolver(T_0, delta_0).monte_carlo(n_samples, n_time_steps)
        
        print("mean (Ito) =", np.mean(delta_samples_Ito))
        print("variance (Ito) =", np.var(delta_samples_Ito))
        
        plt.hist((delta_samples, delta_samples_Ito), bins=50, label=["Monte Carlo", "Ito"])
        plt.xlabel("delta(T_0 / {:.0f})".format(M))
        plt.ylabel("frequency")
        plt.legend()
        plt.show()


def main():
    MonteCarloExperiment().run()

if __name__ == "__main__":
    main()