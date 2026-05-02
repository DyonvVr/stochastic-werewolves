import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class WerewolvesDPSolver():
    def __init__(self, T_0, W_0) -> None:
        self.W_0 = W_0
        self.T_0 = T_0
        self.delta_0: int = T_0 - 2 * W_0
        self.L: int = self.delta_0 // 2 + 1  # delta takes on values in {0, 2, 4, ..., delta_0}
        self.t_max = T_0 // 2
        self.P = np.zeros((self.L, self.t_max))
        self.P[self.delta_0 // 2, 0] = 1

    def p(self, delta: int, t: int) -> float:
        return (0.5 * delta + 0.5 * self.T_0 - t - 1) / (self.T_0 - 2 * t - 1)
    
    def populate_P(self):
        for t in range(0, self.t_max - 1):
            for i in range(self.L):
                if not (self.P[i, t] == 0 or i == 0 or 2 * i == self.T_0 - 2 * t):
                    self.P[i, t + 1] += self.P[i, t] * (1 - self.p(2 * i, t))
                    self.P[i - 1, t + 1] +=  self.P[i, t] * self.p(2 * i, t)
    
    def solve(self) -> float:
        self.populate_P()
        win_prob_wolves = np.sum(self.P[0, :])
        return win_prob_wolves


class WerewolvesDPFairGameFinder:
    def __init__(self, T_0) -> None:
        self.T_0 = T_0

    def find_fair_game(self) -> tuple[int, float]:
        possible_W_0s = list(range(1, self.T_0 // 2))
        p_wins = np.zeros(len(possible_W_0s) + 2)
        p_wins[-1] = 1

        while True:
            i = len(possible_W_0s) // 2

            try:
                W_0 = possible_W_0s[i]
            
                if p_wins[W_0] == 0:
                    p_wins[W_0] = WerewolvesDPSolver(self.T_0, W_0).solve()
                if p_wins[W_0 - 1] == 0 and W_0 - 1 != 0:
                    p_wins[W_0 - 1] = WerewolvesDPSolver(self.T_0, W_0 - 1).solve()
                if p_wins[W_0 + 1] == 0:
                    p_wins[W_0 + 1] = WerewolvesDPSolver(self.T_0, W_0 + 1).solve()
                
                if np.abs(p_wins[W_0] - 0.5) <= np.abs(p_wins[W_0 - 1] - 0.5) and \
                np.abs(p_wins[W_0] - 0.5) < np.abs(p_wins[W_0 + 1] - 0.5):
                    return W_0, float(p_wins[W_0])
                
                if np.abs(p_wins[W_0] - 0.5) > np.abs(p_wins[W_0 - 1] - 0.5):
                    # smaller W_0 is closer to fair game
                    possible_W_0s = list(range(possible_W_0s[0], W_0))
                
                if np.abs(p_wins[W_0] - 0.5) > np.abs(p_wins[W_0 + 1] - 0.5):
                    # larger W_0 is closer to fair game
                    possible_W_0s = list(range(W_0 + 1, possible_W_0s[-1] + 1))
            
            except IndexError:
                return -1, -1


class DPExperiment():
    def __init__(self) -> None:
        pass

    def run(self, plot_only=False):
        if not plot_only:
            T_0s = list(range(4, 700, 2))
            res = []

            for T_0 in T_0s:
                print(T_0, end="\r")
                fair_w, win_p = WerewolvesDPFairGameFinder(T_0).find_fair_game()
                if fair_w != -1:
                    res.append([T_0, fair_w, win_p])
            
            res = np.array(res)
            np.savetxt("res_DP.csv", res, delimiter=",")

        res = pd.read_csv("res_DP.csv", sep=",", header=None).values
        w_max = np.max(res[:, 1])

        fig, (ax1, ax2) = plt.subplots(1, 2)

        colour = "tab:blue"
        ax1.set_xlabel("Aantal spelers")
        ax1.set_ylabel("Eerlijk aantal wolven")
        ax1.set_yticks(np.arange(0, w_max + 1))
        ax1.plot(res[:, 0], res[:, 1], color=colour)
        
        colour = "tab:orange"
        ax2.set_xlabel("Aantal spelers")
        ax2.set_ylabel("Winkans wolven")
        ax2.set_ylim(0, 1)
        ax2.set_yticks(np.array([0, 0.5, 1]))
        ax2.plot(res[:, 0], res[:, 2], color=colour)

        plt.show()
        plt.savefig("res.png")
    

def main() -> None:
    DPExperiment().run()

if __name__ == "__main__":
    main()