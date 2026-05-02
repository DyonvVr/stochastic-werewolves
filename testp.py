import numpy as np

def p(x):
    return 0.5 * x + 0.5

def main():
    n = 10
    x = 0.5 * np.ones(n)
    dx = np.random.choice([-1, 1], size=n)

if __name__ == "__main__":
    main()