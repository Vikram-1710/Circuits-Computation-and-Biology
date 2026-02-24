import random
import numpy as np

# Reaction rate constants
K1, K2, K3 = 1.0, 2.0, 3.0

# Q1(b) setup
X1_0, X2_0, X3_0 = 9, 8, 7
STEPS = 7
N_TRIALS = 10000  # you can change (e.g., 1000 / 5000 / 20000)


def propensities(x1: int, x2: int, x3: int):
    # R1: 2X1 + X2 -> 4X3
    a1 = K1 * (x1 * (x1 - 1) / 2.0) * x2 if (x1 >= 2 and x2 >= 1) else 0.0

    # R2: X1 + 2X3 -> 3X2
    a2 = K2 * x1 * (x3 * (x3 - 1) / 2.0) if (x1 >= 1 and x3 >= 2) else 0.0

    # R3: X2 + X3 -> 2X1
    a3 = K3 * x2 * x3 if (x2 >= 1 and x3 >= 1) else 0.0

    return a1, a2, a3


def step_once(x1: int, x2: int, x3: int):
    a1, a2, a3 = propensities(x1, x2, x3)
    A = a1 + a2 + a3
    if A <= 0.0:
        return x1, x2, x3, False  # no reaction can fire

    r = random.random() * A

    if r < a1:
        # R1
        x1 -= 2
        x2 -= 1
        x3 += 4
    elif r < a1 + a2:
        # R2
        x1 -= 1
        x2 += 3
        x3 -= 2
    else:
        # R3
        x1 += 2
        x2 -= 1
        x3 -= 1

    if x1 < 0 or x2 < 0 or x3 < 0:
        raise RuntimeError(f"Negative state reached: {(x1,x2,x3)}")

    return x1, x2, x3, True


def run_one_trial():
    x1, x2, x3 = X1_0, X2_0, X3_0
    for _ in range(STEPS):
        x1, x2, x3, ok = step_once(x1, x2, x3)
        if not ok:
            break  # stuck early; keep final state as-is
    return x1, x2, x3


def main():
    finals = np.array([run_one_trial() for _ in range(N_TRIALS)], dtype=float)

    means = finals.mean(axis=0)
    variances = finals.var(axis=0, ddof=0)  # population variance; use ddof=1 if you want sample variance

    print(f"Q1(b) step-count Monte Carlo: N={N_TRIALS}, steps={STEPS}, start=[9,8,7]")
    print(f"Mean:     E[X1]={means[0]:.4f}, E[X2]={means[1]:.4f}, E[X3]={means[2]:.4f}")
    print(f"Variance: Var(X1)={variances[0]:.4f}, Var(X2)={variances[1]:.4f}, Var(X3)={variances[2]:.4f}")


if __name__ == "__main__":
    main()  
