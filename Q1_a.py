"""
EE 5393 HW1 - Problem 1(a): Discrete step-count stochastic simulation (no tau/time)

Estimation:
  Pr(C1): reach x1 >= 150
  Pr(C2): reach x2 < 10
  Pr(C3): reach x3 > 100

Starting state: (x1, x2, x3) = (110, 26, 55)
Simulation: up to K reaction events per trial (step-count stopping)
Trials: N
"""

import random
from typing import Callable, Tuple


# -- Inputs -- #

X1_0, X2_0, X3_0 = 110, 26, 55

# Rate constants
K1 = 1.0
K2 = 2.0
K3 = 3.0

K_MAX_STEPS = 50_000
N_TRIALS = 2_000


def propensities(x1: int, x2: int, x3: int) -> Tuple[float, float, float]:
    
    #Returning propensities (a1, a2, a3) for reactions R1, R2, R3 at the current state.
    if x1 >= 2 and x2 >= 1:
        a1 = K1 * (x1 * (x1 - 1) / 2.0) * x2
    else:
        a1 = 0.0

    if x1 >= 1 and x3 >= 2:
        a2 = K2 * x1 * (x3 * (x3 - 1) / 2.0)
    else:
        a2 = 0.0
    if x2 >= 1 and x3 >= 1:
        a3 = K3 * x2 * x3
    else:
        a3 = 0.0

    return a1, a2, a3


def step_once(x1: int, x2: int, x3: int) -> Tuple[int, int, int, bool]:
    """
    Perform ONE reaction event (one 'step') using propensity-weighted choice.
    Returns updated (x1, x2, x3, ok), where ok=False means no reaction could fire.
    """
    a1, a2, a3 = propensities(x1, x2, x3)
    A = a1 + a2 + a3
    if A <= 0.0:
        return x1, x2, x3, False  # stuck

    r = random.random() * A  # uniform in [0, A)

    # Choose reaction by cumulative propensities
    if r < a1:
        # R1: 2X1 + X2 -> 4X3
        x1 -= 2
        x2 -= 1
        x3 += 4
    elif r < a1 + a2:
        # R2: X1 + 2X3 -> 3X2
        x1 -= 1
        x2 += 3
        x3 -= 2
    else:
        # R3: X2 + X3 -> 2X1
        x1 += 2
        x2 -= 1
        x3 -= 1

    # Safety (should never go negative if propensities are correct)
    if x1 < 0 or x2 < 0 or x3 < 0:
        raise RuntimeError(f"Negative molecule count reached: {(x1, x2, x3)}")

    return x1, x2, x3, True


def estimate_probability(
    condition: Callable[[int, int, int], bool],
    N: int = N_TRIALS,
    K: int = K_MAX_STEPS,
) -> float:
    """
    Estimate Pr(condition) by Monte Carlo.
    One trial succeeds if condition becomes True at any step within K steps.
    """
    successes = 0

    for _ in range(N):
        x1, x2, x3 = X1_0, X2_0, X3_0

        # Check at start (usually false, but correct to include)
        if condition(x1, x2, x3):
            successes += 1
            continue

        hit = False
        for _step in range(K):
            x1, x2, x3, ok = step_once(x1, x2, x3)
            if not ok:
                break  # no reaction can fire
            if condition(x1, x2, x3):
                hit = True
                break

        if hit:
            successes += 1

    return successes / float(N)


def main() -> None:
    # Conditions
    C1 = lambda x1, x2, x3: x1 >= 150
    C2 = lambda x1, x2, x3: x2 < 10
    C3 = lambda x1, x2, x3: x3 > 100

    pr_c1 = estimate_probability(C1)
    pr_c2 = estimate_probability(C2)
    pr_c3 = estimate_probability(C3)

    print(f"Step-count simulation (N={N_TRIALS}, K={K_MAX_STEPS}, start=[110,26,55])")
    print(f"Estimated Pr(C1: x1 >= 150)  = {pr_c1:.4f}")
    print(f"Estimated Pr(C2: x2 < 10)    = {pr_c2:.4f}")
    print(f"Estimated Pr(C3: x3 > 100)   = {pr_c3:.4f}")


if __name__ == "__main__":
    main()
