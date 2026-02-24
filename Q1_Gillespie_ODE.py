import numpy as np
import matplotlib.pyplot as plt
import random


# -------------------------------
# Reaction rate constants
# -------------------------------
k1 = 1.0
k2 = 2.0
k3 = 3.0


# -------------------------------
# Stoichiometry matrix
# rows correspond to reactions
# -------------------------------
stoich = np.array([
    [-2, -1,  4],   # R1
    [-1,  3, -2],   # R2
    [ 2, -1, -1]    # R3
])


# -------------------------------
# Propensity calculation
# -------------------------------
def compute_propensities(state):
    x1, x2, x3 = state

    a1 = k1 * (x1*(x1-1)/2) * x2 if x1 >= 2 and x2 >= 1 else 0.0
    a2 = k2 * x1 * (x3*(x3-1)/2) if x1 >= 1 and x3 >= 2 else 0.0
    a3 = k3 * x2 * x3 if x2 >= 1 and x3 >= 1 else 0.0

    return np.array([a1, a2, a3])


# -------------------------------
# Gillespie SSA (continuous time)
# -------------------------------
def run_gillespie(initial_state, t_final, seed=10):

    random.seed(seed)

    state = np.array(initial_state, dtype=int)
    t = 0.0

    times = [t]
    states = [state.copy()]

    while t < t_final:

        a = compute_propensities(state)
        a0 = np.sum(a)

        if a0 == 0:
            break

        # time to next reaction
        r1 = random.random()
        tau = -np.log(r1) / a0
        t_next = t + tau

        if t_next > t_final:
            break

        # determine which reaction fires
        r2 = random.random() * a0
        cumulative = np.cumsum(a)
        reaction_index = np.searchsorted(cumulative, r2)

        # update state
        state = state + stoich[reaction_index]

        if np.any(state < 0):
            raise ValueError("Negative molecule count encountered")

        t = t_next
        times.append(t)
        states.append(state.copy())

    return np.array(times), np.array(states)


# -------------------------------
# ODE system (mean-field model)
# -------------------------------
def ode_system(state):

    x1, x2, x3 = state

    r1 = k1 * (x1*(x1-1)/2) * x2
    r2 = k2 * x1 * (x3*(x3-1)/2)
    r3 = k3 * x2 * x3

    dx1 = -2*r1 - r2 + 2*r3
    dx2 = -r1 + 3*r2 - r3
    dx3 =  4*r1 - 2*r2 - r3

    return np.array([dx1, dx2, dx3])


# -------------------------------
# RK4 integrator
# -------------------------------
def rk4_solver(f, y0, t_grid):

    solution = np.zeros((len(t_grid), len(y0)))
    solution[0] = y0

    for i in range(1, len(t_grid)):

        dt = t_grid[i] - t_grid[i-1]
        y = solution[i-1]

        k1 = f(y)
        k2 = f(y + dt*k1/2)
        k3 = f(y + dt*k2/2)
        k4 = f(y + dt*k3)

        solution[i] = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    return solution


# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":

    initial_state = np.array([110, 26, 55])
    t_max = 0.01

    # Run ODE
    t_ode = np.linspace(0, t_max, 1500)
    ode_solution = rk4_solver(ode_system, initial_state, t_ode)

    # Run Gillespie
    t_stoch, stoch_solution = run_gillespie(initial_state, t_max, seed=15)

    # -------------------------------
    # Plotting
    # -------------------------------
    plt.figure(figsize=(11,6))

    labels = ["X1", "X2", "X3"]

    for i in range(3):
        plt.plot(t_ode, ode_solution[:, i], linestyle="--",
                 linewidth=2, label=f"{labels[i]} (ODE)")

    for i in range(3):
        plt.step(t_stoch, stoch_solution[:, i],
                 where="post", alpha=0.8,
                 label=f"{labels[i]} (Gillespie)")

    plt.xlabel("Time")
    plt.ylabel("Molecule Count")
    plt.title("Deterministic vs Stochastic Dynamics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()