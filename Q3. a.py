import random
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ==========================================
# 1. Advanced Data Structures
# ==========================================
@dataclass
class Reaction:
    """A strictly typed dataclass representing a chemical reaction rule."""
    name: str
    reactants: Dict[str, int]
    products: Dict[str, int]
    rate: float


class AdvancedGillespieCRN:
    """Production-grade stochastic simulator with full state tracking."""

    def __init__(self, reactions: List[Reaction], initial_state: Dict[str, int]):
        self.reactions = reactions
        self.state = initial_state.copy()
        self.initial_state_record = initial_state.copy()
        self.time = 0.0
        self.total_steps = 0

        # Matrix tracking for advanced plotting
        self.history_t: List[float] = [0.0]
        self.history_state: Dict[str, List[int]] = {s: [c] for s, c in initial_state.items()}

    def _get_propensities(self) -> List[float]:
        """Calculates collision probabilities using combinatorics."""
        props = []
        for rxn in self.reactions:
            a = rxn.rate
            for species, required in rxn.reactants.items():
                current = self.state.get(species, 0)
                if current < required:
                    a = 0.0
                    break
                # N-choose-K combinatorics for identical collisions
                if required == 1:
                    a *= current
                elif required == 2:
                    a *= (current * (current - 1)) / 2.0
            props.append(a)
        return props

    def simulate(self, max_steps: int = 100000) -> None:
        """Executes the continuous-time Markov chain (Gillespie algorithm)."""
        for step in range(max_steps):
            props = self._get_propensities()
            a0 = sum(props)

            # System reached stable steady state (computation finished)
            if a0 == 0.0:
                break

            # Continuous Time Math (tau)
            r1 = max(random.random(), 1e-10)  # Prevent math domain errors
            r2 = random.random()
            tau = (1.0 / a0) * math.log(1.0 / r1)
            self.time += tau

            # Reaction Roulette
            threshold = r2 * a0
            running_sum = 0.0
            chosen = self.reactions[-1]
            for i, p in enumerate(props):
                running_sum += p
                if running_sum > threshold:
                    chosen = self.reactions[i]
                    break

            # Execute State Transition
            for s, c in chosen.reactants.items():
                self.state[s] -= c
            for s, c in chosen.products.items():
                if s not in self.state:
                    self.state[s] = 0
                    self.history_state[s] = [0] * len(self.history_t)
                self.state[s] += c

            # Log step metrics
            self.total_steps += 1
            self.history_t.append(self.time)
            for s in self.history_state:
                self.history_state[s].append(self.state.get(s, 0))

    def print_comprehensive_summary(self, title: str, target_math: str) -> None:
        """Generates a detailed console report of the simulation run."""
        print(f"\n{'-' * 50}")
        print(f" EXPERIMENT: {title}")
        print(f" TARGET:     {target_math}")
        print(f"{'-' * 50}")
        print(f" Total Reaction Events : {self.total_steps:,} steps")
        print(f" Total Chemical Time   : {self.time:.5f} seconds")
        print(f"\n | Species | Initial Count | Final Count |")
        print(f" |---------|---------------|-------------|")

        # Sort keys to show outputs (Z, Y) and Inputs (X, Y) first
        all_species = sorted(list(self.history_state.keys()))
        for s in all_species:
            init_val = self.initial_state_record.get(s, 0)
            final_val = self.state.get(s, 0)
            marker = " <--" if s in ['Z', 'Y'] and final_val > 0 else ""
            print(f" | {s:<7} | {init_val:<13} | {final_val:<11} |{marker}")
        print(f"{'-' * 50}\n")


# ==========================================
# 2. Define Network Rules
# ==========================================

rules_part1 = [
    Reaction("Halve_Y", {'S1': 1, 'Y': 2}, {'S1': 1, 'Y_p': 1}, 10000),
    Reaction("Gate_S2", {'S1': 1, 'Y_p': 1}, {'S2': 1, 'Y_p': 1}, 1),
    Reaction("Burn_Y", {'S2': 1, 'Y': 1}, {'S2': 1}, 10000),
    Reaction("Copy_XtoZ", {'S2': 1, 'X': 1}, {'S2': 1, 'X_p': 1, 'Z': 1}, 10000),
    Reaction("Gate_S3", {'S2': 1}, {'S3': 1}, 1),
    Reaction("Restore_Y", {'S3': 1, 'Y_p': 1}, {'S3': 1, 'Y': 1}, 10000),
    Reaction("Restore_X", {'S3': 1, 'X_p': 1}, {'S3': 1, 'X': 1}, 10000),
    Reaction("Loop_S1", {'S3': 1}, {'S1': 1}, 1),
]

rules_part2 = [
    # Logarithm logic
    Reaction("Halve_X", {'S1': 1, 'X': 2}, {'S1': 1, 'X_p': 1}, 10000),
    Reaction("Gen_Linker", {'S1': 1, 'X_p': 1}, {'S2': 1, 'X_p': 1, 'L': 1}, 1),
    Reaction("Burn_X", {'S2': 1, 'X': 1}, {'S2': 1}, 10000),
    Reaction("Gate_S3", {'S2': 1}, {'S3': 1}, 1),
    Reaction("Restore_X", {'S3': 1, 'X_p': 1}, {'S3': 1, 'X': 1}, 10000),
    Reaction("Loop_S1", {'S3': 1}, {'S1': 1}, 1),
    # Exponentiation logic
    Reaction("Read_Linker", {'E1': 1, 'L': 1}, {'E2': 1}, 10),
    Reaction("Double_Y", {'E2': 1, 'Y': 1}, {'E2': 1, 'Y_p': 2}, 10000),
    Reaction("Gate_E3", {'E2': 1}, {'E3': 1}, 1),
    Reaction("Restore_Y", {'E3': 1, 'Y_p': 1}, {'E3': 1, 'Y': 1}, 10000),
    Reaction("Loop_E1", {'E3': 1}, {'E1': 1}, 1),
]

# ==========================================
# 3. Execution and Detailed Output
# ==========================================

sim1 = AdvancedGillespieCRN(rules_part1, {'X': 5, 'Y': 8, 'Z': 0, 'S1': 1})
sim1.simulate()
sim1.print_comprehensive_summary(title="Part 1: Repeated Addition", target_math="Z = 5 * log2(8) = 15")

sim2 = AdvancedGillespieCRN(rules_part2, {'X': 16, 'Y': 1, 'L': 0, 'S1': 1, 'E1': 1})
sim2.simulate()
sim2.print_comprehensive_summary(title="Part 2: Function Composition", target_math="Y = 2^(log2(16)) = 16")

# ==========================================
# 4. Digital Signal Split-Pane Plotting
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col', gridspec_kw={'height_ratios': [2.5, 1]})
fig.subplots_adjust(hspace=0.08)

# --- CRN 1 (Left) ---
ax_z = axes[0, 0]
ax_mech1 = axes[1, 0]

ax_z.fill_between(sim1.history_t, sim1.history_state.get('Z', [0]), step='post', color='#4A90E2', alpha=0.2)
ax_z.step(sim1.history_t, sim1.history_state.get('Z', [0]), where='post', color='#4A90E2', linewidth=2.5,
          label='Output (Z)')
ax_z.axhline(15, color='red', linestyle='--', linewidth=1.5, label='Target Z = 15')
ax_z.set_title("CRN 1: Computing $Z_{\infty} = X_0 \log_2(Y_0)$", fontsize=14, fontweight='bold', pad=15)
ax_z.set_ylabel("Output Molecules", fontsize=11)
ax_z.legend(loc='upper left', fontsize=11)
ax_z.grid(True, linestyle=':', alpha=0.6)

ax_mech1.fill_between(sim1.history_t, sim1.history_state.get('Y', [0]), step='post', color='#F5A623', alpha=0.3)
ax_mech1.step(sim1.history_t, sim1.history_state.get('Y', [0]), where='post', color='#D0021B', linewidth=2,
              label='Halving (Y)')
ax_mech1.step(sim1.history_t, sim1.history_state.get('X', [0]), where='post', color='#7ED321', linewidth=2,
              label='Constant (X)')
ax_mech1.set_xlabel("Chemical Time (s)", fontsize=12)
ax_mech1.set_ylabel("Intermediates", fontsize=11)
ax_mech1.legend(loc='upper right', fontsize=10)
ax_mech1.grid(True, linestyle=':', alpha=0.6)

# --- CRN 2 (Right) ---
ax_y = axes[0, 1]
ax_mech2 = axes[1, 1]

ax_y.fill_between(sim2.history_t, sim2.history_state.get('Y', [0]), step='post', color='#9013FE', alpha=0.2)
ax_y.step(sim2.history_t, sim2.history_state.get('Y', [0]), where='post', color='#9013FE', linewidth=2.5,
          label='Output (Y)')
ax_y.axhline(16, color='red', linestyle='--', linewidth=1.5, label='Target Y = 16')
ax_y.set_title("CRN 2: Computing $Y_{\infty} = 2^{\log_2(X_0)}$", fontsize=14, fontweight='bold', pad=15)
ax_y.legend(loc='upper left', fontsize=11)
ax_y.grid(True, linestyle=':', alpha=0.6)

ax_mech2.fill_between(sim2.history_t, sim2.history_state.get('X', [0]), step='post', color='#F5A623', alpha=0.3)
ax_mech2.step(sim2.history_t, sim2.history_state.get('X', [0]), where='post', color='#D0021B', linewidth=2,
              label='Halving (X)')
ax_mech2.step(sim2.history_t, sim2.history_state.get('L', [0]), where='post', color='#4A4A4A', linestyle='--',
              linewidth=2, label='Linker (L)')
ax_mech2.set_xlabel("Chemical Time (s)", fontsize=12)
ax_mech2.legend(loc='upper right', fontsize=10)
ax_mech2.grid(True, linestyle=':', alpha=0.6)

for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
