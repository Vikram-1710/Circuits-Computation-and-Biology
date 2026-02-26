import random
import math
import matplotlib.pyplot as plt

class GillespieCRN:
    def __init__(self, reactions, initial_state):
        self.reactions = reactions
        self.state = initial_state.copy()
        self.time = 0.0

        # History tracking for plotting
        self.history_t = [0.0]
        self.history_state = {species: [count] for species, count in initial_state.items()}

    def calculate_propensities(self):
        props = []
        for rxn in self.reactions:
            a = rxn['rate']
            for species, required in rxn['reactants'].items():
                current = self.state.get(species, 0)
                if current < required:
                    a = 0
                    break
                # Combinatorics for identical reactants colliding
                if required == 1:
                    a *= current
                elif required == 2:
                    a *= (current * (current - 1)) / 2.0
            props.append(a)
        return props

    def simulate(self, max_steps=50000, target_steady_state=None):
        for step in range(max_steps):
            props = self.calculate_propensities()
            a0 = sum(props)

            # Stop if system is deadlocked, OR if computation is done
            if a0 == 0:
                break
            if target_steady_state and self.state.get(target_steady_state, 1) < 2 and a0 == props[0]:
                break

            # Exact Gillespie time-step math
            r1 = random.random()
            if r1 == 0: r1 = 1e-10  # Safety against log(0)
            r2 = random.random()
            tau = (1.0 / a0) * math.log(1.0 / r1)
            self.time += tau

            # Choose reaction
            threshold = r2 * a0
            running_sum = 0.0
            chosen = self.reactions[-1]

            for i, p in enumerate(props):
                running_sum += p
                if running_sum > threshold:
                    chosen = self.reactions[i]
                    break

            # Update State
            for s, c in chosen['reactants'].items():
                self.state[s] -= c
            for s, c in chosen['products'].items():
                if s not in self.state:
                    self.state[s] = 0
                    self.history_state[s] = [0] * len(self.history_t)
                self.state[s] += c

            # Record History
            self.history_t.append(self.time)
            for s in self.history_state:
                self.history_state[s].append(self.state.get(s, 0))


# ==========================================
# Part 1: Z = X * log2(Y)
# ==========================================
print("--- Part 1: Z = X * log2(Y) ---")
print("Expected Math: 5 * log2(8) = 15")

crn1_rules = [
    {'reactants': {'b': 1}, 'products': {'a': 1, 'b': 1}, 'rate': 0.1},
    {'reactants': {'a': 1, 'Y': 2}, 'products': {'c': 1, 'Y_p': 1, 'a': 1}, 'rate': 10000},
    {'reactants': {'c': 2}, 'products': {'c': 1}, 'rate': 10000},
    {'reactants': {'a': 1}, 'products': {}, 'rate': 1000},
    {'reactants': {'c': 1, 'X': 1}, 'products': {'c': 1, 'X_p': 1, 'Z': 1}, 'rate': 10000},
    {'reactants': {'c': 1}, 'products': {}, 'rate': 10},
    {'reactants': {'X_p': 1}, 'products': {'X': 1}, 'rate': 100},
    {'reactants': {'Y_p': 1}, 'products': {'Y': 1}, 'rate': 100}
]

sim1 = GillespieCRN(crn1_rules, {'X': 5, 'Y': 8, 'Z': 0, 'b': 1})
sim1.simulate(target_steady_state='Y')
print(f"CRN Result: Z = {sim1.state.get('Z', 0)}\n")


# ==========================================
# Part 2: Y = 2^(log2(X))
# ==========================================
print("--- Part 2: Y = 2^(log2(X)) ---")
print("Expected Math: 2^(log2(16)) = 16")

crn2_rules = [
    # Logarithm Module (Computes L = log2(X))
    {'reactants': {'b': 1}, 'products': {'a1': 1, 'b': 1}, 'rate': 0.1},
    {'reactants': {'a1': 1, 'X': 2}, 'products': {'c': 1, 'X_p': 1, 'a1': 1}, 'rate': 10000},
    {'reactants': {'c': 2}, 'products': {'c': 1}, 'rate': 10000},
    {'reactants': {'a1': 1}, 'products': {}, 'rate': 1000},
    {'reactants': {'c': 1}, 'products': {'L': 1}, 'rate': 10},
    {'reactants': {'X_p': 1}, 'products': {'X': 1}, 'rate': 100},

    # Exponentiation Module (Consumes L to double Y)
    {'reactants': {'L': 1}, 'products': {'a2': 1}, 'rate': 100},
    {'reactants': {'a2': 1, 'Y': 1}, 'products': {'a2': 1, 'Y_p': 2}, 'rate': 10000},
    {'reactants': {'a2': 1}, 'products': {}, 'rate': 1000},
    {'reactants': {'Y_p': 1}, 'products': {'Y': 1}, 'rate': 100}
]

sim2 = GillespieCRN(crn2_rules, {'X': 16, 'Y': 1, 'L': 0, 'b': 1})
sim2.simulate(target_steady_state='X')
print(f"CRN Result: Y = {sim2.state.get('Y', 0)}\n")


# ==========================================
# Unique "Digital Signal" Split-Pane Plotting
# ==========================================
# Create a 2x2 grid. Top row is tall (Output), bottom row is short (Mechanics)
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col', gridspec_kw={'height_ratios': [2.5, 1]})
fig.subplots_adjust(hspace=0.08) # Bring top and bottom plots tightly together

# --- CRN 1: Z = X * log2(Y) (Left Column) ---
ax_z = axes[0, 0]       # Top Left: Final Output
ax_mech1 = axes[1, 0]   # Bottom Left: Intermediates

# Top Left Plot (Z Output)
ax_z.fill_between(sim1.history_t, sim1.history_state['Z'], step='post', color='#4A90E2', alpha=0.2)
ax_z.step(sim1.history_t, sim1.history_state['Z'], where='post', color='#4A90E2', linewidth=2.5, label='Output (Z)')
ax_z.axhline(15, color='red', linestyle='--', linewidth=1.5, label='Target Z = 15')
ax_z.set_title("CRN 1: Computing $Z_{\infty} = X_0 \log_2(Y_0)$", fontsize=14, fontweight='bold', pad=15)
ax_z.set_ylabel("Output Molecules", fontsize=11)
ax_z.legend(loc='upper left', fontsize=11)
ax_z.grid(True, linestyle=':', alpha=0.6)

# Bottom Left Plot (X, Y Mechanics)
ax_mech1.fill_between(sim1.history_t, sim1.history_state['Y'], step='post', color='#F5A623', alpha=0.3)
ax_mech1.step(sim1.history_t, sim1.history_state['Y'], where='post', color='#D0021B', linewidth=2, label='Halving (Y)')
ax_mech1.step(sim1.history_t, sim1.history_state['X'], where='post', color='#7ED321', linewidth=2, label='Constant (X)')
ax_mech1.set_xlabel("Chemical Time (s)", fontsize=12)
ax_mech1.set_ylabel("Intermediates", fontsize=11)
ax_mech1.legend(loc='upper right', fontsize=10)
ax_mech1.grid(True, linestyle=':', alpha=0.6)


# --- CRN 2: Y = 2^(log2(X)) (Right Column) ---
ax_y = axes[0, 1]       # Top Right: Final Output
ax_mech2 = axes[1, 1]   # Bottom Right: Intermediates

# Top Right Plot (Y Output)
ax_y.fill_between(sim2.history_t, sim2.history_state['Y'], step='post', color='#9013FE', alpha=0.2)
ax_y.step(sim2.history_t, sim2.history_state['Y'], where='post', color='#9013FE', linewidth=2.5, label='Output (Y)')
ax_y.axhline(16, color='red', linestyle='--', linewidth=1.5, label='Target Y = 16')
ax_y.set_title("CRN 2: Computing $Y_{\infty} = 2^{\log_2(X_0)}$", fontsize=14, fontweight='bold', pad=15)
ax_y.legend(loc='upper left', fontsize=11)
ax_y.grid(True, linestyle=':', alpha=0.6)

# Bottom Right Plot (X, L Mechanics)
ax_mech2.fill_between(sim2.history_t, sim2.history_state['X'], step='post', color='#F5A623', alpha=0.3)
ax_mech2.step(sim2.history_t, sim2.history_state['X'], where='post', color='#D0021B', linewidth=2, label='Halving (X)')
ax_mech2.step(sim2.history_t, sim2.history_state['L'], where='post', color='#4A4A4A', linestyle='--', linewidth=2, label='Linker (L)')
ax_mech2.set_xlabel("Chemical Time (s)", fontsize=12)
ax_mech2.legend(loc='upper right', fontsize=10)
ax_mech2.grid(True, linestyle=':', alpha=0.6)

# Clean up borders
for ax in axes.flat:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()