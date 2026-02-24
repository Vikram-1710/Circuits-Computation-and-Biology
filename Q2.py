import numpy as np
import random
from math import comb
import matplotlib.pyplot as plt


def parse_initial_state(filename):
    """Reads lambda.in and returns a dictionary of starting molecule counts."""
    state = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                state[parts[0]] = int(parts[1])
    return state


def parse_reactions(filename):
    """Reads lambda.r and extracts reactants, products, and rate constants."""
    reactions = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split(':')

            reactants = {}
            r_tokens = parts[0].split()
            for i in range(0, len(r_tokens), 2):
                reactants[r_tokens[i]] = int(r_tokens[i + 1])

            products = {}
            p_tokens = parts[1].split()
            for i in range(0, len(p_tokens), 2):
                products[p_tokens[i]] = int(p_tokens[i + 1])

            rate = float(parts[2].strip())

            reactions.append({'reactants': reactants, 'products': products, 'rate': rate})
    return reactions


def calculate_propensities(state, reactions):
    """Calculates the likelihood of each reaction based on current molecules."""
    propensities = []
    for rxn in reactions:
        a = rxn['rate']
        for species, count in rxn['reactants'].items():
            current_amount = state.get(species, 0)
            if current_amount < count:
                a = 0
                break
            a *= comb(current_amount, count)
        propensities.append(a)
    return propensities


def run_lambda_sim(initial_state, reactions, max_steps=100000):
    """Runs a single simulation until a decision is made."""
    state = initial_state.copy()

    for step in range(max_steps):
        if state.get('cI2', 0) > 145:
            return 'stealth'
        if state.get('Cro2', 0) > 55:
            return 'hijack'

        props = calculate_propensities(state, reactions)
        a0 = sum(props)

        if a0 == 0:
            break

        r2 = random.random()
        threshold = r2 * a0
        running_sum = 0.0
        rxn_idx = -1

        for i, p in enumerate(props):
            running_sum += p
            if running_sum > threshold:
                rxn_idx = i
                break

        chosen_rxn = reactions[rxn_idx]
        for species, count in chosen_rxn['reactants'].items():
            state[species] -= count
        for species, count in chosen_rxn['products'].items():
            state[species] = state.get(species, 0) + count

    return 'timeout'


# --- Main Execution ---
base_state = parse_initial_state('lambda.in')
reactions = parse_reactions('lambda.r')

trials = 200  # Number of simulations per MOI

# Lists to store data for our graph
mois = list(range(1, 11))
stealth_probs = []
hijack_probs = []

print("MOI | Prob (Stealth) | Prob (Hijack) | % Stealth | % Hijack")
print("-" * 59)

for moi in mois:
    stealth_count = 0
    hijack_count = 0

    for _ in range(trials):
        current_state = base_state.copy()
        current_state['MOI'] = moi

        result = run_lambda_sim(current_state, reactions)

        if result == 'stealth':
            stealth_count += 1
        elif result == 'hijack':
            hijack_count += 1

    # Calculate probabilities (0 to 1)
    p_stealth = stealth_count / trials
    p_hijack = hijack_count / trials

    # Store for graphing
    stealth_probs.append(p_stealth)
    hijack_probs.append(p_hijack)

    # Print formatted row
    print(f"{moi:3} | {p_stealth:14.3f} | {p_hijack:13.3f} | {p_stealth * 100:8.1f}% | {p_hijack * 100:7.1f}%")

# --- Plotting the Results ---
plt.figure(figsize=(10, 6))
plt.plot(mois, stealth_probs, marker='o', color='blue', label='Stealth Mode (cI2 > 145)', linewidth=2)
plt.plot(mois, hijack_probs, marker='s', color='red', label='Hijack Mode (Cro2 > 55)', linewidth=2)

plt.title('Probability of Lambda Phage Survival Strategy vs. MOI', fontsize=14)
plt.xlabel('Multiplicity of Infection (MOI)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.xticks(mois)  # Ensure all MOI values show on the x-axis
plt.ylim(-0.05, 1.05)  # Keep the y-axis strictly between 0 and 1
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()