import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================================
# COMMON FUNCTION: f(x) = x * sin(x)
# ==============================================
def fitness_function(x):
    return x * np.sin(x)

# ==============================================
# GENETIC ALGORITHM
# ==============================================
def create_population(pop_size, x_min, x_max):
    return [random.uniform(x_min, x_max) for x in range(pop_size)]

def evaluate_population(pop):
    return [fitness_function(x) for x in pop]

def select_parents(pop, fitness):
    idx1, idx2 = random.sample(range(len(pop)), 2)
    return pop[idx1] if fitness[idx1] > fitness[idx2] else pop[idx2]

def crossover(parent1, parent2):
    alpha = random.random()
    return alpha * parent1 + (1 - alpha) * parent2

def mutate(child, x_min, x_max, mutation_rate=0.1):
    if random.random() < mutation_rate:
        child += np.random.normal(0, 1)
        child = max(min(child, x_max), x_min)
    return child

def run_ga(pop_size, generations, x_min, x_max):
    population = create_population(pop_size, x_min, x_max)
    best_history = []
    for _ in range(generations):
        fitness = evaluate_population(population)
        new_population = []
        for _ in range(pop_size):
            p1 = select_parents(population, fitness)
            p2 = select_parents(population, fitness)
            child = crossover(p1, p2)
            child = mutate(child, x_min, x_max)
            new_population.append(child)
        population = new_population
        best = population[np.argmax(evaluate_population(population))]
        best_history.append(best)
    return best_history

# ==============================================
# PARTICLE SWARM OPTIMIZATION
# ==============================================
def run_pso(num_particles, iterations, x_min, x_max):
    positions = np.random.uniform(x_min, x_max, num_particles)
    velocities = np.random.uniform(-1, 1, num_particles)
    personal_best = positions.copy()
    personal_best_val = fitness_function(personal_best)
    global_best = personal_best[np.argmax(personal_best_val)]

    history = []

    for _ in range(iterations):
        for i in range(num_particles):
            fitness = fitness_function(positions[i])
            if fitness > fitness_function(personal_best[i]):
                personal_best[i] = positions[i]
        global_best = personal_best[np.argmax(fitness_function(personal_best))]
        # Update velocities and positions
        w, c1, c2 = 0.7, 1.4, 1.4
        r1, r2 = np.random.rand(), np.random.rand()
        velocities = w * velocities + c1 * r1 * (personal_best - positions) + c2 * r2 * (global_best - positions)
        positions = positions + velocities
        positions = np.clip(positions, x_min, x_max)
        history.append(global_best)
    return history

# ==============================================
# CELLULAR AUTOMATA
# ==============================================
def run_ca(rule, steps, size):
    grid = np.zeros((steps, size), dtype=int)
    grid[0, size//2] = 1  # start with one live cell in center

    for t in range(1, steps):
        for i in range(1, size-1):
            left, center, right = grid[t-1, i-1], grid[t-1, i], grid[t-1, i+1]
            pattern = (left << 2) | (center << 1) | right
            grid[t, i] = (rule >> pattern) & 1
    return grid

# ==============================================
# DNA COMPUTING
# ==============================================
dna_complement_map = {"A": "T", "T": "A", "C": "G", "G": "C"}

def dna_complement(seq):
    return "".join(dna_complement_map.get(base, "N") for base in seq)

def dna_is_palindrome(seq):
    comp = dna_complement(seq)
    return seq == comp[::-1]

def dna_find_subsequence(seq, pattern):
    return seq.find(pattern)

# ==============================================
# QUANTUM-INSPIRED GENETIC ALGORITHM (QGA)
# ==============================================
def run_qga(pop_size, generations, chrom_length=10):
    population = np.ones((pop_size, chrom_length)) * 0.5
    best_history = []

    def measure(chrom):
        return np.array([1 if random.random() < p else 0 for p in chrom])

    def decode(bits, x_min=-10, x_max=10):
        value = int("".join(map(str, bits)), 2)
        max_val = 2**chrom_length - 1
        return x_min + (x_max - x_min) * (value / max_val)

    best_solution = None
    best_fitness = -1e9

    for _ in range(generations):
        classical_pop = [measure(ind) for ind in population]
        decoded = [decode(bits) for bits in classical_pop]
        fitness = [fitness_function(x) for x in decoded]

        gen_best = decoded[np.argmax(fitness)]
        if max(fitness) > best_fitness:
            best_fitness = max(fitness)
            best_solution = gen_best
        best_history.append(best_solution)

        best_bits = classical_pop[np.argmax(fitness)]
        for i in range(pop_size):
            for j in range(chrom_length):
                if classical_pop[i][j] != best_bits[j]:
                    population[i][j] = population[i][j] * 0.9 + best_bits[j] * 0.1

    return best_history

# ==============================================
# ARTIFICIAL IMMUNE SYSTEM (AIS) – Negative Selection
# ==============================================
def run_ais(num_self=100, num_nonself=20, num_detectors=50, threshold=0.2):
    # Generate "self" (normal data)
    self_data = np.random.normal(0, 1, (num_self, 2))
    # Generate "non-self" (anomalies)
    nonself_data = np.random.uniform(-5, 5, (num_nonself, 2))

    detectors = []
    while len(detectors) < num_detectors:
        cand = np.random.uniform(-5, 5, 2)
        if np.min(np.linalg.norm(self_data - cand, axis=1)) > threshold:
            detectors.append(cand)
    detectors = np.array(detectors)

    detected = []
    for ns in nonself_data:
        if np.min(np.linalg.norm(detectors - ns, axis=1)) < threshold:
            detected.append(ns)
    detected = np.array(detected)

    return self_data, nonself_data, detectors, detected

# ==============================================
# STREAMLIT APP
# ==============================================
st.title("🧬 Bio-Inspired Computing Playground")
st.write("Interactive demos of GA, PSO, CA, DNA, QGA, and AIS")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Genetic Algorithm", 
    "Particle Swarm Optimization", 
    "Cellular Automata", 
    "DNA Computing", 
    "Quantum-Inspired GA",
    "Artificial Immune System"
])

# ---- GA TAB ----
with tab1:
    st.subheader("Genetic Algorithm Optimization")
    pop_size = st.slider("Population Size", 10, 200, 50, key="ga_pop")
    generations = st.slider("Generations", 10, 200, 50, key="ga_gen")
    x_min, x_max = -10, 10

    if st.button("Run GA"):
        best_history = run_ga(pop_size, generations, x_min, x_max)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        x_vals = np.linspace(x_min, x_max, 500)
        y_vals = fitness_function(x_vals)
        ax[0].plot(x_vals, y_vals)
        ax[0].scatter(best_history, [fitness_function(x) for x in best_history], c="red", s=15)
        ax[0].set_title("Function Optimization")
        best_fitness = [fitness_function(x) for x in best_history]
        ax[1].plot(best_fitness, marker="o")
        ax[1].set_title("Best Fitness")
        st.pyplot(fig)

# ---- PSO TAB ----
with tab2:
    st.subheader("Particle Swarm Optimization")
    num_particles = st.slider("Particles", 5, 100, 30, key="pso_particles")
    iterations = st.slider("Iterations", 10, 200, 50, key="pso_iter")
    x_min, x_max = -10, 10

    if st.button("Run PSO"):
        history = run_pso(num_particles, iterations, x_min, x_max)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        x_vals = np.linspace(x_min, x_max, 500)
        y_vals = fitness_function(x_vals)
        ax[0].plot(x_vals, y_vals)
        ax[0].scatter(history, [fitness_function(x) for x in history], c="blue", s=15)
        ax[0].set_title("Function Optimization")
        best_fitness = [fitness_function(x) for x in history]
        ax[1].plot(best_fitness, marker="o", c="blue")
        ax[1].set_title("Best Fitness")
        st.pyplot(fig)

# ---- CA TAB ----
with tab3:
    st.subheader("Cellular Automata (1D)")
    rule = st.slider("Rule (0-255)", 0, 255, 30, key="ca_rule")
    steps = st.slider("Steps", 10, 100, 40, key="ca_steps")
    size = st.slider("Grid Size", 20, 200, 100, key="ca_size")

    if st.button("Run CA"):
        grid = run_ca(rule, steps, size)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(grid, cmap="binary")
        st.pyplot(fig)

# ---- DNA TAB ----
with tab4:
    st.subheader("DNA Computing")
    dna_seq = st.text_input("Enter DNA Sequence", "ATCGGCTA")
    pattern = st.text_input("Enter subsequence", "CG")

    if st.button("Analyze DNA"):
        complement = dna_complement(dna_seq)
        is_pal = dna_is_palindrome(dna_seq)
        pos = dna_find_subsequence(dna_seq, pattern)
        st.write(f"Complement: {complement}")
        st.write(f"Palindrome: {'Yes' if is_pal else 'No'}")
        st.write(f"Subsequence Position: {pos if pos!=-1 else 'Not Found'}")

# ---- QGA TAB ----
with tab5:
    st.subheader("Quantum-Inspired Genetic Algorithm")
    pop_size = st.slider("Population Size", 5, 100, 30, key="qga_pop")
    generations = st.slider("Generations", 10, 200, 50, key="qga_gen")
    chrom_length = st.slider("Chromosome Length", 5, 16, 10, key="qga_len")

    if st.button("Run QGA"):
        best_history = run_qga(pop_size, generations, chrom_length)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        x_vals = np.linspace(-10, 10, 500)
        y_vals = fitness_function(x_vals)
        ax[0].plot(x_vals, y_vals)
        ax[0].scatter(best_history, [fitness_function(x) for x in best_history], c="purple", s=15)
        ax[0].set_title("QGA Optimization")
        best_fitness = [fitness_function(x) for x in best_history]
        ax[1].plot(best_fitness, marker="o", c="purple")
        ax[1].set_title("Best Fitness")
        st.pyplot(fig)

# ---- AIS TAB ----
with tab6:
    st.subheader("Artificial Immune System (AIS) – Anomaly Detection")
    num_self = st.slider("Normal Samples", 20, 200, 100, key="ais_self")
    num_nonself = st.slider("Anomaly Samples", 5, 50, 20, key="ais_nonself")
    num_detectors = st.slider("Detectors", 10, 100, 50, key="ais_det")
    threshold = st.slider("Threshold", 0.05, 1.0, 0.2, key="ais_thresh")

    if st.button("Run AIS"):
        self_data, nonself_data, detectors, detected = run_ais(num_self, num_nonself, num_detectors, threshold)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self_data[:,0], self_data[:,1], c="green", label="Normal (Self)")
        ax.scatter(nonself_data[:,0], nonself_data[:,1], c="red", label="Anomalies (Non-Self)")
        ax.scatter(detectors[:,0], detectors[:,1], c="blue", marker="x", label="Detectors")
        if len(detected) > 0:
            ax.scatter(detected[:,0], detected[:,1], c="yellow", edgecolor="black", label="Detected Anomalies", s=100)
        ax.legend()
        ax.set_title("AIS Anomaly Detection")
        st.pyplot(fig)
