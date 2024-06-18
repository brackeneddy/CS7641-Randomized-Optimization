import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import time

# Load Wholesale Customers dataset from UCI
wholesale_customers_data = fetch_ucirepo(id=292)
wholesale_customers = wholesale_customers_data.data.features

# Preprocess the dataset
wholesale_customers["Total Spend"] = wholesale_customers[
    ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
].sum(axis=1)
wholesale_customers["Category"] = pd.qcut(
    wholesale_customers["Total Spend"], q=3, labels=["Low", "Medium", "High"]
)

label_encode_wholesale = LabelEncoder()
wholesale_customers["Category"] = label_encode_wholesale.fit_transform(
    wholesale_customers["Category"]
)

X = wholesale_customers.drop(columns=["Total Spend", "Category"])
y = wholesale_customers["Category"]

scaler_wholesale = StandardScaler()
X_scaled = scaler_wholesale.fit_transform(X)

# Use the full dataset for analysis
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define optimization problem sizes
problem_sizes = list(range(10, 101, 10))

# Containers for results
times_six_peaks = {'RHC': [], 'SA': [], 'GA': []}
times_cont_peaks = {'RHC': [], 'SA': [], 'GA': [], 'MIMIC': []}
fitness_six_peaks = {'RHC': [], 'SA': [], 'GA': []}
fitness_cont_peaks = {'RHC': [], 'SA': [], 'GA': [], 'MIMIC': []}

# Function to solve optimization problems and measure time
def solve_problem(problem, use_mimic=False):
    start_time = time.time()
    best_state_rhc, best_fitness_rhc, rhc_curve = mlrose.random_hill_climb(problem, max_attempts=100, max_iters=1000,
                                                                           restarts=10, curve=True)
    rhc_time = time.time() - start_time

    start_time = time.time()
    best_state_sa, best_fitness_sa, sa_curve = mlrose.simulated_annealing(problem, schedule=mlrose.GeomDecay(),
                                                                          max_attempts=100, max_iters=1000, curve=True)
    sa_time = time.time() - start_time

    start_time = time.time()
    best_state_ga, best_fitness_ga, ga_curve = mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1,
                                                                  max_attempts=100, max_iters=1000, curve=True)
    ga_time = time.time() - start_time

    if use_mimic:
        start_time = time.time()
        best_state_mimic, best_fitness_mimic, mimic_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2,
                                                                         max_attempts=100, max_iters=1000, curve=True)
        mimic_time = time.time() - start_time
        return rhc_curve, sa_curve, ga_curve, mimic_curve, rhc_time, sa_time, ga_time, mimic_time

    return rhc_curve, sa_curve, ga_curve, rhc_time, sa_time, ga_time

# Solve problems for different sizes
for size in problem_sizes:
    # Six Peaks
    fitness_six_peaks_fn = mlrose.SixPeaks(t_pct=0.1)
    problem_six_peaks = mlrose.DiscreteOpt(length=size, fitness_fn=fitness_six_peaks_fn, maximize=True, max_val=2)
    six_peaks_rhc_curve, six_peaks_sa_curve, six_peaks_ga_curve, rhc_time, sa_time, ga_time = solve_problem(
        problem_six_peaks)
    times_six_peaks['RHC'].append(rhc_time)
    times_six_peaks['SA'].append(sa_time)
    times_six_peaks['GA'].append(ga_time)
    fitness_six_peaks['RHC'].append(six_peaks_rhc_curve[-1, 0])
    fitness_six_peaks['SA'].append(six_peaks_sa_curve[-1, 0])
    fitness_six_peaks['GA'].append(six_peaks_ga_curve[-1, 0])
    print(
        f"Six Peaks (Size: {size}) - RHC: {six_peaks_rhc_curve[-1, 0]}, SA: {six_peaks_sa_curve[-1, 0]}, GA: {six_peaks_ga_curve[-1, 0]}")

    # Continuous Peaks
    fitness_cont_peaks_fn = mlrose.ContinuousPeaks(t_pct=0.1)
    problem_cont_peaks = mlrose.DiscreteOpt(length=size, fitness_fn=fitness_cont_peaks_fn, maximize=True, max_val=2)
    cont_peaks_rhc_curve, cont_peaks_sa_curve, cont_peaks_ga_curve, cont_peaks_mimic_curve, rhc_time, sa_time, ga_time, mimic_time = solve_problem(
        problem_cont_peaks, use_mimic=True)
    times_cont_peaks['RHC'].append(rhc_time)
    times_cont_peaks['SA'].append(sa_time)
    times_cont_peaks['GA'].append(ga_time)
    times_cont_peaks['MIMIC'].append(mimic_time)
    fitness_cont_peaks['RHC'].append(cont_peaks_rhc_curve[-1, 0])
    fitness_cont_peaks['SA'].append(cont_peaks_sa_curve[-1, 0])
    fitness_cont_peaks['GA'].append(cont_peaks_ga_curve[-1, 0])
    fitness_cont_peaks['MIMIC'].append(cont_peaks_mimic_curve[-1, 0])
    print(
        f"Continuous Peaks (Size: {size}) - RHC: {cont_peaks_rhc_curve[-1, 0]}, SA: {cont_peaks_sa_curve[-1, 0]}, GA: {cont_peaks_ga_curve[-1, 0]}, MIMIC: {cont_peaks_mimic_curve[-1, 0]}")

# Calculate the number of weights for the neural network
num_features = X_train.shape[1]
num_hidden_nodes = 10  # Adjust this to match A1.py neural network architecture
num_output_nodes = len(np.unique(y_train))
num_weights = num_features * num_hidden_nodes + num_hidden_nodes * num_output_nodes

# Define a custom fitness function for NN weight optimization based on A1.py
def custom_fitness_function(weights):
    input_to_hidden_weights = weights[:num_features * num_hidden_nodes].reshape((num_features, num_hidden_nodes))
    hidden_to_output_weights = weights[num_features * num_hidden_nodes:].reshape((num_hidden_nodes, num_output_nodes))

    nn = MLPClassifier(hidden_layer_sizes=(num_hidden_nodes,), max_iter=1, activation='relu', solver='sgd', alpha=0.001)
    nn.coefs_ = [input_to_hidden_weights, hidden_to_output_weights]
    nn.intercepts_ = [np.zeros(num_hidden_nodes), np.zeros(num_output_nodes)]

    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    fitness = accuracy_score(y_test, y_pred)  # Positive fitness for accuracy
    return fitness

# Define an optimization problem object for the neural network weights using the custom fitness function
fitness_nn = mlrose.CustomFitness(custom_fitness_function)
problem_nn = mlrose.ContinuousOpt(length=num_weights, fitness_fn=fitness_nn, maximize=True, min_val=-1, max_val=1)

# Function to solve the neural network weight optimization problem using different algorithms
def solve_nn_problem(problem, algorithm_name):
    if algorithm_name == 'RHC':
        best_state, best_fitness, curve = mlrose.random_hill_climb(problem, max_attempts=200, max_iters=2000,
                                                                   restarts=20, curve=True)
    elif algorithm_name == 'SA':
        best_state, best_fitness, curve = mlrose.simulated_annealing(problem, schedule=mlrose.GeomDecay(),
                                                                     max_attempts=200, max_iters=2000, curve=True)
    elif algorithm_name == 'GA':
        best_state, best_fitness, curve = mlrose.genetic_alg(problem, pop_size=300, mutation_prob=0.2, max_attempts=200,
                                                             max_iters=2000, curve=True)
    else:
        raise ValueError("Algorithm not recognized. Choose from 'RHC', 'SA', 'GA'.")

    print(f'Neural Network Weight Optimization - {algorithm_name}: Best fitness: {best_fitness}')
    return best_state, best_fitness, curve

# Solve the neural network weight optimization problem using different algorithms
nn_rhc_state, nn_rhc_fitness, nn_rhc_curve = solve_nn_problem(problem_nn, 'RHC')
nn_sa_state, nn_sa_fitness, nn_sa_curve = solve_nn_problem(problem_nn, 'SA')
nn_ga_state, nn_ga_fitness, nn_ga_curve = solve_nn_problem(problem_nn, 'GA')

# Plotting fitness vs. iterations for each algorithm
def plot_fitness_iterations(curves, title, algorithms, colors):
    plt.figure(figsize=(10, 6))
    for curve, algo, color in zip(curves, algorithms, colors):
        plt.plot(range(len(curve)), curve[:, 0], label=algo,
                 color=color)  # Use the first column which is the fitness value
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

# Six Peaks Problem
plot_fitness_iterations(
    [six_peaks_rhc_curve, six_peaks_sa_curve, six_peaks_ga_curve],
    'Six Peaks Problem',
    ['RHC', 'SA', 'GA'],
    ['blue', 'green', 'red']
)

# Continuous Peaks Problem
plot_fitness_iterations(
    [cont_peaks_rhc_curve, cont_peaks_sa_curve, cont_peaks_ga_curve, cont_peaks_mimic_curve],
    'Continuous Peaks Problem',
    ['RHC', 'SA', 'GA', 'MIMIC'],
    ['blue', 'green', 'red', 'purple']
)

# Neural Network Weight Optimization Problem
plot_fitness_iterations(
    [nn_rhc_curve, nn_sa_curve, nn_ga_curve],
    'Neural Network Weight Optimization',
    ['RHC', 'SA', 'GA'],
    ['blue', 'green', 'red']
)

# Plot Time Taken vs Problem Size for Six Peaks
plt.figure(figsize=(10, 6))
for algo in times_six_peaks.keys():
    plt.plot(problem_sizes, times_six_peaks[algo], label=f'{algo} - Six Peaks', marker='o')
plt.title('Time Taken vs Problem Size (Six Peaks)')
plt.xlabel('Problem Size')
plt.ylabel('Time Taken (seconds)')
plt.legend()
plt.show()

# Plot Time Taken vs Problem Size for Continuous Peaks
plt.figure(figsize=(10, 6))
for algo in times_cont_peaks.keys():
    plt.plot(problem_sizes, times_cont_peaks[algo], label=f'{algo} - Continuous Peaks', marker='x')
plt.title('Time Taken vs Problem Size (Continuous Peaks)')
plt.xlabel('Problem Size')
plt.ylabel('Time Taken (seconds)')
plt.legend()
plt.show()

# Plot Best Fitness vs Problem Size for Six Peaks
plt.figure(figsize=(10, 6))
for algo in fitness_six_peaks.keys():
    plt.plot(problem_sizes, fitness_six_peaks[algo], label=f'{algo} - Six Peaks', marker='o')
plt.title('Best Fitness vs Problem Size (Six Peaks)')
plt.xlabel('Problem Size')
plt.ylabel('Best Fitness')
plt.legend()
plt.show()

# Plot Best Fitness vs Problem Size for Continuous Peaks
plt.figure(figsize=(10, 6))
for algo in fitness_cont_peaks.keys():
    plt.plot(problem_sizes, fitness_cont_peaks[algo], label=f'{algo} - Continuous Peaks', marker='x')
plt.title('Best Fitness vs Problem Size (Continuous Peaks)')
plt.xlabel('Problem Size')
plt.ylabel('Best Fitness')
plt.legend()
plt.show()

# Bar chart for best fitness values for peak problems
best_fitness_values_peaks = {
    'Algorithm': ['RHC', 'SA', 'GA', 'MIMIC'],
    'Six Peaks': [fitness_six_peaks['RHC'][-1], fitness_six_peaks['SA'][-1], fitness_six_peaks['GA'][-1], None],
    'Continuous Peaks': [fitness_cont_peaks['RHC'][-1], fitness_cont_peaks['SA'][-1], fitness_cont_peaks['GA'][-1],
                         fitness_cont_peaks['MIMIC'][-1]]
}

fitness_df_peaks = pd.DataFrame(best_fitness_values_peaks)

# Fixing the issue with the x-axis labels
fitness_df_melted_peaks = fitness_df_peaks.melt(id_vars='Algorithm', var_name='Problem', value_name='Best Fitness')
# fitness_df_melted_peaks['Problem'] = fitness_df_melted_peaks['Problem'].replace({
#     '0.0': 'Six Peaks',
#     '1.0': 'Continuous Peaks'
# })

plt.figure(figsize=(12, 6))
sns.barplot(x='Problem', y='Best Fitness', hue='Algorithm', data=fitness_df_melted_peaks)
plt.title('Best Fitness Values for Each Algorithm and Peaks Problem')
plt.ylabel('Best Fitness')
plt.xlabel('Problem')  # Added x-axis label
plt.show()

# Bar chart for NN weight optimization separately
best_fitness_values_nn = {
    'Algorithm': ['RHC', 'SA', 'GA'],
    'NN Weight Opt.': [nn_rhc_fitness, nn_sa_fitness, nn_ga_fitness]
}

fitness_df_nn = pd.DataFrame(best_fitness_values_nn)

# Fixing the issue with the x-axis labels
fitness_df_melted_nn = fitness_df_nn.melt(id_vars='Algorithm', var_name='Problem', value_name='Best Fitness')
fitness_df_melted_nn['Problem'] = fitness_df_melted_nn['Problem'].replace({
    'NN Weight Opt.': 'NN Weight Opt.'
})

plt.figure(figsize=(12, 6))
sns.barplot(x='Problem', y='Best Fitness', hue='Algorithm', data=fitness_df_melted_nn)
plt.title('Best Fitness Values for Each Algorithm in NN Weight Optimization')
plt.ylabel('Best Fitness')
plt.xlabel('Problem')  # Added x-axis label
plt.show()

# Comparison of A1.py Neural Network accuracy to new algorithms
a1_accuracy = 0.9697  # Accuracy from A1.py for Neural Network

# Filter out None values before taking the max
nn_weight_opt_best = [value for value in best_fitness_values_nn['NN Weight Opt.'] if value is not None]

comparison_data = {
    'Metric': ['Accuracy'],
    'A1.py (Neural Network)': [a1_accuracy],
    'New Algorithms (Best)': [max(nn_weight_opt_best)]
}

comparison_df = pd.DataFrame(comparison_data)

plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='value', hue='variable', data=pd.melt(comparison_df, id_vars='Metric'))
plt.title('Comparison of Accuracy: A1.py vs New Algorithms')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limit for better comparison
plt.show()

# Grid search for NN parameter tuning
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (10, 10)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
}

grid_search = GridSearchCV(MLPClassifier(max_iter=100), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best parameters found by grid search:", grid_search.best_params_)

# Apply the best parameters found to the NN weight optimization problem
best_nn = grid_search.best_estimator_

# Define a custom fitness function using the best parameters found by grid search
def custom_fitness_function_tuned(weights):
    input_to_hidden_weights = weights[:num_features * num_hidden_nodes].reshape((num_features, num_hidden_nodes))
    hidden_to_output_weights = weights[num_features * num_hidden_nodes:].reshape((num_hidden_nodes, num_output_nodes))

    nn = MLPClassifier(hidden_layer_sizes=(num_hidden_nodes,), max_iter=1, activation=best_nn.activation,
                       solver=best_nn.solver, alpha=best_nn.alpha)
    nn.coefs_ = [input_to_hidden_weights, hidden_to_output_weights]
    nn.intercepts_ = [np.zeros(num_hidden_nodes), np.zeros(num_output_nodes)]

    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    fitness = accuracy_score(y_test, y_pred)
    return fitness

# Redefine the optimization problem with the tuned fitness function
fitness_nn_tuned = mlrose.CustomFitness(custom_fitness_function_tuned)
problem_nn_tuned = mlrose.ContinuousOpt(length=num_weights, fitness_fn=fitness_nn_tuned, maximize=True, min_val=-1,
                                        max_val=1)

# Solve the tuned NN weight optimization problem
nn_rhc_state_tuned, nn_rhc_fitness_tuned, nn_rhc_curve_tuned = solve_nn_problem(problem_nn_tuned, 'RHC')
nn_sa_state_tuned, nn_sa_fitness_tuned, nn_sa_curve_tuned = solve_nn_problem(problem_nn_tuned, 'SA')
nn_ga_state_tuned, nn_ga_fitness_tuned, nn_ga_curve_tuned = solve_nn_problem(problem_nn_tuned, 'GA')

# Plot the tuned fitness curves
plot_fitness_iterations(
    [nn_rhc_curve_tuned, nn_sa_curve_tuned, nn_ga_curve_tuned],
    'Neural Network Weight Optimization (Tuned)',
    ['RHC', 'SA', 'GA'],
    ['blue', 'green', 'red']
)
