# %%
import numpy as np
import pandas as pd

# Folder where results are stored
name_folder = 'results/'

# List of scenarios and methods used in the experiments
scenario_list = ['I', 'II', 'III', 'IV']
method_list = ['DR-CF', 'KPE', 'PE-linear']
method_names = ['DR-CF', 'KPE', 'PE-linear']  # For LaTeX table headers

# Load results
results = dict()
for scenario in scenario_list:
    for method in method_list:
        filename = f"{name_folder}/scenario{scenario}_{method}.csv"
        results[(scenario, method)] = pd.read_csv(filename)

# Compute rejection proportions (p-value < 0.05)
rejection_matrix = np.zeros((len(scenario_list), len(method_list)))
alpha = 0.05

# %%
for i, scenario in enumerate(scenario_list):
    for j, method in enumerate(method_list):
        df = results[(scenario, method)]
        rejection_matrix[i, j] = (df['p_value'] < alpha).mean()

# Wrap into a DataFrame for display and export
df_reject = pd.DataFrame(rejection_matrix, index=scenario_list, columns=method_names)
print(df_reject)

# Output LaTeX version
print(df_reject.T.to_latex(float_format="{:0.2f}".format))

# %%
