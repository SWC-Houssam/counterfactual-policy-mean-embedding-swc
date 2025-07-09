# %%

import numpy as np
import pandas as pd

d = dict()
name_folder = 'data_ihdp/'

# %%
# Update this to match the methods and scenarios you ran
b_list = ['I', 'II', 'III', 'IV']
method_list = ['DR-CF', 'KPE', 'PE-linear']
method_names = ['DR-CF', 'KPE', 'PE-linear']

# Read all results
for b in b_list:
    for method in method_list:
        name = f"{name_folder}b{b}_{method}.csv"
        d[name] = pd.read_csv(name)

# %%
# Compute rejection proportions
rejection_proportion = np.zeros((len(b_list), len(method_list)))
confidence_level = 0.05

for i, b in enumerate(b_list):
    for j, method in enumerate(method_list):
        name = f"{name_folder}b{b}_{method}.csv"
        rejection_proportion[i, j] = (d[name]['p_value'] < confidence_level).mean()

# Wrap into dataframe
df = pd.DataFrame(rejection_proportion, index=b_list, columns=method_names)
print(df)

# Display LaTeX
print(df.T.to_latex(float_format="{:0.2f}".format))


# %%
