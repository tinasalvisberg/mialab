import matplotlib.pyplot as plt
import pandas as pd
import warnings

path_to_xlsx = f'results_analysis/results_consistency.xlsx'

# set sheet_name = None if you have more than two excel sheets -> creates a dict of dataframes
res_a = pd.read_excel(path_to_xlsx, sheet_name=0)
res_b = pd.read_excel(path_to_xlsx, sheet_name=1)

dice_mean_a = res_a[(res_a['METRIC'] == 'DICE') & (res_a['STATISTIC'] == 'MEAN')]
dice_mean_b = res_b[(res_b['METRIC'] == 'DICE') & (res_b['STATISTIC'] == 'MEAN')]

hausd_mean_a = res_a[(res_a['METRIC'] == 'HDRFDST') & (res_a['STATISTIC'] == 'MEAN')]
hausd_mean_b = res_b[(res_b['METRIC'] == 'HDRFDST') & (res_b['STATISTIC'] == 'MEAN')]

plt.figure(figsize=(8, 6))
plt.scatter(dice_mean_a["Baseline"], dice_mean_b["Baseline"], color='blue', label='Mean DICE Scores')

# Adding diagonal for reference
plt.plot([0.5, 0.85], [0.5, 0.85], color='red', linestyle='--', label='y=x (Perfect Agreement)')

# Adding labels and title
plt.xlabel("Experiment run A")
plt.ylabel("Experiment run B")
plt.title("Run A vs B baseline: Mean DICE Scores")
plt.legend()
plt.grid(True)


# Annotating points with labels
x_label = dice_mean_a['Baseline'].values
for i, label in enumerate(dice_mean_a["LABEL"]):
    plt.text(dice_mean_a["Baseline"].values[i] + 0.002, dice_mean_b["Baseline"].values[i], label, fontsize=9)
plt.show()
