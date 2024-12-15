import matplotlib.pyplot as plt
import pandas as pd
import warnings

path_to_xlsx = f'results_analysis/results_consistency.xlsx'
experiment_names = ['Baseline', 'Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 4', 'Experiment 5',
                    'Experiment 6', 'Experiment 7']

# set sheet_name = None if you have more than two excel sheets -> creates a dict of dataframes
res_a = pd.read_excel(path_to_xlsx, sheet_name=0)
res_b = pd.read_excel(path_to_xlsx, sheet_name=1)

dice_mean_a = res_a[(res_a['METRIC'] == 'DICE') & (res_a['STATISTIC'] == 'MEAN')]
dice_mean_b = res_b[(res_b['METRIC'] == 'DICE') & (res_b['STATISTIC'] == 'MEAN')]

hausd_mean_a = res_a[(res_a['METRIC'] == 'HDRFDST') & (res_a['STATISTIC'] == 'MEAN')]
hausd_mean_b = res_b[(res_b['METRIC'] == 'HDRFDST') & (res_b['STATISTIC'] == 'MEAN')]


def compare_experiments(run_a: pd.DataFrame, run_b: pd.DataFrame, metric_name: str, experiment: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(run_a[experiment], run_b[experiment], color='blue', label=f'Mean {metric_name} Scores')

    # Adding diagonal for reference
    plt.plot([0.5, 0.85], [0.5, 0.85], color='red', linestyle='--', label='y=x (Perfect Agreement)')

    # Adding labels and title
    plt.xlabel("Experiment A")
    plt.ylabel("Experiment B")
    plt.title(f'Run A vs B {experiment}: Mean {metric_name} Scores')
    plt.legend()
    plt.grid(True)

    # Annotating points with labels
    for i, label in enumerate(run_a["LABEL"]):
        plt.text(run_a[experiment].values[i] + 0.002, run_b[experiment].values[i], label, fontsize=9)
    plt.show()


for exp in experiment_names:
    compare_experiments(dice_mean_a, dice_mean_b, metric_name='Dice', experiment=exp)
    compare_experiments(hausd_mean_a, hausd_mean_b, metric_name='Hausdorff distance', experiment=exp)
