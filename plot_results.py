import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd


def main(result_folder):
    results_file_path = f'mia-result/{result_folder}/results.csv'
    summary_file_path = f'mia-result/{result_folder}/results_summary.csv'
    df = pd.read_csv(results_file_path, sep=';')
    summary = pd.read_csv(summary_file_path, sep=';')

    print(f'Results from: {result_folder}\n')

    labels = df['LABEL'].unique()
    metrics = {'DICE': 'Dice score', 'HDRFDST': 'Hausdorff distance'}

    for key, value in metrics.items():
        plt.figure(figsize=(10, 6))
        df.boxplot(column=key, by='LABEL', grid=True)
        plt.title(f"{value} by label")
        plt.suptitle('')  # Remove the automatic Pandas title
        plt.ylabel(value)
        plt.xlabel("Label")
        # plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting the results.")
    parser.add_argument("result_folder",
                        type=str,
                        help="The folder name inside mia-result that you want the metrics plotted from.")
    args = parser.parse_args()
    main(args.result_folder)
