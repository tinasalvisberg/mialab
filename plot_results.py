import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd


def main(result_folder):
    results_file_path = f'mia-result/{result_folder}/results.csv'
    summary_file_path = f'mia-result/{result_folder}/results_summary.csv'
    df = pd.read_csv(results_file_path, sep=';')
    summary = pd.read_csv(summary_file_path, sep=';')

    # remove post-processed results from df
    df = df[~df['SUBJECT'].str.endswith('-PP')]

    print(f'Results from: {result_folder}\n')

    labels = df['LABEL'].unique()
    metrics = {'DICE': 'Dice score', 'HDRFDST': 'Hausdorff distance'}

    for key, value in metrics.items():

        df.boxplot(column=key, by='LABEL', grid=True)
        unique_labels = df['LABEL'].unique()

        for i, label in enumerate(unique_labels, start=1):
            # Select data for the current label
            data_points = df[df['LABEL'] == label][key]

            # Add jitter to the x-coordinates
            x_positions = np.random.normal(i, 0.04, size=len(data_points))

            # Plot data points
            plt.scatter(x_positions, data_points, color="blue", alpha=0.6, label="Data Points" if i == 1 else "")

        plt.title(f"{value} by label")
        plt.suptitle('')  # Remove the automatic Pandas title
        plt.ylabel(value)
        plt.xlabel("Label")
        # plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        plt.grid(True)

        # Set vertical axis limits
        if key == "DICE":
            plt.ylim(0, 1)  # DICE scores range from 0 to 1
        elif key == "HDRFDST":
            plt.ylim(0, 16)  # Hausdorff distance range from 0 to 16

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting the results.")
    parser.add_argument("result_folder",
                        type=str,
                        help="The folder name inside mia-result that you want the metrics plotted from.")
    args = parser.parse_args()
    main(args.result_folder)
