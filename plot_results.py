import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd


def main(result_folder):
    # TODO: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot
    results_file_path = f'mia-result/{result_folder}/results.csv'
    summary_file_path = f'mia-result/{result_folder}/results_summary.csv'
    df = pd.read_csv(results_file_path, sep=';')
    summary = pd.read_csv(summary_file_path, sep=';')

    print(f'Results from: {result_folder}\n')
    print(df.head())
    print(summary.head())

    amygdala = df[df['LABEL'] == 'Amygdala']
    grey_matter = df[df['LABEL'] == 'GreyMatter']

    print(amygdala.head())

    plt.boxplot([amygdala['DICE'], grey_matter['DICE']])
    plt.title('Dice score by class')
    plt.ylabel('Dice')
    plt.xticks([1, 2], ['Amygdala', 'Grey matter'])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting the results.")
    parser.add_argument("result_folder",
                        type=str,
                        help="The folder name inside mia-result that you want the metrics plotted from.")
    args = parser.parse_args()
    main(args.result_folder)
