import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd


def main(result_folder):
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot
    file_path = f'mia-result/{result_folder}'
    df = pd.read_csv(file_path)

    print(f'Your folder is: {result_folder}\n')
    print(df.head())

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting the results.")
    parser.add_argument("result_folder",
                        type=str,
                        help="The folder name inside mia-result that you want the metrics plotted from.")
    args = parser.parse_args()
    main(args.result_folder)
