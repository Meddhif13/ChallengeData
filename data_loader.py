import os
import argparse
import pandas as pd

def load_data(data_folder: str = './data'):
    """Load train and test CSV files from the specified folder."""
    x_train = pd.read_csv(os.path.join(data_folder, 'x_train.csv'))
    y_train = pd.read_csv(os.path.join(data_folder, 'y_train.csv'))
    x_test = pd.read_csv(os.path.join(data_folder, 'x_test.csv'))
    return x_train, y_train, x_test


def main() -> None:
    parser = argparse.ArgumentParser(description='Load ChallengeData CSV files')
    parser.add_argument('--data-folder', default='./data', help='Path containing x_train.csv, y_train.csv and x_test.csv')
    args = parser.parse_args()

    x_train, y_train, x_test = load_data(args.data_folder)
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')


if __name__ == '__main__':
    main()
