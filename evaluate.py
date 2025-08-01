import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_data


def evaluate(data_folder: str = './data', model_path: str = 'model.h5') -> float:
    x_train, y_train, _ = load_data(data_folder)
    _, X_val, _, y_val = train_test_split(x_train.values, y_train.values,
                                          test_size=0.2, random_state=42)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X_val, batch_size=32)
    pred_classes = predictions.argmax(axis=1)
    true_classes = y_val.argmax(axis=1)
    acc = accuracy_score(true_classes, pred_classes)
    print(f'Validation accuracy: {acc:.4f}')
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--data-folder', default='./data', help='Folder containing CSV files')
    parser.add_argument('--model-path', default='model.h5', help='Trained model file')
    args = parser.parse_args()

    evaluate(args.data_folder, args.model_path)


if __name__ == '__main__':
    main()
