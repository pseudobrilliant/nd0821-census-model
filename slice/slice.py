"""Slice Performance"""
from sklearn.model_selection import train_test_split
import csv
import pandas as pd

from api.utils import get_labels, get_production_model
from ml.process import process_data
from ml.train_model import inference, compute_model_metrics


def slice_performance(data_path="./data/cleaned/census_clean.csv"):
    """Performs a performance test on each categorical slice"""
    categorical, target = get_labels()
    model, encoder, lb = get_production_model()

    df = pd.read_csv(data_path)
    _, test = train_test_split(df)

    results = []
    for cat in categorical:
        for unique_item in test[cat].unique():
            slice_df = test.loc[test[cat] == unique_item]

            x_test, y_test, _, _ = process_data(
                slice_df, categorical, label=target, training=False,
                encoder=encoder, lb=lb
            )

            y_hat = inference(model, x_test)
            precision, recall, f_beta = compute_model_metrics(y_test, y_hat)

            print(f"{cat} | {unique_item} - precision: {precision}, recall: {recall}, fbeta: {f_beta}\n")
            results.append([cat, unique_item, precision, recall, f_beta])

    with open("./slice/slice_performance.csv", 'w', encoding="utf8") as fh:
        csv_writer = csv.writer(fh)
        csv_writer.writerow(["category", "slice", "precision", "recall", "fbeta"])
        csv_writer.writerows(results)


if __name__ == "__main__":
    slice_performance()
