"""ML Tests"""
from ml.process import process_data
from ml.train_model import inference, compute_model_metrics

NUM_EXPECTED_FEATURES = 108


def test_processing(testing_data, input_labels, production_model):
    """Tests the processed data is structured as expected"""
    _, encoder, lb = production_model
    cat, target = input_labels

    x, _, _, _ = process_data(
        testing_data,
        categorical_features=cat,
        label=target,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    assert x.shape == (len(testing_data), NUM_EXPECTED_FEATURES)
    unique_features = len(set(testing_data.columns) - set(cat) - set(target))
    unique_features += sum(encoder._n_features_outs)  # pylint: disable=protected-access
    assert unique_features == x.shape[1]


def test_saved_inference(testing_data, production_model, input_labels):
    """Test the model to ensure it is generating the appropriate inference value and structure"""
    model, encoder, lb = production_model
    cat, target = input_labels

    x, _, _, _ = process_data(
        testing_data,
        categorical_features=cat,
        label=target,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_hat = inference(model, x)
    labels = lb.inverse_transform(y_hat)

    assert len(labels) == len(testing_data)
    assert set(labels) == set(lb.classes_)


def test_metrics(testing_data, input_labels, production_model):
    """Test metric computation to ensure the right values are being returned"""
    model, encoder, lb = production_model
    cat, target = input_labels

    x, y, _, _ = process_data(
        testing_data,
        categorical_features=cat,
        label=target,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_hat = inference(model, x)
    precision, recall, fbeta = compute_model_metrics(y, y_hat)

    assert isinstance(precision, float) and precision >= 0.0 and precision <= 1.0
    assert isinstance(recall, float) and recall >= 0.0 and recall <= 1.0
    assert isinstance(fbeta, float) and fbeta >= 0.0 and fbeta <= 1.0
