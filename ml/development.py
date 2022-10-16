import logging
import os
import hydra
import joblib
import mlflow
import pandas as pd

from dvc.repo import Repo
from mlflow.tracking import MlflowClient
from process import process_data
from train_model import train_random_forest, inference, compute_model_metrics
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


def get_experiment(client: MlflowClient, experiment_name: str):

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    else:
        experiment_id = experiment.experiment_id

    return experiment, experiment_id


@hydra.main(config_name='training_config', config_path='../conf')
def main(config: DictConfig):

    client = MlflowClient()
    _, experiment_id = get_experiment(client, config['experiment_name'])
    mlflow.start_run(experiment_id=experiment_id)

    logging.info("Run Started")
    logging.info("Inputs %s", config)

    root_path = hydra.utils.get_original_cwd()

    mlflow.log_artifact(os.path.join(root_path,'conf','training_config.yaml'), 'config.yaml')

    logging.info('Retrieving Data')
    repo = Repo(root_dir=root_path)
    repo.pull()

    train_data_path = os.path.join(root_path,'data','cleaned','census_clean.csv')
    mlflow.log_artifact(train_data_path,"census_clean.csv")
    data = pd.read_csv(train_data_path)

    data_config = config['data']
    train, test = train_test_split(data, **data_config)

    logging.info("Processing Data")
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=config['categorical'], label=config['label'], training=True)

    joblib.dump(encoder,"encoder.pkl")
    mlflow.log_artifact("encoder.pkl")

    joblib.dump(lb,"lb.pkl")
    mlflow.log_artifact("lb.pkl")

    X_test, y_test, _, _ = process_data(
        test,categorical_features=config['categorical'], label=config['label'], training=False,
        encoder=encoder, lb=lb
    )

    logging.info("Training Model")
    mlflow.log_params(config['random_forest'])
    model = train_random_forest(X_train, y_train, config['random_forest'])


    joblib.dump(model,"random_forest_model.pkl")
    mlflow.log_artifact("random_forest_model.pkl")
    mlflow.sklearn.log_model(model,"random_forest_model")

    logging.info("Evaluating Model")
    y_hat = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_hat)

    mlflow.log_metric("precision", precision)
    logging.info("Precision %f", precision)

    mlflow.log_metric("recall", recall)
    logging.info("Recall %f", recall)

    mlflow.log_metric("fbeta", fbeta)
    logging.info("FBeta %f", fbeta)

    mlflow.end_run()

if __name__ == "__main__":
    main()