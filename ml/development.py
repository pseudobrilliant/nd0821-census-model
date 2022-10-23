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

SUPPORTED_ALGORITHMS = {"random_forest": train_random_forest}

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

    categorical_features = config['labels']['categorical']
    target_feature = config['labels']['target']

    logging.info("Processing Data")
    X_train, y_train, encoder, lb = process_data(
    train, categorical_features=categorical_features, label=target_feature, training=True)

    joblib.dump(encoder,"encoder.pkl")
    mlflow.log_artifact("encoder.pkl")

    joblib.dump(lb,"lb.pkl")
    mlflow.log_artifact("lb.pkl")

    X_test, y_test, _, _ = process_data(
        test, categorical_features=categorical_features, label=target_feature, training=False,
        encoder=encoder, lb=lb
    )

    model_name = config['model']['model_name']
    model_params = config['model']['model_parameters']

    if model_name in SUPPORTED_ALGORITHMS:
        logging.info(f"Training Model {model_name}")
        mlflow.log_params(model_params)

        model = SUPPORTED_ALGORITHMS[model_name](X_train, y_train, model_params)
    else:
        raise ValueError("Model type requested not currently available")

    model_file_name = f"{model_name}_model.pkl"
    joblib.dump(model, model_file_name)
    mlflow.log_artifact(model_file_name)
    mlflow.sklearn.log_model(model,f"{model_name}_model")

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