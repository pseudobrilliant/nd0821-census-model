Live Project Deployment: https://cblythe-census.herokuapp.com/

Live Test Script: liive_inference_test.py

Model Selected: Random Forest Classifies - See model_card.md for more details

Screenshots: See the screenshots directory for images of the live product results and mlflow integration.

# Running Locally

## Experiment Set up

To run the MLflow experiment you must have pyenv installed. Once running in a compatible pyenv shell you should install MLFlow with the following command `pip install mlflow`.

you can then run the mlflow experiments using the following command `mlflow run ./ -P hydra_options="-m"`.

This will run the experiment with additional sweeps and configurations set in the `conf` directory. To view the results you can inspect the `mlruns` and `multirun` directories.

You can also view the results by launching the mlflow ui using the command `mlflow ui`.

## API Set up

To run the API locally you should first install the required dependencies in your local or virtual python environment with the command `python -m pip install -r requirements.txt`.

Once the dependencies are accessible within your enviroment you may start the API application by running `python api/app.py` within your terminal.
