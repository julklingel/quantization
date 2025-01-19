from mlflow import log_param, log_metric, start_run, end_run
import mlflow

def setup_mlflow(experiment_name):
    mlflow.set_experiment(experiment_name)

def log_experiment_params(params):
    for key, value in params.items():
        log_param(key, value)

def log_experiment_metrics(metrics):
    for key, value in metrics.items():
        log_metric(key, value)

def track_experiment(experiment_name, params, metrics):
    setup_mlflow(experiment_name)
    with start_run():
        log_experiment_params(params)
        log_experiment_metrics(metrics)
        end_run()