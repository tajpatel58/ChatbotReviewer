import mlflow
import numpy as np
from prefect import task
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


@task
def configure_mlflow():
    pass


@task
def initialise_model_dict():
    pass


@task
def split_data_task(X: np.array, y: np.array, train_size: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y
    )
    return X_train, X_test, y_train, y_test


@task
def hyperparam_tune_and_train_task(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    model_info_dict: dict,
    cv: int,
) -> None:
    model_name = model_info_dict["name"]
    model = model_info_dict["model"]
    model_grid_search_dict = model_info_dict["grid_search_dict"]
    with mlflow.start_run(run_name=model_name):
        # calculate metrics:
        grid_search = GridSearchCV(
            estimator=model, param_grid=model_grid_search_dict, cv=cv
        )
        grid_search.fit(X=X_train, y=y_train)
        best_estimator = grid_search.best_estimator_
        best_hyper_params = grid_search.best_params_
        cv_score = (
            cross_val_score(best_estimator, X_train, y_train, cv=cv).mean().round(3)
        )
        predicted_test = best_estimator.predict(X_test)
        accuracy = accuracy_score(y_true=y_test, y_pred=predicted_test).round(3)

        # log metrics with mlflow:
        mlflow.log_metric(key="cross_val_score", value=cv_score)
        mlflow.log_metric(key="accuracy", value=accuracy)
        mlflow.log_param(key="grid_searched_params", value=model_grid_search_dict)
        mlflow.log_params(best_hyper_params)
        mlflow.sklearn.log_model(best_estimator)
    return None
