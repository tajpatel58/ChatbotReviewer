import mlflow
import numpy as np
from pathlib import Path
from prefect import task, flow
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


@task(name="Configuring MlFlow")
def configure_mlflow_task(project_home: Path) -> None:
    experiment_tracking_path = (
        project_home / "src" / "modelling" / "experiment_tracking"
    )
    mlflow.set_tracking_uri("sqlite:///" + str(experiment_tracking_path / "mlflow.db"))
    return None


@task(name="Initialising Hyperparam Search Info")
def initialise_model_dicts_task(random_state: int) -> list:
    # logistic regression hyperparams search space
    log_reg_params = {
        "penalty": ["l1", "l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }
    # svm hyperparams search space
    svc_params = {
        "C": [0.5, 0.7, 0.9, 1],
        "kernel": ["rbf", "poly", "sigmoid", "linear"],
    }
    # random forest hyperparams search space:
    rf_params = {
        "max_depth": list(range(2, 7, 1)),
        "min_samples_leaf": list(range(2, 7, 1)),
        "random_state": [random_state],
    }
    # extra trees hyperparams search space:
    et_params = {
        "n_estimators": list(range(100, 500, 50)),
        "max_features": [10, "sqrt", "log2"],
        "random_state": [random_state],
    }

    log_reg_model = LogisticRegression(solver="liblinear")
    svm_model = SVC()
    rf_model = RandomForestClassifier()
    et_model = ExtraTreesClassifier()

    model_names = [
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest",
        "Extra Trees",
    ]
    models = [log_reg_model, svm_model, rf_model, et_model]
    hyperparam_dicts = [log_reg_params, svc_params, rf_params, et_params]

    model_dicts = []
    for index in range(4):
        model_dict = {}
        model_dict["name"] = model_names[index]
        model_dict["model"] = models[index]
        model_dict["grid_search_dict"] = hyperparam_dicts[index]
        model_dicts.append(model_dict)
    return model_dicts


@task(name="Splitting Dataset to Train/Test")
def split_data_task(X: np.array, y: np.array, train_size: float):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y
    )
    return X_train, X_test, y_train, y_test


@task
def hyperparam_tune_and_train_task(
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
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
        mlflow.sklearn.log_model(best_estimator, model_name)
    return None


@flow(validate_parameters=False, log_prints=True)
def training_models_flow(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    train_size: float,
    cv: int,
    project_home: Path,
):
    configure_mlflow_task(project_home)
    X_train, X_test, y_train, y_test = split_data_task(X, y, train_size)
    model_dicts = initialise_model_dicts_task(random_state=random_state)
    for model_dict in model_dicts:
        training_task_name = f"Training Model: {model_dict['name']}"
        hyperparam_tune_and_train_task.with_options(name=training_task_name)(
            X_train, X_test, y_train, y_test, model_info_dict=model_dict, cv=cv
        )
    return None
