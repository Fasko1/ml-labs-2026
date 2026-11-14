"""Финальная оценка KNN на тесте и сохранение прогнозов.

Для каждой задачи оцениваются четыре заранее выбранных варианта модели.
"""

from time import perf_counter
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


def make_final_pipeline(X, model, scale_numeric=False):
    """Создаём отдельный Pipeline для каждого варианта модели."""
    preprocessor = make_preprocessor(X)
    if scale_numeric:
        preprocessor.set_params(numeric=StandardScaler())
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# Проверяем зафиксированные параметры до первого обращения к test.
assert params_cls == {"n_neighbors": 5, "weights": "distance", "p": 2}
assert params_reg == {"n_neighbors": 5, "weights": "distance", "p": 1}

final_models_cls = {
    "sklearn baseline": make_final_pipeline(
        X_train_cls, KNeighborsClassifier(),
    ),
    "sklearn improved": make_final_pipeline(
        X_train_cls, KNeighborsClassifier(**params_cls),
    ),
    "own baseline": make_final_pipeline(
        X_train_cls, MyKNNClassifier(),
    ),
    "own improved": make_final_pipeline(
        X_train_cls, MyKNNClassifier(**params_cls),
    ),
}

final_models_reg = {
    "sklearn baseline": make_final_pipeline(
        X_train_reg, KNeighborsRegressor(),
    ),
    "sklearn improved": make_final_pipeline(
        X_train_reg, KNeighborsRegressor(**params_reg), scale_numeric=True,
    ),
    "own baseline": make_final_pipeline(
        X_train_reg, MyKNNRegressor(),
    ),
    "own improved": make_final_pipeline(
        X_train_reg, MyKNNRegressor(**params_reg), scale_numeric=True,
    ),
}


def evaluate_final(models, X_train, y_train, X_test, y_test, task):
    """Обучаем на полном train и оцениваем зафиксированные варианты на test."""
    rows = []
    predictions = pd.DataFrame({
        "row_position": X_test.index.to_numpy(),
        "target": y_test.to_numpy(),
    })

    for name, pipeline in models.items():
        started = perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = perf_counter() - started

        started = perf_counter()
        predicted = pipeline.predict(X_test)
        predict_seconds = perf_counter() - started

        row = {
            "model": name,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
        predictions[name] = predicted

        if task == "classification":
            # ROC-AUC рассчитываем по вероятностям класса 1.
            started = perf_counter()
            positive_column = list(pipeline.classes_).index(1)
            probabilities = pipeline.predict_proba(X_test)[:, positive_column]
            row["predict_proba_seconds"] = perf_counter() - started
            predictions[f"{name} probability_1"] = probabilities
            row.update({
                "macro_f1": f1_score(y_test, predicted, average="macro"),
                "accuracy": accuracy_score(y_test, predicted),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "precision_1": precision_score(y_test, predicted, zero_division=0),
                "recall_1": recall_score(y_test, predicted, zero_division=0),
            })
        else:
            row.update({
                "mae": mean_absolute_error(y_test, predicted),
                "rmse": np.sqrt(mean_squared_error(y_test, predicted)),
                "r2": r2_score(y_test, predicted),
            })
        rows.append(row)

    return pd.DataFrame(rows), predictions


final_test_cls, final_predictions_cls = evaluate_final(
    final_models_cls, X_train_cls, y_train_cls,
    X_test_cls, y_test_cls, "classification",
)
final_test_reg, final_predictions_reg = evaluate_final(
    final_models_reg, X_train_reg, y_train_reg,
    X_test_reg, y_test_reg, "regression",
)

# Сохраняем значения полной точности и индивидуальные прогнозы.
results_dir.mkdir(parents=True, exist_ok=True)
for filename, table in {
    "lab1_final_test_classification.csv": final_test_cls,
    "lab1_final_test_regression.csv": final_test_reg,
    "lab1_final_predictions_classification.csv": final_predictions_cls,
    "lab1_final_predictions_regression.csv": final_predictions_reg,
}.items():
    table.to_csv(results_dir / filename, index=False)

print("Финальная оценка классификации на test:")
display(final_test_cls.round(4))
print("Финальная оценка регрессии на test:")
display(final_test_reg.round(4))
print("Обучение и предсказание измерены вместе с предобработкой, один запуск.")
print("Результаты и прогнозы сохранены в:", results_dir)
