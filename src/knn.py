"""Собственный KNN: полный перебор соседей, NumPy, две задачи.

Расстояния, выбор соседей, веса и агрегация реализованы здесь.
Из sklearn используются интерфейсы оценщиков и проверка входных данных.
Предобработка категорий и масштабирование остаются снаружи, в Pipeline.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _BaseKNN(BaseEstimator):
    """Общие параметры, хранение train и поиск ближайших соседей."""

    def __init__(self, n_neighbors=5, weights="uniform", p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X, y):
        X, y = check_X_y(
            X, y, dtype=np.float64,
            y_numeric=isinstance(self, RegressorMixin),
        )
        if (
            isinstance(self.n_neighbors, (bool, np.bool_))
            or not isinstance(self.n_neighbors, (int, np.integer))
            or not 1 <= self.n_neighbors <= len(X)
        ):
            raise ValueError("n_neighbors должен быть целым числом от 1 до len(X).")
        if self.weights not in ("uniform", "distance"):
            raise ValueError("Допустимые веса: uniform, distance.")
        if self.p not in (1, 2):
            raise ValueError("Поддерживаются p=1 и p=2.")

        # У KNN обучение состоит в сохранении обучающих наблюдений.
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.n_features_in_ = X.shape[1]
        return self

    def _neighbors_and_weights(self, X):
        check_is_fitted(self, ["X_", "y_"])
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Число признаков отличается от обучающей выборки.")

        for query in X:
            difference = self.X_ - query
            if self.p == 1:
                distances = np.abs(difference).sum(axis=1)
            else:
                distances = np.sqrt((difference ** 2).sum(axis=1))

            # При равных расстояниях сохраняем порядок строк train.
            indices = np.argsort(distances, kind="stable")[:self.n_neighbors]
            neighbor_distances = distances[indices]

            if self.weights == "uniform":
                weights = np.ones(len(indices), dtype=float)
            else:
                exact_matches = neighbor_distances == 0
                if exact_matches.any():
                    # При точных совпадениях учитываем только их.
                    weights = exact_matches.astype(float)
                else:
                    # Относительные веса эквивалентны 1/d, но не превышают 1.
                    weights = neighbor_distances.min() / neighbor_distances

            yield indices, weights / weights.sum()


class MyKNNClassifier(ClassifierMixin, _BaseKNN):
    """Классификация по сумме весов соседей каждого класса."""

    def fit(self, X, y):
        super().fit(X, y)
        check_classification_targets(self.y_)
        self.classes_, self.encoded_y_ = np.unique(
            self.y_, return_inverse=True,
        )
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "encoded_y_"])
        probabilities = []
        for indices, weights in self._neighbors_and_weights(X):
            probabilities.append(np.bincount(
                self.encoded_y_[indices],
                weights=weights,
                minlength=len(self.classes_),
            ))
        return np.asarray(probabilities)

    def predict(self, X):
        # При равных голосах выбираем первый класс в отсортированном classes_.
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class MyKNNRegressor(RegressorMixin, _BaseKNN):
    """Регрессия через взвешенное среднее целевых значений соседей."""

    def predict(self, X):
        return np.asarray([
            np.dot(weights, self.y_[indices])
            for indices, weights in self._neighbors_and_weights(X)
        ])
