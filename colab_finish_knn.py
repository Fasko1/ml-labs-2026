"""Графики и выгрузка результатов ЛР1.

Матрицы ошибок, графики регрессии и архив результатов с настройками.
"""

import importlib.metadata
import zipfile
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

figures_dir = results_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Строим матрицы ошибок всех четырёх классификаторов.
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
confusion_rows = []
for ax, name in zip(axes.flat, final_models_cls):
    predicted = final_predictions_cls[name].to_numpy()
    ConfusionMatrixDisplay.from_predictions(
        y_test_cls, predicted, labels=[0, 1],
        display_labels=["Не купил", "Купил"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(name)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    tn, fp, fn, tp = confusion_matrix(
        y_test_cls, predicted, labels=[0, 1],
    ).ravel()
    confusion_rows.append({
        "model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    })

fig.suptitle("Классификация страховки: отложенный тест")
fig.tight_layout()
fig.savefig(figures_dir / "lab1_confusion_matrices.png", dpi=180)
plt.show()

confusion_table = pd.DataFrame(confusion_rows)
confusion_table.to_csv(
    results_dir / "lab1_confusion_counts.csv", index=False,
)
print("Количество ошибок и правильных ответов:")
display(confusion_table)

# Строим прогнозы и остатки двух улучшенных регрессоров.
selected_names = ["sklearn improved", "own improved"]
actual = y_test_reg.to_numpy()
plot_limit = max(
    actual.max(),
    *(final_predictions_reg[name].max() for name in selected_names),
) * 1.03
residual_limit = max(
    np.abs(actual - final_predictions_reg[name].to_numpy()).max()
    for name in selected_names
) * 1.05

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for column, name in enumerate(selected_names):
    predicted = final_predictions_reg[name].to_numpy()
    ax = axes[0, column]
    ax.scatter(actual, predicted, alpha=0.5, s=22)
    ax.plot([0, plot_limit], [0, plot_limit], "--", color="black")
    ax.set_xlim(0, plot_limit)
    ax.set_ylim(0, plot_limit)
    ax.set_title(name)
    ax.set_xlabel("Фактическая цена, евро")
    ax.set_ylabel("Предсказанная цена, евро")

    ax = axes[1, column]
    ax.scatter(predicted, actual - predicted, alpha=0.5, s=22)
    ax.axhline(0, linestyle="--", color="black")
    ax.set_ylim(-residual_limit, residual_limit)
    ax.set_xlabel("Предсказанная цена, евро")
    ax.set_ylabel("Фактическая цена − прогноз, евро")

fig.suptitle("Улучшенные регрессоры: отложенный тест")
fig.tight_layout()
fig.savefig(figures_dir / "lab1_regression_errors.png", dpi=180)
plt.show()

# Сохраняем части CV в координатах строк исходных CSV.
for name, positions, groups, folds in [
    ("travel", train_idx_cls, groups_cls, cv_folds_cls),
    ("laptops", train_idx_reg, groups_reg, cv_folds_reg),
]:
    validation_fold = np.full(len(positions), -1, dtype=int)
    for fold_number, (_, validation_positions) in enumerate(folds):
        assert (validation_fold[validation_positions] == -1).all()
        validation_fold[validation_positions] = fold_number
    assert (validation_fold >= 0).all()
    pd.DataFrame({
        "row_position": positions,
        "group_id": groups[positions],
        "validation_fold": validation_fold,
    }).to_csv(split_dir / f"{name}_cv.csv", index=False)

# Записываем версии библиотек и зафиксированные настройки.
package_names = [
    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn",
]
package_versions = {
    name: importlib.metadata.version(name) for name in package_names
}
settings = {
    "lab": 1,
    "algorithm": "KNN",
    "python": sys.version.split()[0],
    "random_state": RANDOM_STATE,
    "packages": package_versions,
    "classification": {
        "features": X_train_cls.columns.tolist(),
        "improved_parameters": params_cls,
        "numeric_scaling": False,
        "selection_metric": "macro_f1",
    },
    "regression": {
        "features": X_train_reg.columns.tolist(),
        "improved_parameters": params_reg,
        "numeric_scaling": "StandardScaler",
        "ppi": False,
        "log_target": False,
        "selection_metric": "mae",
    },
    "timing": "One run; fit/predict include preprocessing; proba timed separately",
}
(results_dir / "lab1_settings.json").write_text(
    json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8",
)

# Упаковываем результаты, рисунки, разбиения и версии зависимостей.
# Архив содержит результаты и разбиения; файл ipynb сохраняется отдельно.
archive_path = Path("/content/lab1_artifacts.zip")
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for directory, prefix in [
        (results_dir, Path("results")),
        (split_dir, Path("data/splits")),
    ]:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, str(prefix / path.relative_to(directory)))
    archive.writestr(
        "requirements.txt",
        "\n".join(f"{name}=={version}" for name, version in package_versions.items())
        + "\n",
    )

print("Архив результатов подготовлен:", archive_path.name)
files.download(str(archive_path))
