"""Explainability dashboard: Flask server and JSON API.

Run from the project root:
    python app.py
then open http://127.0.0.1:5000/

Endpoints (all take/return JSON):
    POST /api/explain    prediction, four explanations, per-image deletion scores
    POST /api/sanity     explanations from a fully randomized model + correlation
    GET  /api/benchmark/<dataset>   results table from evaluate_explanations.py

Input for /api/explain and /api/sanity:
    {"dataset": "mnist", "image": "data:image/png;base64,..."}   drawing/upload
    {"dataset": "mnist", "sample_index": 123}                    test-set image

The server has no authentication. It binds to 127.0.0.1 (see config.py), so it
is only reachable from this computer.
"""

import threading

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

import config
from xai.data import DATASETS, class_names, info, is_decoy, load_split
from xai.explain import METHODS, explain, normalize
from xai.metrics import area_under_curve, cascading_randomization, deletion_curve, rank_correlation
from xai.models import checkpoint_path, load_model
from xai.paths import RESULTS_DIR
from xai.preprocess import InvalidImage, decode_data_url, image_to_tensor

METHOD_DESCRIPTIONS = {
    "saliency": "Absolute gradient of the class score with respect to each pixel.",
    "smoothgrad": "Saliency averaged over 25 noisy copies of the input.",
    "integrated_gradients": "Gradients accumulated along a path from a black image to the input.",
    "grad_cam": "Last convolutional feature maps weighted by their gradients (6x6, upsampled).",
    "occlusion": "Drop in the class score when each 4x4 patch is covered.",
}
DELETION_STEPS = 20
RANDOM_REPEATS = 5  # random orderings averaged for the per-image baseline
SUMMARY_DIR = RESULTS_DIR / "analysis"
# Benchmark columns shown in the dashboard: (column, title).
BENCHMARK_COLUMNS = [
    ("method", "Method"),
    ("deletion_zero", "Deletion, black ↓"),
    ("deletion_blur", "Deletion, blur ↓"),
    ("deletion_inpaint", "Deletion, inpaint ↓"),
    ("insertion_blur", "Insertion, blur ↑"),
    ("corr_object", "Sanity corr. ↓"),
    ("mass_accuracy", "Ground truth: mass acc. ↑"),
    ("rank_accuracy", "Ground truth: rank acc. ↑"),
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

# Models are shared between requests. Captum adds temporary hooks to them,
# so explanation work runs one request at a time.
LOCK = threading.Lock()
_models, _random_models, _test_sets = {}, {}, {}


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


@app.errorhandler(ApiError)
def handle_api_error(error):
    return jsonify({"error": error.message}), error.status


@app.errorhandler(413)
def handle_too_large(_):
    return jsonify({"error": "The file is too large (limit 5 MB)."}), 413


def get_model(dataset):
    if dataset not in _models:
        try:
            _models[dataset] = load_model(dataset)
        except FileNotFoundError as error:
            raise ApiError(str(error), 503) from error
    return _models[dataset]


def get_random_model(dataset):
    """The model with every layer re-initialised (last sanity-check stage)."""
    if dataset not in _random_models:
        for _, randomized in cascading_randomization(get_model(dataset), seed=0):
            pass
        _random_models[dataset] = randomized
    return _random_models[dataset]


def get_test_set(dataset):
    if dataset not in _test_sets:
        # Decoy datasets: patch at a wrong class's position, as in the
        # evaluation, so the demo shows the model following the shortcut.
        mode = "conflicting" if is_decoy(dataset) else "aligned"
        _test_sets[dataset] = load_split(dataset, "test", mode)
    return _test_sets[dataset]


def available_datasets():
    """Datasets that have a trained model."""
    return [name for name in DATASETS if checkpoint_path(name).exists()]


def rounded(values, digits=3):
    return np.round(np.asarray(values, dtype=float), digits).tolist()


def read_input():
    """Parse the request into (dataset, input tensor, true label or None)."""
    payload = request.get_json(silent=True) or {}
    dataset = payload.get("dataset")
    if dataset not in DATASETS:
        raise ApiError(f"Unknown dataset. Choose from {list(DATASETS)}.")
    if not checkpoint_path(dataset).exists():
        raise ApiError(f"No trained model for {dataset}. Run train_models.py first.", 503)

    if "sample_index" in payload:
        test_set = get_test_set(dataset)
        try:
            index = int(payload["sample_index"])
        except (TypeError, ValueError) as error:
            raise ApiError("sample_index must be a number.") from error
        if not 0 <= index < len(test_set):
            raise ApiError(f"sample_index must be between 0 and {len(test_set) - 1}.")
        image, label = test_set[index]
        return dataset, image.unsqueeze(0), int(label)

    if isinstance(payload.get("image"), str):
        try:
            return dataset, image_to_tensor(decode_data_url(payload["image"]), dataset), None
        except InvalidImage as error:
            raise ApiError(str(error)) from error

    raise ApiError("Send either 'image' (a data URL) or 'sample_index'.")


def predict(model, x):
    with torch.no_grad():
        probs = torch.exp(model(x))[0]
    return int(probs.argmax()), probs


@app.route("/")
def start():
    datasets = {
        name: {
            "title": info(name)["title"],
            "classes": class_names(name),
            "test_size": len(get_test_set(name)),
        }
        for name in available_datasets()
    }
    methods = {
        name: {"title": title, "description": METHOD_DESCRIPTIONS[name]}
        for name, (title, _) in METHODS.items()
    }
    return render_template("index.html", page_config={"datasets": datasets, "methods": methods})


@app.post("/api/explain")
def api_explain():
    dataset, x, true_label = read_input()
    with LOCK:
        model = get_model(dataset)
        pred, probs = predict(model, x)
        rng = np.random.default_rng(0)

        explanations = []
        for name, (title, _) in METHODS.items():
            attribution = explain(model, x, pred, name)
            curve = deletion_curve(model, x, pred, attribution, steps=DELETION_STEPS, rng=rng)
            explanations.append({
                "method": name,
                "title": title,
                "heatmap": rounded(normalize(attribution)),
                "deletion_auc": round(area_under_curve(curve), 3),
                "deletion_curve": rounded(curve),
            })

        random_curve = np.mean([
            deletion_curve(model, x, pred, rng.random(x.shape[-2:]), steps=DELETION_STEPS, rng=rng)
            for _ in range(RANDOM_REPEATS)
        ], axis=0)

    names = class_names(dataset)
    return jsonify({
        "dataset": dataset,
        "input": rounded(x[0, 0].numpy()),
        "prediction": {"index": pred, "label": names[pred], "confidence": round(float(probs[pred]), 4)},
        "true_label": None if true_label is None else {"index": true_label, "label": names[true_label]},
        "probabilities": rounded(probs.numpy(), 4),
        "explanations": explanations,
        "random_baseline": {
            "deletion_auc": round(area_under_curve(random_curve), 3),
            "deletion_curve": rounded(random_curve),
        },
    })


@app.post("/api/sanity")
def api_sanity():
    dataset, x, _ = read_input()
    mask = x[0, 0].numpy() > 0  # object pixels
    with LOCK:
        model = get_model(dataset)
        randomized = get_random_model(dataset)
        pred, _ = predict(model, x)
        results = []
        for name, (title, _) in METHODS.items():
            trained_map = explain(model, x, pred, name)
            random_map = explain(randomized, x, pred, name)
            correlation = rank_correlation(trained_map, random_map, mask)
            results.append({
                "method": name,
                "title": title,
                "heatmap": rounded(normalize(random_map)),
                "correlation": None if np.isnan(correlation) else round(correlation, 3),
            })
    return jsonify({"dataset": dataset, "explanations": results})


@app.get("/api/benchmark/<dataset>")
def api_benchmark(dataset):
    if dataset not in DATASETS:
        raise ApiError("Unknown dataset.", 404)
    path = SUMMARY_DIR / f"{dataset}_summary.csv"
    if not path.exists():
        return jsonify({"dataset": dataset, "rows": [], "columns": [],
                        "message": "No results yet. Run evaluate_explanations.py, then analyze_results.py."})
    table = pd.read_csv(path)
    columns = [(key, title) for key, title in BENCHMARK_COLUMNS if key in table]
    table = table[[key for key, _ in columns]]
    table = table.astype(object).where(table.notna(), None)  # NaN is not valid JSON
    return jsonify({"dataset": dataset, "columns": columns, "rows": table.to_dict(orient="records")})


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)
