"""
Tests for src/models/classical.py and src/models/deep_learning.py
"""

import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.classical import EEGClassicalClassifier, MODEL_REGISTRY
from src.models.deep_learning import (
    build_cnn,
    build_lstm,
    build_cnn_lstm,
    EEGDeepClassifier,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

N_SAMPLES   = 200
N_FEATURES  = 80
N_CHANNELS  = 4
EPOCH_LEN   = 128   # samples per epoch


@pytest.fixture
def binary_feature_data():
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_classes=2, random_state=42
    )
    return X.astype(np.float32), y.astype(int)


@pytest.fixture
def multiclass_feature_data():
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_classes=5, n_informative=10, random_state=42
    )
    return X.astype(np.float32), y.astype(int)


@pytest.fixture
def binary_epoch_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((N_SAMPLES, N_CHANNELS, EPOCH_LEN)).astype(np.float32)
    y = rng.integers(0, 2, N_SAMPLES)
    return X, y


@pytest.fixture
def multiclass_epoch_data():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((N_SAMPLES, N_CHANNELS, EPOCH_LEN)).astype(np.float32)
    y = rng.integers(0, 5, N_SAMPLES)
    return X, y


# ── Classical models ──────────────────────────────────────────────────────────

class TestEEGClassicalClassifier:

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_train_predict_shape(self, model_name, binary_feature_data):
        X, y = binary_feature_data
        clf = EEGClassicalClassifier(model_name=model_name)
        clf.train(X, y)
        preds = clf.predict(X)
        assert preds.shape == y.shape

    @pytest.mark.parametrize("model_name", ["svm", "random_forest"])
    def test_predict_proba_shape(self, model_name, binary_feature_data):
        X, y = binary_feature_data
        clf = EEGClassicalClassifier(model_name=model_name)
        clf.train(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (N_SAMPLES, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_evaluate_returns_metrics(self, binary_feature_data):
        X, y = binary_feature_data
        clf = EEGClassicalClassifier(model_name="svm")
        clf.train(X, y)
        result = clf.evaluate(X, y, target_names=["neg", "pos"])
        assert "accuracy" in result
        assert "f1_macro" in result
        assert "confusion_matrix" in result
        assert 0 <= result["accuracy"] <= 1

    def test_multiclass_random_forest(self, multiclass_feature_data):
        X, y = multiclass_feature_data
        clf = EEGClassicalClassifier(model_name="random_forest")
        clf.train(X, y)
        preds = clf.predict(X)
        assert len(np.unique(preds)) >= 2

    def test_save_and_load(self, binary_feature_data):
        X, y = binary_feature_data
        clf = EEGClassicalClassifier(model_name="lda")
        clf.train(X, y)
        preds_before = clf.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            clf.save(path)
            loaded = EEGClassicalClassifier.load(path, model_name="lda")
            preds_after = loaded.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_predict_before_train_raises(self, binary_feature_data):
        X, _ = binary_feature_data
        clf = EEGClassicalClassifier(model_name="svm")
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(X)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            EEGClassicalClassifier(model_name="unicorn_model")

    def test_cross_validate(self, binary_feature_data):
        X, y = binary_feature_data
        clf = EEGClassicalClassifier(model_name="knn")
        result = clf.cross_validate(X, y, cv=3)
        assert "mean" in result
        assert 0 <= result["mean"] <= 1


# ── Deep learning models ──────────────────────────────────────────────────────

class TestDeepLearningModels:

    def test_cnn_output_shape_binary(self, binary_epoch_data):
        X, y = binary_epoch_data
        model = build_cnn(input_shape=(N_CHANNELS, EPOCH_LEN), n_classes=2)
        out = model.predict(X[:4], verbose=0)
        assert out.shape == (4, 1)

    def test_cnn_output_shape_multiclass(self, multiclass_epoch_data):
        X, y = multiclass_epoch_data
        model = build_cnn(input_shape=(N_CHANNELS, EPOCH_LEN), n_classes=5)
        out = model.predict(X[:4], verbose=0)
        assert out.shape == (4, 5)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)

    def test_lstm_output_shape(self, binary_epoch_data):
        X, y = binary_epoch_data
        model = build_lstm(
            input_shape=(N_CHANNELS, EPOCH_LEN), n_classes=2, lstm_units=[32, 16]
        )
        out = model.predict(X[:4], verbose=0)
        assert out.shape == (4, 1)

    def test_cnn_lstm_output_shape(self, multiclass_epoch_data):
        X, y = multiclass_epoch_data
        model = build_cnn_lstm(
            input_shape=(N_CHANNELS, EPOCH_LEN), n_classes=5,
            cnn_filters=[16, 32], lstm_units=32
        )
        out = model.predict(X[:4], verbose=0)
        assert out.shape == (4, 5)

    def test_deep_classifier_train_evaluate(self, binary_epoch_data):
        X, y = binary_epoch_data
        model = build_cnn(
            input_shape=(N_CHANNELS, EPOCH_LEN), n_classes=2,
            filters=[16, 32], dropout_rate=0.0
        )
        clf = EEGDeepClassifier(model, n_classes=2, task="binary")
        clf.train(X, y, epochs=3, batch_size=32, patience=3)
        result = clf.evaluate(X, y)
        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1

    def test_deep_classifier_multiclass(self, multiclass_epoch_data):
        X, y = multiclass_epoch_data
        model = build_cnn(
            input_shape=(N_CHANNELS, EPOCH_LEN), n_classes=5,
            filters=[16], dropout_rate=0.0
        )
        clf = EEGDeepClassifier(model, n_classes=5, task="multiclass")
        clf.train(X, y, epochs=2, batch_size=32, patience=2)
        result = clf.evaluate(X, y)
        assert "confusion_matrix" in result
        assert result["confusion_matrix"].shape == (5, 5)

    def test_deep_classifier_save_load(self, binary_epoch_data):
        X, y = binary_epoch_data
        model = build_cnn(
            input_shape=(N_CHANNELS, EPOCH_LEN), n_classes=2, filters=[8]
        )
        clf = EEGDeepClassifier(model, n_classes=2)
        clf.train(X, y, epochs=1, batch_size=32, patience=1)
        preds_before = (model.predict(X[:4], verbose=0).squeeze() > 0.5).astype(int)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.keras")
            clf.save(path)
            loaded = EEGDeepClassifier.load(path, n_classes=2)
            preds_after = (loaded.model.predict(X[:4], verbose=0).squeeze() > 0.5).astype(int)

        np.testing.assert_array_equal(preds_before, preds_after)
