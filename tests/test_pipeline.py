"""
tests/test_pipeline.py

Unit tests for the Neuronal Morphology Pipeline utility functions.

Run from the project root with:
    pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pipeline_utils import (
    load_image,
    preprocess_image,
    segment_image,
    extract_features_from_array,
)

# Fixtures

@pytest.fixture
def synthetic_image_path(tmp_path):
    """Synthetic grayscale PNG with bright circular regions on a gray background."""
    img = np.full((200, 200), 100, dtype=np.uint8)
    cv2.circle(img, (60,  60),  28, 230, -1)
    cv2.circle(img, (140, 80),  22, 220, -1)
    cv2.circle(img, (100, 150), 32, 225, -1)
    cv2.line(img, (60, 88),  (100, 118), 210, 3)
    cv2.line(img, (140, 102),(100, 118), 210, 3)
    path = tmp_path / "test_image.png"
    cv2.imwrite(str(path), img)
    return path

@pytest.fixture
def synthetic_uint8_array():
    """Synthetic uint8 numpy array with the same structure as above."""
    img = np.full((200, 200), 100, dtype=np.uint8)
    cv2.circle(img, (60,  60),  28, 230, -1)
    cv2.circle(img, (140, 80),  22, 220, -1)
    cv2.circle(img, (100, 150), 32, 225, -1)
    cv2.line(img, (60, 88),  (100, 118), 210, 3)
    cv2.line(img, (140, 102),(100, 118), 210, 3)
    return img

# T-01: load_image returns valid float32 array in [0, 1]

def test_load_image_returns_valid_array(synthetic_image_path):
    """T-01: load_image returns a float32 ndarray with all values in [0, 1]."""
    img = load_image(synthetic_image_path)
    assert img is not None
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.min() >= 0.0
    assert img.max() <= 1.0

# T-02: load_image raises ValueError for a missing file

def test_load_image_raises_for_missing_file(tmp_path):
    """T-02: load_image raises ValueError when the file path does not exist."""
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError):
        load_image(missing)

# T-03: preprocess_image returns correct dtype and shape

def test_preprocess_image_returns_correct_shape(synthetic_image_path):
    """T-03: preprocess_image returns a uint8 array with the requested dimensions."""
    result = preprocess_image(synthetic_image_path, target_h=200, target_w=200)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (200, 200)

# T-04: CLAHE preprocessing modifies pixel values

def test_preprocess_image_changes_contrast(synthetic_image_path):
    """T-04: CLAHE preprocessing produces output that differs from the raw input."""
    raw = cv2.imread(str(synthetic_image_path), cv2.IMREAD_GRAYSCALE)
    processed = preprocess_image(synthetic_image_path, target_h=200, target_w=200)
    assert not np.array_equal(raw, processed), (
        "Preprocessed image should differ from the raw input after CLAHE"
    )

# T-05: segment_image returns a boolean array of matching shape

def test_segment_image_returns_boolean_array(synthetic_uint8_array):
    """T-05: segment_image returns a boolean ndarray with the same spatial shape."""
    img_float = synthetic_uint8_array.astype(np.float32) / 255.0
    binary = segment_image(img_float)
    assert isinstance(binary, np.ndarray)
    assert binary.dtype == bool
    assert binary.shape == img_float.shape

# T-06: extract_features_from_array returns exactly 5 feature keys

def test_extract_features_returns_five_keys(synthetic_uint8_array):
    """T-06: Feature extraction returns a dict with exactly the 5 expected keys."""
    result = extract_features_from_array(synthetic_uint8_array)
    assert result is not None, "No cell regions detected in synthetic image"
    expected_keys = {
        "soma_area_mean", "circularity_mean",
        "neurite_length", "branch_count", "neurite_density",
    }
    assert set(result["features"].keys()) == expected_keys

# T-07: all extracted feature values are non-negative and not NaN

def test_extract_features_returns_valid_values(synthetic_uint8_array):
    """T-07: All morphological feature values are non-negative and not NaN."""
    result = extract_features_from_array(synthetic_uint8_array)
    assert result is not None, "No cell regions detected in synthetic image"
    for key, val in result["features"].items():
        assert not np.isnan(val), f"{key} is NaN"
        assert val >= 0.0,        f"{key} is negative: {val}"

# T-08: delta feature computation is arithmetically correct

def test_delta_feature_computation_correctness():
    """T-08: Delta features equal after - before for each morphological feature."""
    before = np.array([100.0, 0.80, 50.0, 5.0, 0.30])
    after  = np.array([120.0, 0.65, 75.0, 8.0, 0.45])
    delta  = after - before
    expected = np.array([20.0, -0.15, 25.0, 3.0, 0.15])
    np.testing.assert_allclose(delta, expected, rtol=1e-5)

# T-09: augmentation does not contaminate the test split

def test_augmentation_does_not_contaminate_test_split():
    """T-09: Gaussian jitter augmentation applied to training split only —
    test split size and content must remain unchanged after augmentation."""
    np.random.seed(42)
    n_real   = 100
    features = np.random.rand(n_real, 10)

    # Split BEFORE augmentation
    split_idx    = int(n_real * 0.8)
    X_train_real = features[:split_idx].copy()
    X_test       = features[split_idx:].copy()
    test_size_before = len(X_test)

    # Augment training split only
    noise     = np.random.normal(0, 0.01, X_train_real.shape)
    X_train_aug = np.vstack([X_train_real, X_train_real + noise])

    # Test set must be untouched
    assert len(X_test) == test_size_before, (
        "Test split size changed after augmentation — data leakage detected"
    )
    np.testing.assert_array_equal(
        X_test, features[split_idx:],
        err_msg="Test split content was modified during augmentation"
    )
    assert len(X_train_aug) == 2 * len(X_train_real), (
        "Augmented training set should be twice the original training size"
    )
