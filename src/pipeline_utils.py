"""
pipeline_utils.py

Shared utility functions for the Neuronal Morphology Pipeline.
Imported by both pipeline_CoND.ipynb and app.py so that core logic
lives in one place and stays in sync.

Functions

load_image              — load any TIFF/PNG as grayscale float32 [0,1]
_preprocess_and_save    — resize → CLAHE → Gaussian blur → save PNG
segment_image           — background subtraction + watershed segmentation
extract_features        — compute 5 morphological features from one image
"""

import cv2
import numpy as np
from pathlib import Path

from skimage import measure, morphology
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter, convolve
from scipy import ndimage as ndi

# Image loading

def load_image(path):
    """Load a PNG or TIFF file as a grayscale float32 array normalised to [0, 1].

    Handles 16-bit TIFFs (common in phase contrast microscopy) correctly —
    OpenCV's IMREAD_GRAYSCALE preserves the full bit depth, and min-max
    normalisation maps the full dynamic range to [0, 1].
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    img = img.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    return img

def _preprocess_and_save(img_path, save_path, target_h, target_w):
    """Preprocess a single image and save as PNG.

    Pipeline: load → resize → CLAHE (local contrast enhancement) →
              Gaussian blur (noise reduction) → save.

    Parameters
    ----------
    img_path   : path-like  — source image (TIFF or PNG)
    save_path  : path-like  — destination PNG
    target_h   : int        — output height in pixels
    target_w   : int        — output width in pixels
    """
    img       = load_image(img_path)
    img       = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_uint8 = (img * 255).astype(np.uint8)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_uint8 = clahe.apply(img_uint8)
    img_uint8 = cv2.GaussianBlur(img_uint8, (3, 3), 0)
    cv2.imwrite(str(save_path), img_uint8)

def preprocess_image(img_path, target_h=200, target_w=200):
    """Preprocess and return image as uint8 numpy array (without saving).

    Used by the Streamlit app where we want the processed pixels directly
    rather than writing a file to disk.
    """
    img       = load_image(img_path)
    img       = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_uint8 = (img * 255).astype(np.uint8)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_uint8 = clahe.apply(img_uint8)
    img_uint8 = cv2.GaussianBlur(img_uint8, (3, 3), 0)
    return img_uint8

# Segmentation

def segment_image(img_float):
    """Segment cells from background using local deviation + watershed.

    Steps
    -----
    1. Estimate a smooth background with a large Gaussian blur (sigma=20).
    2. Compute per-pixel absolute deviation from the background.
    3. Threshold at the 88th percentile to isolate high-contrast cell regions.
    4. Remove small objects (<150 px) and fill small holes (<300 px).
    5. Apply distance-transform watershed to separate touching cells.

    Returns
    -------
    binary : bool ndarray — True where a cell region was detected.
    """
    background = gaussian_filter(img_float, sigma=20)
    deviation  = np.abs(img_float - background)
    thresh     = np.percentile(deviation, 88)
    binary     = deviation > thresh
    binary     = morphology.remove_small_objects(binary, min_size=150)
    binary     = morphology.remove_small_holes(binary, area_threshold=300)

    distance   = ndi.distance_transform_edt(binary)
    coords     = peak_local_max(distance, min_distance=15, labels=binary)
    mask       = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels     = watershed(-distance, markers, mask=binary)
    return labels > 0

# Feature extraction

def extract_features(image_path):
    """Extract 5 core morphological features from a preprocessed image.

    Features
    --------
    soma_area_mean   : mean cell body area (pixels²)
    circularity_mean : mean circularity = 4π·area/perimeter²  (1=circle, 0=rod)
    neurite_length   : total skeleton length (pixels)
    branch_count     : number of branch points (pixels with ≥4 skeleton neighbours)
    neurite_density  : neurite_length / total segmented area

    Parameters
    ----------
    image_path : path-like — preprocessed PNG (output of _preprocess_and_save)

    Returns
    -------
    dict with the 5 feature values, or None if no cell regions detected.
    """
    img       = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_float = img.astype(np.float32) / 255.0
    binary    = segment_image(img_float)

    labeled  = measure.label(binary)
    regions  = measure.regionprops(labeled, intensity_image=img_float)
    if not regions:
        return None

    areas         = [r.area      for r in regions]
    perimeters    = [r.perimeter for r in regions]
    circularities = [4 * np.pi * r.area / (r.perimeter ** 2)
                     if r.perimeter > 0 else 0 for r in regions]

    skeleton        = skeletonize(binary)
    neurite_length  = int(skeleton.sum())
    kernel          = np.ones((3, 3), dtype=int)
    neighbour_count = convolve(skeleton.astype(int), kernel)
    branch_points   = int(((neighbour_count >= 4) & skeleton).sum())
    total_area      = binary.sum()
    neurite_density = neurite_length / total_area if total_area > 0 else 0

    return {
        'soma_area_mean'   : float(np.mean(areas)),
        'circularity_mean' : float(np.mean(circularities)),
        'neurite_length'   : float(neurite_length),
        'branch_count'     : float(branch_points),
        'neurite_density'  : float(neurite_density),
    }

def extract_features_from_array(img_uint8):
    """Extract features directly from a uint8 numpy array (no file I/O).

    Used by the Streamlit app where the image is already in memory.
    """
    img_float = img_uint8.astype(np.float32) / 255.0
    binary    = segment_image(img_float)

    labeled  = measure.label(binary)
    regions  = measure.regionprops(labeled, intensity_image=img_float)
    if not regions:
        return None

    areas         = [r.area      for r in regions]
    perimeters    = [r.perimeter for r in regions]
    circularities = [4 * np.pi * r.area / (r.perimeter ** 2)
                     if r.perimeter > 0 else 0 for r in regions]

    skeleton        = skeletonize(binary)
    neurite_length  = int(skeleton.sum())
    kernel          = np.ones((3, 3), dtype=int)
    neighbour_count = convolve(skeleton.astype(int), kernel)
    branch_points   = int(((neighbour_count >= 4) & skeleton).sum())
    total_area      = binary.sum()
    neurite_density = neurite_length / total_area if total_area > 0 else 0

    # Also return intermediate arrays for visualisation
    labeled_out = measure.label(binary)
    return {
        'features': {
            'soma_area_mean'   : float(np.mean(areas)),
            'circularity_mean' : float(np.mean(circularities)),
            'neurite_length'   : float(neurite_length),
            'branch_count'     : float(branch_points),
            'neurite_density'  : float(neurite_density),
        },
        'binary'  : binary,
        'labeled' : labeled_out,
        'skeleton': skeleton,
    }
