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

import numpy as np
from pathlib import Path
from PIL import Image
from skimage import measure, morphology, exposure
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.transform import resize as skimage_resize
from scipy.ndimage import gaussian_filter, convolve
from scipy import ndimage as ndi

# Image loading

def load_image(path):
    """Load a PNG or TIFF file as a grayscale float32 array normalised to [0, 1].

    Handles 16-bit TIFFs (common in phase contrast microscopy) correctly —
    PIL preserves full bit depth, and min-max normalisation maps the full
    dynamic range to [0, 1].
    """
    img = Image.open(str(path)).convert('L')
    img_arr = np.array(img, dtype=np.float32)
    if img_arr.max() > img_arr.min():
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
    return img_arr

def _preprocess_and_save(img_path, save_path, target_h, target_w):
    """Preprocess a single image and save as PNG.

    Pipeline: load → resize → CLAHE (local contrast enhancement) →
              Gaussian blur (noise reduction) → save.
    """
    img       = load_image(img_path)
    img       = skimage_resize(img, (target_h, target_w), anti_aliasing=True)
    img_clahe = exposure.equalize_adapthist(img, clip_limit=0.01, kernel_size=25)
    img_blur  = gaussian_filter(img_clahe, sigma=0.9)
    img_uint8 = (img_blur * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode='L').save(str(save_path))

def preprocess_image(img_path, target_h=200, target_w=200):
    """Preprocess and return image as uint8 numpy array (without saving).

    Used by the Streamlit app where we want the processed pixels directly
    rather than writing a file to disk.
    """
    img       = load_image(img_path)
    img       = skimage_resize(img, (target_h, target_w), anti_aliasing=True)
    img_clahe = exposure.equalize_adapthist(img, clip_limit=0.01, kernel_size=25)
    img_blur  = gaussian_filter(img_clahe, sigma=0.9)
    img_uint8 = (img_blur * 255).astype(np.uint8)
    return img_uint8

# Segmentation

def segment_image(img_float):
    """Segment cells from background using local deviation + watershed."""
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
    """Extract 5 core morphological features from a preprocessed image."""
    img       = np.array(Image.open(str(image_path)).convert('L'), dtype=np.float32) / 255.0
    binary    = segment_image(img)

    labeled  = measure.label(binary)
    regions  = measure.regionprops(labeled, intensity_image=img)
    if not regions:
        return None

    areas         = [r.area for r in regions]
    circularities = [4 * np.pi * r.area / (r.perimeter ** 2)
                     if r.perimeter > 0 else 0 for r in regions]

    skeleton        = skeletonize(binary)
    neurite_length  = int(skeleton.sum())
    neighbour_count = convolve(skeleton.astype(int), np.ones((3, 3), dtype=int))
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

    areas         = [r.area for r in regions]
    circularities = [4 * np.pi * r.area / (r.perimeter ** 2)
                     if r.perimeter > 0 else 0 for r in regions]

    skeleton        = skeletonize(binary)
    neurite_length  = int(skeleton.sum())
    neighbour_count = convolve(skeleton.astype(int), np.ones((3, 3), dtype=int))
    branch_points   = int(((neighbour_count >= 4) & skeleton).sum())
    total_area      = binary.sum()
    neurite_density = neurite_length / total_area if total_area > 0 else 0

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
