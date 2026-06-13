"""
app.py — Neuronal Differentiation Analyzer
===========================================
Streamlit frontend for the CoND pipeline.

Usage:
    cd "Capstone Project"
    streamlit run app.py

Requires:
    - outputs/models_CoND/rf_model.pkl   (produced by Module 8 in the notebook)
    - outputs/models_CoND/scaler.pkl     (saved alongside the RF model)
    - src/pipeline_utils.py              (shared preprocessing + feature code)
"""

import sys
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib

import streamlit as st
from pathlib import Path

warnings.filterwarnings('ignore')

# Path setup — makes 'from src.pipeline_utils import ...' work
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_utils import preprocess_image, extract_features_from_array

# Constants
MODEL_PATH   = PROJECT_ROOT / 'outputs/models_CoND/rf_model.pkl'
SCALER_PATH  = PROJECT_ROOT / 'outputs/models_CoND/scaler.pkl'

BASE_FEATURES = [
    'soma_area_mean', 'circularity_mean',
    'neurite_length', 'branch_count', 'neurite_density',
]
DELTA_COLS = [f'delta_{f}' for f in BASE_FEATURES]
DELTA_LABELS = [
    'Δ Soma Area', 'Δ Circularity',
    'Δ Neurite Length', 'Δ Branch Count', 'Δ Neurite Density',
]

# Page config
st.set_page_config(
    page_title='Neuronal Differentiation Analyzer',
    page_icon='🧬',
    layout='wide',
)

st.title('🧬 Neuronal Differentiation Analyzer')

st.warning(
    '⚠️ **Research Prototype — Not for clinical or production use.**  \n'
    'This tool was developed as a Software Engineering capstone thesis project. '
    'It is trained on a public academic dataset (funalab/CoND, SH-SY5Y neuroblastoma cells) '
    'using population-level image pairing. Predictive accuracy is limited (R² ≈ 0.08–0.13). '
    'Feature importance findings are hypothesis-generating only and have not been clinically validated.',
    icon='⚠️',
)

st.markdown(
    'Upload a **before** image (day 0, untreated) and an **after** image '
    '(post-treatment) to predict the differentiation stage and see which '
    'morphological changes drove the result.'
)

# Load model
@st.cache_resource(show_spinner='Loading model...')
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)

model, scaler = load_model()

if model is None:
    st.error(
        '**Model not found.**  '
        'Run the notebook through **Module 8** first — it saves '
        '`outputs/models_CoND/rf_model.pkl` automatically.'
    )
    st.stop()

# File upload
st.subheader('Step 1 — Upload Images')
col_up1, col_up2 = st.columns(2)
with col_up1:
    before_file = st.file_uploader(
        'Before image (day 0 — untreated baseline)',
        type=['png', 'tif', 'tiff', 'jpg', 'jpeg'],
        key='before',
    )
with col_up2:
    after_file = st.file_uploader(
        'After image (post-treatment)',
        type=['png', 'tif', 'tiff', 'jpg', 'jpeg'],
        key='after',
    )

if before_file is None or after_file is None:
    st.info('Upload both images above to continue.')
    st.stop()

# Process uploaded images
def process_upload(uploaded_file):
    """Save → preprocess → extract features + segmentation intermediates."""
    suffix = Path(uploaded_file.name).suffix or '.png'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    img_uint8 = preprocess_image(tmp_path)
    result    = extract_features_from_array(img_uint8)
    tmp_path.unlink(missing_ok=True)
    return img_uint8, result

with st.spinner('Preprocessing and segmenting images...'):
    before_img, before_result = process_upload(before_file)
    after_img,  after_result  = process_upload(after_file)

if before_result is None or after_result is None:
    st.error(
        'Could not detect cell regions in one or both images. '
        'Please check that the images contain visible cell structures.'
    )
    st.stop()

# 4-panel visualization
def render_pipeline_panels(img_uint8, result, title):
    from skimage.color import label2rgb

    binary   = result['binary']
    labeled  = result['labeled']
    skeleton = result['skeleton']
    img_f    = img_uint8.astype(np.float32) / 255.0

    region_overlay = label2rgb(labeled, image=img_f, bg_label=0)
    skel_overlay   = np.stack([img_f] * 3, axis=-1)
    skel_overlay[skeleton] = [1.0, 0.15, 0.15]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    panels = [
        (img_uint8,       'gray',  'Preprocessed\n(CLAHE + Gaussian blur)'),
        (binary,          'gray',  'Segmentation Mask\n(soma detection)'),
        (region_overlay,  None,    'Detected Regions\n(each cell = colour)'),
        (skel_overlay,    None,    'Neurite Skeleton\n(red = neurites)'),
    ]
    for ax, (data, cmap, label) in zip(axes, panels):
        if cmap:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)
        ax.set_title(label, fontsize=8, fontweight='bold')
        ax.axis('off')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.subheader('Step 2 — Image Processing Pipeline')

st.markdown('**Before image** — baseline morphology')
render_pipeline_panels(before_img, before_result, 'Before — Preprocessing & Segmentation')

st.markdown('**After image** — post-treatment morphology')
render_pipeline_panels(after_img, after_result, 'After — Preprocessing & Segmentation')

# Compute deltas
bf = before_result['features']
af = after_result['features']
deltas = {f'delta_{k}': af[k] - bf[k] for k in BASE_FEATURES}

st.subheader('Step 3 — Morphological Changes (Δ After − Before)')
metric_cols = st.columns(5)
for col, col_name, label in zip(metric_cols, DELTA_COLS, DELTA_LABELS):
    val = deltas[col_name]
    col.metric(label, f'{val:+.3f}')

with st.expander('Raw feature values (before / after / delta)'):
    raw_df = pd.DataFrame({
        'Feature': BASE_FEATURES,
        'Before' : [round(bf[k], 4) for k in BASE_FEATURES],
        'After'  : [round(af[k], 4) for k in BASE_FEATURES],
        'Delta'  : [round(af[k] - bf[k], 4) for k in BASE_FEATURES],
    })
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

# Prediction
# Random Forest was trained on unscaled delta features (scaler was only for SVR)
X = np.array([[deltas[c] for c in DELTA_COLS]])
pred_day = float(np.clip(model.predict(X)[0], 0, 22))

st.subheader('Step 4 — Differentiation Stage Prediction')

res_col1, res_col2 = st.columns([1, 2])
with res_col1:
    st.metric('Predicted day', f'{pred_day:.1f}', help='Estimated days since treatment start (range: 0–22)')
    progress_pct = pred_day / 22.0
    st.progress(progress_pct)

with res_col2:
    if pred_day <= 4:
        st.info('**Early stage** (day 0–4)  \nMinimal morphological change — cells appear similar to baseline.')
    elif pred_day <= 12:
        st.warning('**Mid-stage differentiation** (day 4–12)  \nModerate neurite outgrowth and branching beginning.')
    else:
        st.success('**Late-stage differentiation** (day 12–22)  \nSubstantial neurite development and morphological remodelling.')

# Feature importance
st.subheader('Step 5 — Feature Contribution')
st.markdown(
    'The bar chart shows each feature\'s importance in the Random Forest model, '
    'signed by the direction of change (red = pushes prediction higher, blue = lower).'
)

importances = model.feature_importances_
delta_vals  = np.array([deltas[c] for c in DELTA_COLS])
signed_imp  = importances * np.sign(delta_vals)

fig, ax = plt.subplots(figsize=(8, 3.6))
bar_colors = ['#d9534f' if v > 0 else '#5b9bd5' for v in signed_imp]
y_pos = np.arange(len(DELTA_LABELS))
ax.barh(y_pos, signed_imp, color=bar_colors, edgecolor='white', height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(DELTA_LABELS, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
ax.set_xlabel('Feature importance × direction of change', fontsize=9)
ax.set_title('Feature Contribution to Prediction', fontsize=10)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Feature importance legend
with st.expander('What do the features mean?'):
    st.markdown("""
| Feature | Meaning |
|---------|---------|
| Δ Soma Area | Change in average cell body size (pixels²) |
| Δ Circularity | Change in cell roundness (1 = perfect circle, 0 = elongated) |
| Δ Neurite Length | Change in total neurite skeleton length (pixels) |
| Δ Branch Count | Change in number of neurite branch points |
| Δ Neurite Density | Change in neurite length / total cell area |
""")

st.caption(
    'Model: Random Forest Regressor (cross-validated R² = 0.018 ± 0.058, single-split R² = 0.081) '
    '| Dataset: funalab/CoND — SH-SY5Y neuroblastoma, 1826 real longitudinal pairs + jitter augmentation to 7,000 rows '
    '| Features: 5 morphological delta features | Research prototype only'
)
