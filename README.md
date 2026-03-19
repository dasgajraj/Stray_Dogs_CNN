# CanineCare AI — Stray Dog Health Risk Classifier

**Das Gajraj Sharma · 20213215723 · B.Tech CSE DS 6th Sem**  
B.Tech Engineering Project · SSMDA · Powered by TensorFlow 2.x & Streamlit

---

## What This Project Does

CanineCare AI is a CNN-based image classification system that analyzes skin photographs of stray dogs and assigns one of three health risk categories:

| Category | What it means |
|---|---|
| **Healthy (Low Risk)** | No significant dermal abnormalities. Monitoring only. |
| **Medium Risk** | Fungal infection or allergic dermatitis patterns detected. Follow-up advised. |
| **Critical Risk** | Bacterial infection or severe mange. Immediate veterinary intervention required. |

A custom Streamlit web app wraps the model with a live animated inference pipeline, CNN architecture diagram, and per-class probability breakdown.

---

## Project Structure

```
caninecare-ai/
│
├── app.py                        # Streamlit web application (main entry point)
├── dog_health_cnn.keras          # Trained model weights (download from Colab after training)
├── StrayDogCNN_SSDM.ipynb        # Google Colab training notebook
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Model Architecture

The CNN follows a classic feature-extraction + classification head design:

```
Input (150×150×3)
    │
    ├─ Conv2D(32, 3×3) + ReLU → MaxPool(2×2)
    ├─ Conv2D(64, 3×3) + ReLU → MaxPool(2×2)
    ├─ Conv2D(128, 3×3) + ReLU → MaxPool(2×2)
    │
    ├─ Flatten
    ├─ Dense(512) + ReLU
    ├─ Dropout(0.5)
    └─ Dense(3) + Softmax   →   [Critical, Healthy, Medium]
```

**Key design choices:**
- Three progressively deeper conv blocks (32 → 64 → 128 filters) to capture coarse-to-fine skin texture features
- MaxPooling halves spatial dimensions after each block, reducing compute and forcing spatial invariance
- Dropout (p=0.5) before the output head prevents the dense layer from memorising training data
- Softmax output gives calibrated probabilities across all three classes

**Class index order** (determined by alphabetical folder sorting during `flow_from_directory`):

| Index | Folder name | Label |
|---|---|---|
| 0 | `critical_risk` | Critical Risk |
| 1 | `low_risk` | Healthy (Low Risk) |
| 2 | `medium_risk` | Medium Risk |

---

## Dataset

Two Kaggle datasets are combined and balanced to 1,000 images per class (3,000 total):

| Source | Used for |
|---|---|
| [Stanford Dogs Dataset](https://www.kaggle.com/datasets/miljan/stanford-dogs-dataset-traintest) | Low Risk / Healthy class |
| [Dogs Skin Disease Dataset](https://www.kaggle.com/datasets/yashmotiani/dogs-skin-disease-dataset) | Medium Risk (Fungal + Allergic) and Critical Risk (Bacterial) |

**Balancing strategy:** The disease dataset has fewer images per category, so images are looped with modulo indexing (`files[i % len(files)]`) to reach the 1,000-image target without discarding any data.

**Train / Validation split:** 80% training (2,400 images) / 20% validation (600 images), set via `ImageDataGenerator(validation_split=0.2)`.

**Augmentation applied during training:**
- Random rotation ±20°
- Width / height shift ±20%
- Horizontal flip
- Fill mode: nearest

---

## Training (Google Colab)

The model is trained on Colab's free T4 GPU. Local training on CPU (e.g. HP EliteBook with 8 GB RAM) would take hours per epoch — Colab reduces this to minutes.

### Steps to train

1. Open `StrayDogCNN_SSDM.ipynb` in [Google Colab](https://colab.research.google.com)
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Set up Kaggle API secrets:
   - Click the 🔑 key icon on the left sidebar
   - Add secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`
   - Toggle **Notebook access** ON for both
4. Run all cells in order:
   | Cell | What it does |
   |---|---|
   | 1 | Loads Kaggle credentials, creates folder structure |
   | 2 | Downloads Stanford + Disease datasets, balances to 1,000/class |
   | 3 | Builds data generators with augmentation |
   | 4 | Defines the CNN architecture |
   | 5 | Trains with EarlyStopping + ReduceLROnPlateau callbacks |
   | 6 | Plots accuracy and loss curves |
   | 7 | Packages and downloads the trained `.keras` file |

5. After training completes, `dog_health_cnn.keras` will automatically download to your machine

### Training callbacks

| Callback | Setting | Purpose |
|---|---|---|
| `EarlyStopping` | patience=5, monitor=val_accuracy | Stops if val accuracy stalls for 5 epochs |
| `ReduceLROnPlateau` | patience=3, factor=0.5 | Halves learning rate if val loss plateaus |

Maximum epochs: 30 (usually stops earlier via EarlyStopping)

---

## Running the App Locally

### Prerequisites

Python 3.9–3.11 recommended. TensorFlow does not yet have stable wheels for Python 3.12+.

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd caninecare-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the model file

Copy `dog_health_cnn.keras` (downloaded from Colab) into the project root — the same folder as `app.py`.

```
caninecare-ai/
├── app.py
├── dog_health_cnn.keras   ← here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## requirements.txt

```
streamlit>=1.32.0
tensorflow>=2.13.0
Pillow>=10.0.0
numpy>=1.24.0
requests>=2.31.0
streamlit-lottie>=0.0.5
```

---

## How Inference Works

When you click **Execute Neural Analysis**, the app runs these steps (shown live in the animated pipeline):

1. **Load weights** — The cached model (`@st.cache_resource`) is already in memory after the first run
2. **Resize** — Input image resized to 150×150 px using Lanczos resampling
3. **Normalise** — Pixel values divided by 255.0 → range [0, 1]
4. **Forward pass** — Image tensor passed through the 3 conv blocks
5. **Dense head** — Flattened features → 512-unit Dense → Dropout → 3-unit output
6. **Softmax** — Raw logits converted to probabilities; argmax gives predicted class

**Important:** The model was trained with `flow_from_directory`, which sorts class folders alphabetically. The app's `categories` list must match this order exactly:

```python
categories = ["Critical Risk", "Healthy (Low Risk)", "Medium Risk"]
#              index 0           index 1               index 2
```

---

## Known Issues & Notes

**`quantization_config` metadata error on load**  
Some TensorFlow/Keras version mismatches cause the full `model.load()` to fail. The app works around this by reconstructing the architecture with `build_model()` and calling `model.load_weights()` instead — weights load cleanly regardless of metadata in the `.keras` container.

**Model not found**  
If `dog_health_cnn.keras` is missing, the app shows a warning and disables inference. Train the model via the Colab notebook and place the file in the project root.

**Lottie animation**  
The hero animation loads from an external URL (`lottiefiles.com`). If you're offline, it silently skips — the rest of the app works fine.

---

## Subject Context

This project was built as part of the **SSMDA (Statistics and Machine Learning Data Analysis)** subject in B.Tech CSE (Data Science), 6th Semester.
---

## License

Academic project — not licensed for commercial use.

---

*Das Gajraj Sharma · 20213215723 · B.Tech CSE DS 6th Sem · SSMDA*
