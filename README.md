# Emotion detection

A multi-label emotion detection project trained on GoEmotions with a DeBERTa v3 model. The repository includes the original training notebook and a Streamlit app for local inference.

## What it does

- Predicts one or more emotions from text
- Uses a transformer classifier with sigmoid outputs
- Applies a tuned confidence threshold for multi-label prediction
- Shows detected emotions and confidence scores
- Displays a probability chart for all 28 GoEmotions labels

## Tech stack

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- pandas
- Altair

## Project files

```text
app.py                    Streamlit inference app
emotion_detection.ipynb   Training and evaluation notebook
label_map.json            GoEmotions label mapping
requirements.txt          Runtime dependencies
artifacts/                Local model files, ignored by Git
```

## Run locally

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app expects trained model files in `artifacts/`. Keep those files out of normal Git commits because the model checkpoint is large.

## Training notebook

Open `emotion_detection.ipynb` in VS Code, Jupyter, or Colab to review the training workflow. The notebook covers dataset loading, preprocessing, transformer training, threshold tuning, final evaluation, per-label metrics, and misclassified examples.

## Deployment

Use Streamlit Community Cloud or Hugging Face Spaces for a free recruiter demo.

For Hugging Face Spaces, create a Streamlit Space and upload:

- `app.py`
- `requirements.txt`
- `label_map.json`
- the model artifacts, preferably with Git LFS


## Limitations

- The model checkpoint is too large for a normal Git commit
- Inference runs on CPU unless deployed with GPU hardware
- The app is a demo, not a moderation or clinical tool
