# Emotion detection

A notebook based machine learning project for multi-label emotion classification on GoEmotions. The notebook trains a transformer classifier, tunes the prediction threshold, evaluates test metrics, and saves model artifacts and analysis outputs.

## What it does

- Loads the GoEmotions dataset
- Trains a DeBERTa v3 based multi-label classifier
- Uses `BCEWithLogitsLoss` for multi-label prediction
- Tunes the decision threshold on the validation split
- Reports micro and macro metrics
- Produces per-label metrics, confusion matrices, and misclassified examples
- Saves model artifacts and output files from the notebook run

## Tech stack

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- pandas, NumPy, Matplotlib, Seaborn

## Run locally

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open `emotion_detection.ipynb` in VS Code, JupyterLab, or Jupyter Notebook and run the cells in order.

## Run in Colab

The notebook includes a Colab badge. Colab is the easiest free option if you need a GPU runtime.

## Deployment

Use Hugging Face Spaces if you want recruiters to try the model in a browser.

Recommended path:

1. Train or load the model from the notebook.
2. Create a small Gradio app with a text box and predicted emotion labels.
3. Push the Gradio app to a free Hugging Face Space.

If you do not build the Gradio demo yet, share the GitHub repo and a Colab link.

## Recruiter note

This is a good ML portfolio project if the notebook has clean outputs and metrics. To make it stronger, add a short results section with the final test metrics and include example predictions in the README.

## Limitations

- The current repo is a notebook workflow, not a packaged application
- Full training is much easier with a GPU
- Trained model files are not included in the repository
- A browser demo still needs to be built with Gradio or Streamlit
