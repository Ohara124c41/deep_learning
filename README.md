# Udacity Deep Learning Portfolio

A collection of four Udacity nanodegree projects covering generative modeling, computer vision, classic digit recognition, and multilingual NLP. Each folder is self‑contained with its own notebooks, assets, and requirement files; the summaries below highlight the goal, notable work, and core tooling for quick orientation.

## Projects

### 1. GAN Face Generator (`generative_adversarial_network_face_generator`)
- **Goal**: Train a custom generative adversarial network on the CelebA subset to synthesize realistic human faces from random latent vectors.
- **Highlights**: Torchvision-based augmentation pipeline, custom `DatasetDirectory`, DCGAN-style generator/discriminator upgraded with WGAN-GP losses and gradient penalty, configurable training loop that logs losses and renders fixed-latent samples every epoch.
- **Tech**: PyTorch, torchvision, CelebA preprocessed set, NumPy, Matplotlib, Jupyter Notebook.

### 2. Landmark Classification & Social Media Tagging (`landmark_classification_tagging_social_media`)
- **Goal**: Build an end-to-end pipeline that predicts which of 50 global landmarks appears in a user photo so social-media services can auto-tag uploads.
- **Highlights**: Two complementary notebooks (CNN from scratch + transfer learning with pre-trained backbones), modular training utilities under `src/`, and an app notebook that packages the best checkpoint into an interactive predictor for arbitrary photos.
- **Tech**: PyTorch, torchvision, PIL, numpy/pandas, Matplotlib/Seaborn, Jupyter, helper scripts for training/evaluation/deployment.

### 3. MNIST Handwritten Digit Classifier (`MNIST_handwritten_digits`)
- **Goal**: Achieve >98% accuracy on MNIST using a compact multilayer perceptron while documenting every rubric checkpoint inside a single notebook.
- **Highlights**: Clean data loaders with normalization/visualization helpers, `MNISTMLP` architecture (BatchNorm + dropout), AdamW training with learning-rate scheduling and fine-tuning pass, saved `mnist_mlp_state_dict.pth` for reuse.
- **Tech**: PyTorch, torchvision (datasets), Matplotlib, Jupyter Notebook.

### 4. Text Translation & Sentiment Analysis with Transformers (`text_translation_sentiment_analysis_transformers`)
- **Goal**: Merge 30 multilingual movie reviews, translate non-English text to English, and attach sentiment labels to deliver a unified analytics-ready CSV.
- **Highlights**: MarianMT translation workflow, Hugging Face sentiment pipeline, reproducible notebook that ingests CSVs, tracks provenance columns, and exports `result/reviews_with_sentiment.csv`.
- **Tech**: Python, pandas, Hugging Face Transformers (MarianMT, DistilBERT SST-2), PyTorch, sacremoses, Jupyter.

## Getting Started
```bash
# clone repo and choose a project folder
pip install -r <project>/requirements.txt
# launch jupyter or run the provided scripts/notebooks
```
Each subproject README/notebook explains the exact runtime steps, optional GPU usage, and expected outputs. Feel free to explore them independently or mix and match components for your own experiments.
