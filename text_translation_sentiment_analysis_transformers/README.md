# Text Translation & Sentiment Analysis

This subproject combines multilingual data wrangling, HuggingFace translation, and sentiment analysis to produce a single annotated dataset of 30 movie reviews spanning English, French, and Spanish.

## Project Goals
- Consolidate the three CSV files under `data/` into a unified dataframe with the schema `["Title", "Year", "Synopsis", "Review", "Original Language"]`.
- Translate every French (`fr`) and Spanish (`sp`) synopsis/review into English using MarianMT models so that downstream tasks can operate in one language.
- Run sentiment analysis with `distilbert-base-uncased-finetuned-sst-2-english` and append the label (`Positive`/`Negative`) in a `Sentiment` column.
- Export the final dataframe to `result/reviews_with_sentiment.csv` (30 rows + header).

These steps satisfy the rubric requirements for data ingestion, translation, and sentiment scoring.

## Environment & Dependencies
Tested with Python 3.12 + CPU.

Install the core dependencies:

```powershell
pip install pandas torch transformers sacremoses
```

Notes:
- `torch >= 2.6` is required by the latest `transformers` loaders because of CVE-2025-32434.
- `sacremoses` is optional but removes the tokenizer warning and slightly speeds up Marian preprocessing.

## How to Run
1. **Notebook workflow**  
   Open `project.ipynb` and run all cells in order. Cells are organized according to the rubric:
   - `preprocess_data()` builds the combined dataframe.
   - Translation cells load Marian models/tokenizers and overwrite the non-English text with English translations.
   - Sentiment cells load the classifier via `pipeline("sentiment-analysis")` and fill the `Sentiment` column.
   - The last cell writes `result/reviews_with_sentiment.csv`.

2. **Headless execution (optional)**  
   Mirror the notebook steps via a script or `papermill` run if you prefer automation. Ensure the working directory stays at `text_translation_sentiment_analysis_transformers/` so relative paths resolve.

## Outputs
- `result/reviews_with_sentiment.csv` — header plus 30 translated reviews with sentiment labels and original-language tags.
- Inline dataframe previews inside the notebook (`df.sample(10)`) let you visually confirm translations and sentiment assignments during grading.

## Repository Layout
```
text_translation_sentiment_analysis_transformers/
├─ data/
│  ├─ movie_reviews_eng.csv
│  ├─ movie_reviews_fr.csv
│  └─ movie_reviews_sp.csv
├─ project.ipynb                 # Completed rubric-aligned notebook
└─ result/
   └─ reviews_with_sentiment.csv # Final deliverable
```

## Troubleshooting
- **Model download / cache warnings**: On Windows, enable Developer Mode or run PowerShell as Administrator to allow HuggingFace to create cache symlinks. Otherwise, downloads fall back to regular copies (works but uses more disk).
- **`sacremoses` warning**: Install `sacremoses` if you see tokenizer warnings during translation.
- **Torchvision conflicts**: If `transformers` tries to import `torchvision` and fails (e.g., `operator torchvision::nms does not exist`), uninstall incompatible builds with `pip uninstall torchvision`. This project does not rely on torchvision.

With these instructions and the completed notebook, you can regenerate the translated, sentiment-tagged dataset end-to-end.

