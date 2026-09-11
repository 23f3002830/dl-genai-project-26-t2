# Smart MCQ Solver Challenge – Top 3 Answer Prediction

DL & GenAI Course Project | IIT Madras
Author: Prince Patel (23f3002830)

Kaggle Score (latest, v37): **0.75602** | Target cutoff: 0.7300 ✅ (+0.0260 above cutoff)

---

## 1. Problem Statement

The Kaggle competition "Smart MCQ Solver Challenge" gives multiple-choice questions (5 options: A–E) from mixed domains and asks us to predict the **top 3 most likely correct options** for each question. Scoring is done using **MAP@3 (Mean Average Precision @ 3)**, so getting the correct answer in rank 1 is worth more than getting it in rank 3, and order matters.

Goal for this project: build a working end-to-end pipeline that beats the target cutoff of 0.7300, while trying out a mix of techniques taught in the DL & GenAI course — a model trained from scratch, a pretrained transformer embedding model, and a classic gradient boosting model, and then combine all of them.

## 2. Dataset

| | |
|---|---|
| Train set | 2,000 questions (`train.csv`) |
| Test set | 500 questions (`test.csv`) |
| Options per question | 5 (A, B, C, D, E) |
| Missing values | 0 (fully clean dataset) |
| Class balance | A: 18.5%, B: 24.5%, C: 23.0%, D: 18.0%, E: 16.0% |

### EDA findings that shaped the feature engineering
- The dataset has zero missing values, so no imputation was needed.
- **Longest-option bias:** the longest option out of the 5 choices turns out to be the correct answer ~40.55% of the time, more than double the 20% random baseline. This alone was a strong enough signal that I built explicit length-based features (character length, difference from mean option length, and length rank) for both the PyTorch model and LightGBM.
- Average option length correlates fairly well with prompt-option word overlap (r ≈ 0.49), but prompt length itself is basically uncorrelated with option length (r ≈ 0.01), so I didn't need to worry about prompt length leaking answer info.
- A KDE plot of prompt-character-length for train vs test showed near-identical distributions, meaning there's no real train/test distribution shift to worry about.

## 3. Approach — Overview

The final pipeline has two parts stacked together:

1. **A 3-model ML/DL ensemble** (trained with 5-fold Stratified K-Fold CV) that scores each option and ranks the top 3.
2. **A hybrid text-retrieval matcher** that checks whether a test question closely matches a training question (via Jaccard + SequenceMatcher text similarity). If a strong match is found, that historical answer is placed at Rank 1, and the ensemble fills in the rest.

```
Test Question + 5 Options
        │
        ▼
Jaccard Prompt Match (≥ 0.85)? ──No──► 3-Model Ensemble Fallback
        │ Yes
        ▼
SequenceMatcher Option Match (≥ 0.50)
        │
        ▼
Place matched option at Rank 1, use ensemble scores for Rank 2 & 3
        │
        ▼
Final Top-3 submission.csv
```

This matcher found matches for **374 / 500 (74.8%)** of test questions, which gave a big free accuracy boost since those Rank-1 predictions are essentially guaranteed correct.

## 4. Feature Engineering (for Model 1 & Model 3)

Built a 262-dimensional feature vector per (prompt, option) pair:

| Feature group | What it is | Dims |
|---|---|---|
| Prompt & option SVD blocks | TF-IDF (5,000 max features, 1–2 grams) → TruncatedSVD(64) embeddings for prompt and option separately | 128 |
| SVD interaction blocks | `\|p_svd - o_svd\|` and `p_svd * o_svd` (Hadamard product) | 128 |
| Similarity metrics | cosine similarity between p_svd & o_svd + raw TF-IDF similarity | 2 |
| Option length features | char length, diff from mean option length, length rank (0–4) | 3 |
| Word overlap | count of shared vocab tokens between prompt & option | 1 |
| **Total** | | **262** |

## 5. Models

### Model 1 — `OptionScorerNN` (PyTorch, trained from scratch)
A simple 3-layer MLP that scores a single option given the 262-dim feature vector.

```
Input (262) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
            → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
            → Linear(1)   → option score
```
- Optimizer: Adam, lr = 5e-3, loss = BCEWithLogitsLoss
- 30 epochs/fold, 5-fold StratifiedKFold
- Val MAP@3: **0.98267**

### Model 2 — Pretrained SentenceTransformer + MLP head
Uses `intfloat/e5-small-v2` (frozen, 384-dim output) to embed the whole question (prompt + all 5 options, prefixed with `passage:`), then trains a small MLP head on top.

```
Text (passage: prompt + options) → e5-small-v2 (frozen, 33M params) → 384-dim embedding
         → Linear(256) → ReLU → Dropout(0.2)
         → Linear(64)
         → Linear(5)  → 5-way option logits
```
- Optimizer: Adam, lr = 1e-3, loss = CrossEntropyLoss, batch size 32
- 15 epochs/fold, 5-fold StratifiedKFold
- Val MAP@3: **0.99433** (best single model)

### Model 3 — LightGBM (GBDT, binary classifier)
Same 262-dim features as Model 1, but each option row is scored independently as a binary "is this the right option" problem, then reshaped back into 5 scores per question.
- `objective='binary'`, `max_depth=5`, `num_leaves=31`, `learning_rate=0.05`, `feature_fraction=0.8`
- Up to 150 boosting rounds with 15-round early stopping
- Val MAP@3: **0.96258**

### Ensembling — Powell weight optimization
Used `scipy.optimize.minimize` (Powell's method) on out-of-fold probabilities to find the blend weights that maximize OOF MAP@3:

```
final_prob = 0.5010 * PyTorch_NN + 0.4892 * SentenceTransformer + 0.0098 * LightGBM
```

This pushed the OOF MAP@3 up to **0.99808**.

## 6. Results

| Model | Val Accuracy | Val F1-macro | Val MAP@3 |
|---|---|---|---|
| PyTorch NN (scratch) | 0.9690 | 0.9692 | 0.98267 |
| e5-small-v2 + MLP head | 0.9920 | 0.9919 | 0.99433 |
| LightGBM | 0.9390 | 0.9396 | 0.96258 |
| **Powell Weighted Ensemble** | – | – | **0.99808** |

Note: The OOF/validation numbers above are much higher than the actual Kaggle leaderboard score (0.75602). This gap mostly comes from the fact that the 5-fold CV setup and the text-matcher were tuned on the same 2,000 training rows, and the real test set behaves quite differently — some of that CV performance doesn't generalize. Worth keeping in mind for future iterations.

### Kaggle score progression across notebook versions

| Version | Change | Score |
|---|---|---|
| v3 | Baseline: TF-IDF cosine + MiniLM + DeBERTa-v3-small fine-tune | 0.38861 |
| v17 | Added 262-dim features + PyTorch NN + cross-encoder + LightGBM | 0.74272 |
| v23 | Added EDA insights + option text normalization | 0.74522 |
| v24 | Added option-length bias features + cosine similarity matrix | 0.74979 |
| v27 | Switched to pretrained SentenceTransformer + MLP head, added checkpoint averaging | 0.75062 |
| **v37 (final)** | Added two-stage hybrid text matcher + Powell ensemble | **0.75602** |

## 7. Error Analysis

A few failure modes I noticed while checking wrong predictions:

- **Distractor prefix confusion** — options that differ only by a small prefix (e.g. "hypo-" vs "hyper-") are hard for the length/overlap features to tell apart. The transformer embeddings help here since they pick up semantic meaning, not just surface text.
- **Short correct-answer exception** — sometimes the correct answer is a short one-word option while distractors are longer, which goes against the "longest option is usually correct" heuristic. Blending in the transformer model reduces over-reliance on the length signal.
- **Option reordering** — when the same question appears with options shuffled into a different order, matching on option text (via SequenceMatcher) instead of the fixed letter (A/B/C/D/E) avoids mismatches from the retrieval stage.

## 8. Repo Structure

```
.
├── notebook.ipynb         # full training + inference pipeline (Kaggle notebook)
├── DG_T22026.pdf           # project report (this README is a condensed version of it)
├── train.csv                # provided by competition
├── test.csv                 # provided by competition
├── submission.csv           # final predictions
└── README.md
```

## 9. How to Run

1. Get `train.csv`, `test.csv`, `sample_submission.csv` from the competition data page and put them in the working directory (or `/kaggle/input/...` if running on Kaggle).
2. Install dependencies:
   ```bash
   pip install torch scikit-learn lightgbm sentence-transformers wandb pandas numpy seaborn matplotlib
   ```
3. (Optional) If you want live logging, set your Weights & Biases API key as an environment variable / Kaggle secret named `WB` or `WANDB_API_KEY`. The notebook will silently skip logging if it isn't found.
4. Run the notebook top to bottom. It will:
   - do EDA and print out feature correlations,
   - build the 262-dim TF-IDF/SVD features,
   - train all 3 models with 5-fold CV,
   - fit the Powell ensemble weights,
   - run the hybrid text matcher, and
   - write `submission.csv`.

## 10. Future Work

- Fine-tune a larger cross-encoder (e.g. DeBERTa-v3-large or ModernBERT) directly on question-option pairs instead of relying on frozen embeddings.
- Try a small RAG setup — retrieve relevant Wikipedia / domain-specific passages before scoring options, since a lot of the questions look knowledge-heavy.
- Use Optuna instead of manually picking learning rates / dropout for each model — didn't have time to properly tune this round.
- Investigate the CV vs LB gap more carefully — probably some leakage or overfitting in the text-matching stage that inflates OOF numbers.

## 11. References

- Wang, L. et al. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training.* arXiv:2212.03533.
- Vaswani, A. et al. (2017). *Attention is All You Need.* NeurIPS 30.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS 30.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
- Biewald, L. (2020). *Experiment Tracking with Weights and Biases.* wandb.ai.

---

*Course project for DL & GenAI, IIT Madras. Feel free to open an issue if you spot a bug in the pipeline.*
