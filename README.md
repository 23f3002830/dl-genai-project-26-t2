# dl-genai-project-26-t2

# Smart MCQ Solver Challenge

## Name

Prince Patel

## Student ID

23F3002830

## Project Overview

This repository contains the work for the **Smart MCQ Solver Challenge**, a Kaggle machine learning project. The objective is to build a model that can predict the most likely correct answers for multiple-choice questions.

## Folder Structure

```text
Smart-MCQ-Solver-Challenge/
│
├── dl-23f3002830-notebook-t22026.ipynb (Model Pipeline & Experiments)
├── README.md (Project Documentation)
└── requirements.txt (Dependencies)
```

## Models & Methodology

The project implements two primary models for scoring and ranking multiple-choice answers:

1. **Neural Network Option Scorer (OptionScorerNN)**:
   - Built using PyTorch.
   - Uses TF-IDF representation of prompts and options combined with Truncated SVD dimensionality reduction.
   - Features include text overlap, Jaccard similarity, and vector similarity metrics.
   - Trained with binary cross-entropy loss to score the likelihood of each option.

2. **LightGBM Ranker**:
   - Built using LightGBM.
   - Uses a tree-based ensemble to rank options based on similarity, length, and overlap statistics.
   - Fine-tuned using cross-validation.

## Getting Started

### Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

### Running the Notebook

You can open and run the Jupyter notebook `dl-23f3002830-notebook-t22026.ipynb` in your preferred notebook environment (Jupyter, VS Code, or Kaggle) to reproduce the preprocessing, training, and evaluation steps.
