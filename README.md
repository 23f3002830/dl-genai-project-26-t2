🧠 Smart MCQ Solver Challenge – Top 3 Answer Prediction
Deep Learning & Generative AI Project | IIT Madras

A Deep Learning and NLP-based solution for the Smart MCQ Solver Challenge on Kaggle. The goal is to predict and rank the Top 3 most likely correct options for each multiple-choice question from five candidate options: A, B, C, D, and E.

The project combines TF-IDF + SVD feature engineering, a custom PyTorch neural network, pretrained Transformer embeddings, LightGBM, Powell-based ensemble optimization, and hybrid text retrieval.

🏆 Project Highlights
Metric	Result
Final Kaggle Score	0.75602
Target Cutoff	0.73000
Improvement over Cutoff	+0.02602
Best Individual MAP@3	0.99433
Optimized OOF MAP@3	0.99808
Test Questions	500
Historical Questions Matched	374 / 500
Hybrid Matching Coverage	74.8%
Cross Validation	5-Fold StratifiedKFold

📌 Table of Contents
Project Overview
Problem Statement
Objective
Dataset
Exploratory Data Analysis
Data Preprocessing
Feature Engineering
Tokenization Strategy
Model 1 - PyTorch OptionScorerNN
Model 2 - E5-small-v2 + MLP
Model 3 - LightGBM
Model Comparison
Ensemble Optimization
Hybrid Text Matching
Complete Pipeline
Kaggle Score Progression
Experiment Tracking
Technologies Used
Project Structure
How to Run
Results
Error Analysis
Key Learnings
Future Work
References
Author
🔍 Project Overview

The Smart MCQ Solver Challenge is a multiple-choice question ranking problem where each question contains five candidate answers.

Instead of predicting only one answer, the system generates a ranking of all five options and returns the Top 3 predictions.

The project explores multiple approaches to solving this problem:

Traditional NLP feature engineering
Custom deep learning
Pretrained Transformer embeddings
Gradient boosting
Ensemble learning
Text similarity and retrieval

The final system combines these approaches into a hybrid prediction pipeline.

❓ Problem Statement

For every multiple-choice question, there are five candidate options:

A
B
C
D
E

The task is to identify the three most likely correct options and rank them.

For example:

Question ID: 101

Prediction:
B A D

This means:

Rank 1 → B
Rank 2 → A
Rank 3 → D

The competition evaluates the quality of this ranking using Mean Average Precision @ 3 (MAP@3).

🎯 Objective

The main objectives of this project were:

Perform exploratory analysis of the MCQ dataset.
Identify useful structural patterns in the answer options.
Engineer question-option interaction features.
Build a custom deep learning model from scratch.
Use pretrained Transformer sentence embeddings.
Train a LightGBM classifier.
Compare the performance of different models.
Optimize ensemble weights using Powell optimization.
Build a hybrid text-matching system.
Generate a final Top-3 Kaggle submission.

The target competition cutoff was:

MAP@3 = 0.7300

The final Kaggle submission achieved:

MAP@3 = 0.75602
📊 Dataset

The competition provides training and testing datasets containing multiple-choice questions.

Training Dataset
Rows    : 2,000
Columns : 8

Columns:

id
prompt
A
B
C
D
E
answer
Test Dataset
Rows    : 500
Columns : 7

Columns:

id
prompt
A
B
C
D
E

Each question has five candidate options.

The notebook loads the datasets using:

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub_df = pd.read_csv(sub_path)

OPTIONS = ['A', 'B', 'C', 'D', 'E']
📈 Class Distribution

The training dataset contains the following answer distribution:

Answer	Count	Percentage
A	370	18.5%
B	490	24.5%
C	460	23.0%
D	360	18.0%
E	320	16.0%
🔎 Exploratory Data Analysis

EDA was performed to understand:

Dataset completeness
Answer distribution
Prompt length
Option length
Prompt-option overlap
Relationship between option length and correctness
Train/test distribution
Missing Values

The dataset contains zero missing values across the training data.

Missing Values = 0
Dataset Completeness = 100%

Therefore, no text imputation was required.

📏 Longest Option Bias

One of the most important discoveries from the EDA was that the longest answer option is correct approximately 40.55% of the time.

For comparison:

Random baseline      = 20.00%
Longest-option rule  = 40.55%

This strong structural pattern motivated the creation of explicit option-length features.

These include:

Option length
Difference from mean option length
Option length rank
🔗 Correlation Analysis

The analysis showed:

Option Length ↔ Prompt Word Overlap
Pearson correlation ≈ 0.49

while:

Prompt Length ↔ Option Length
Pearson correlation ≈ 0.01

The prompt-length distributions of the training and test datasets also showed substantial overlap.

🧹 Data Preprocessing

The project uses different preprocessing strategies for the tabular models and Transformer model.

1. Text Normalization

For the retrieval pipeline, repetitive question prefixes were normalized.

Examples include:

pick the best possible answer:
what is

The normalized prompt is then used for historical question matching.

2. Missing Value Handling

The dataset was checked for missing values.

Result:

0 missing values

No imputation was necessary.

🧩 Feature Engineering

A major component of the project is the creation of a 262-dimensional feature vector for each question-option pair.

The feature representation combines:

Prompt SVD embeddings
Option SVD embeddings
Absolute SVD differences
Hadamard products
TF-IDF similarity
Cosine similarity
Option length
Length difference
Word overlap
Option length rank
📚 TF-IDF Feature Extraction

The project uses TfidfVectorizer with:

Maximum Features = 5,000
N-Gram Range     = (1, 2)
Stop Words       = English
Sublinear TF     = True

This creates a sparse representation of the question and answer-option text.

📉 Truncated SVD

The TF-IDF representation is compressed using:

TruncatedSVD
Components = 64

This creates dense 64-dimensional representations.

For each question-option pair:

Prompt SVD → 64 dimensions
Option SVD → 64 dimensions
🧮 262-Dimensional Feature Vector

The complete feature vector contains:

Feature Group	Dimensions
Prompt SVD	64
Option SVD	64
Absolute SVD Difference	64
Hadamard Product	64
TF-IDF Similarity	1
Cosine Similarity	1
Option Length	1
Length Difference	1
Word Overlap	1
Length Rank	1
Total	262

Therefore:

64 + 64 + 64 + 64 + 6 = 262
🔤 Tokenization Strategy

Two different tokenization approaches are used.

N-Gram Word Vectorization

Used by:

PyTorch OptionScorerNN
LightGBM

Configuration:

Unigrams + Bigrams
English Stop Words
Sublinear TF Scaling
5,000 Maximum Features
64 SVD Components
WordPiece Subword Tokenization

Used by:

intfloat/e5-small-v2

The combined question and options are encoded into a 384-dimensional sentence embedding.

The E5 model uses WordPiece subword tokenization with a vocabulary of approximately 30K subwords.

🧠 Model 1 - PyTorch OptionScorerNN

The first model is a custom neural network developed from scratch using PyTorch.

It operates on the 262 engineered features.

Architecture
262 Input Features
        │
        ▼
Linear(262 → 128)
        │
BatchNorm1d
        │
ReLU
        │
Dropout(0.30)
        │
        ▼
Linear(128 → 64)
        │
BatchNorm1d
        │
ReLU
        │
Dropout(0.30)
        │
        ▼
Linear(64 → 1)
        │
        ▼
Option Score
Training Configuration
Parameter	Value
Framework	PyTorch
Architecture	262 → 128 → 64 → 1
Optimizer	Adam
Learning Rate	5e-3
Batch Size	256
Epochs	30
Dropout	0.30
Loss	BCEWithLogitsLoss
Validation	5-Fold StratifiedKFold
Performance
Validation Accuracy = 0.9690
Validation F1-Macro = 0.9692
Validation MAP@3    = 0.98267
🤗 Model 2 - E5-small-v2 + MLP

The second model uses a pretrained SentenceTransformer:

intfloat/e5-small-v2

The model generates a semantic representation of the question and its candidate options.

The base Transformer remains frozen while a custom MLP classification head is trained.

E5 Encoder
Input Text
    │
    ▼
E5-small-v2
    │
    ▼
384-Dimensional Embedding

Model characteristics:

Transformer Layers = 12
Embedding Size     = 384
Parameters         ≈ 33M
MLP Classification Head
384
 │
 ▼
Linear(384 → 256)
 │
 ▼
Linear(256 → 64)
 │
 ▼
Linear(64 → 5)
 │
 ▼
A / B / C / D / E

The final layer produces scores for the five answer choices.

Training Configuration
Parameter	Value
Base Model	intfloat/e5-small-v2
Embedding Size	384
MLP Architecture	384 → 256 → 64 → 5
Optimizer	Adam
Learning Rate	1e-3
Batch Size	32
Epochs	15
Loss	CrossEntropyLoss
Validation	5-Fold StratifiedKFold
Precision	FP32
Device	CUDA GPU
Performance
Validation Accuracy = 0.9920
Validation F1-Macro = 0.9919
Validation MAP@3    = 0.99433

This was the best individual model.

🌳 Model 3 - LightGBM

The third model uses LightGBM Gradient Boosted Decision Trees.

It uses the 262 engineered features created from TF-IDF and SVD.

The model predicts binary candidate-option probabilities which are reshaped into five-option score vectors.

Configuration
Objective         = binary
Boosting          = gbdt
Max Depth         = 5
Num Leaves        = 31
Learning Rate     = 0.05
Feature Fraction  = 0.8
Boosting Rounds   = up to 150
Early Stopping    = 15
Performance
Validation Accuracy = 0.9390
Validation F1-Macro = 0.9396
Validation MAP@3    = 0.96258
📊 Model Comparison
Model	Model Type	Val Accuracy	Val F1-Macro	Val MAP@3
PyTorch OptionScorerNN	Custom PyTorch MLP	0.9690	0.9692	0.98267
E5-small-v2 + MLP	Pretrained Transformer	0.9920	0.9919	0.99433
LightGBM	Gradient Boosted Trees	0.9390	0.9396	0.96258
Powell Weighted Ensemble	3-Model Ensemble	—	—	0.99808
🔀 Ensemble Optimization

The predictions from the three models are combined using Powell weight optimization.

The optimization is performed using:

scipy.optimize.minimize

The optimization operates on out-of-fold probability predictions.

Optimized Weights

The final optimized weights were:

PyTorch NN          → 0.5010
E5-small-v2 + MLP   → 0.4892
LightGBM            → 0.0098

The optimized ensemble achieved:

OOF MAP@3 = 0.99808
🔎 Hybrid Text Matching

In addition to machine learning predictions, the final system uses a two-stage text matching pipeline.

This helps identify test questions that are highly similar to questions already available in the training dataset.

Stage 1 - Jaccard Similarity

Normalized test prompts are compared with historical training prompts using Jaccard similarity.

Threshold:

Jaccard Similarity >= 0.85
Stage 2 - SequenceMatcher

For highly similar prompts, the candidate answer options are compared using:

difflib.SequenceMatcher

Threshold:

Sequence Similarity >= 0.50

The system compares option text rather than relying only on option letters.

📊 Hybrid Matcher Results

The text matching pipeline successfully matched:

374 / 500

test questions to historical training records.

Therefore:

Historical Match Coverage = 74.8%

For matched questions:

Historical Answer → Rank 1

For unmatched questions:

3-Model Ensemble → Top-3 Ranking
🏗️ Complete Pipeline
                    ┌───────────────────┐
                    │   Train Dataset   │
                    │     2000 MCQs     │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Text Preprocessing│
                    │ & Normalization   │
                    └─────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
      ┌─────────────────┐          ┌─────────────────┐
      │ TF-IDF + SVD    │          │ E5-small-v2     │
      │ 262 Features    │          │ 384 Embeddings  │
      └────────┬────────┘          └────────┬────────┘
               │                            │
        ┌──────┴──────┐                     ▼
        │             │              ┌──────────────┐
        ▼             ▼              │  MLP Head    │
   PyTorch NN      LightGBM          └──────┬───────┘
        │             │                     │
        └──────┬──────┴─────────────────────┘
               │
               ▼
       ┌────────────────────┐
       │ Powell Optimization│
       │ Weighted Ensemble  │
       └──────────┬─────────┘
                  │
                  ▼
       ┌────────────────────┐
       │ Hybrid Text Matcher│
       │                    │
       │ Jaccard >= 0.85    │
       │ Sequence >= 0.50   │
       └──────────┬─────────┘
                  │
            ┌─────┴─────┐
            │           │
          Match      No Match
            │           │
            ▼           ▼
       Historical    Ensemble
         Answer      Prediction
            │           │
            └─────┬─────┘
                  │
                  ▼
           ┌──────────────┐
           │ Top-3 Ranking│
           └──────┬───────┘
                  │
                  ▼
           submission.csv
📈 Kaggle Score Progression

The solution was improved through multiple notebook versions.

Version	Improvement	Kaggle Score
v3	Baseline TF-IDF + MiniLM + DeBERTa v3 Small Fine-Tuning	0.38861
v17	262 Features + PyTorch NN + Cross-Encoder + LightGBM	0.74272
v23	EDA + Option Text Normalization	0.74522
v24	Option Length Bias + Cosine Matrix	0.74979
v27	SentenceTransformer + MLP + 5-Fold Checkpoint Averaging	0.75062
v37	Hybrid Text Matcher + Powell Ensemble	0.75602
🏆 Final Results
Kaggle
Final Kaggle Score = 0.75602

Target Cutoff      = 0.73000

Improvement        = +0.02602

The final submission successfully exceeded the competition target.

Validation Results
PyTorch NN MAP@3       = 0.98267

E5-small-v2 + MLP     = 0.99433

LightGBM MAP@3        = 0.96258

Powell Ensemble       = 0.99808
📡 Experiment Tracking

The project uses Weights & Biases (W&B) for experiment tracking.

The notebook tracks:

Training loss
Validation accuracy
Validation F1-Macro
Validation MAP@3
Training steps
Fold information
Boosting iterations
Binary logloss

The notebook also automatically detects the available device:

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

GPU acceleration was used for the Transformer-based model when available.

🛠️ Technologies Used
Programming
Python
Deep Learning
PyTorch
Sentence Transformers
Hugging Face Transformers
Machine Learning
Scikit-learn
LightGBM
NLP
TF-IDF
N-Gram Vectorization
Truncated SVD
Sentence Embeddings
Jaccard Similarity
SequenceMatcher
Optimization
SciPy
Powell Optimization
Data Processing
Pandas
NumPy
Visualization
Matplotlib
Seaborn
Experiment Tracking
Weights & Biases
Platform
Kaggle
📁 Project Structure

Recommended repository structure:

smart-mcq-solver/
│
├── README.md
│
├── notebook/
│   └── smart-mcq-solver.ipynb
│
├── report/
│   └── DL_GenAI_Project_Report.pdf
│
├── submission/
│   └── submission.csv
│
└── requirements.txt
📓 Notebook

The main implementation is available in:

https://github.com/23f3002830/dl-genai-project-26-t2/blob/main/dl-23f3002830-notebook-t22026%20(6).ipynb

The notebook contains:

Environment Setup
        ↓
Data Loading
        ↓
EDA
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
TF-IDF + SVD
        ↓
PyTorch Neural Network
        ↓
E5-small-v2 + MLP
        ↓
LightGBM
        ↓
5-Fold Cross Validation
        ↓
Model Evaluation
        ↓
Powell Ensemble
        ↓
Hybrid Text Matching
        ↓
Top-3 Prediction
        ↓
submission.csv
🚀 How to Run
Option 1: Run on Kaggle

The notebook was developed and executed in the Kaggle environment.

The competition files are available at:

/kaggle/input/competitions/smart-mcq-solver-challenge/

Required files:

train.csv
test.csv
sample_submission.csv
Steps
Open the Kaggle competition.
Create a new Kaggle Notebook.
Add the competition dataset.
Upload/import the notebook.
Run all cells.
The final submission file will be generated.
💻 Option 2: Run Locally

Clone the repository:

git clone https://github.com/23f3002830/smart-mcq-solver.git

Navigate to the project:

cd smart-mcq-solver

Install the dependencies:

pip install -r requirements.txt

Open the notebook:

notebook/smart-mcq-solver.ipynb
📦 Requirements

Create a requirements.txt file with:

numpy
pandas
matplotlib
seaborn
scikit-learn
torch
transformers
sentence-transformers
lightgbm
scipy
wandb
kagglehub
🔐 Weights & Biases Configuration

The notebook uses a Kaggle Secret for the W&B API key.

The secret is expected under one of the following names:

WB

or:

WANDB_API_KEY

For security:

Never commit your W&B API key or other private credentials to GitHub.

⚠️ Dataset

The competition dataset is not included in this repository.

The required competition files are:

train.csv
test.csv
sample_submission.csv

They should be obtained from the corresponding Kaggle competition.

🧪 Cross Validation

All three models use:

5-Fold StratifiedKFold

This allows the models to be evaluated across multiple train/validation splits while maintaining the distribution of the answer classes.

The resulting out-of-fold predictions are also used for ensemble optimization.

⚠️ Error Analysis

Several failure modes were observed during the experiments.

1. Distractor Prefix Confusion

Some candidate options differ only by small prefixes.

Example:

hypo-
hyper-

These subtle differences can be difficult to distinguish.

The contextual representations generated by the Transformer help provide additional semantic information.

2. Short Correct Answers

A correct answer can sometimes be a short one-word answer while incorrect distractors are significantly longer.

Because the dataset contains a strong longest-option bias, relying too heavily on option length can lead to incorrect predictions.

The ensemble helps reduce this dependency.

3. Option Choice Reordering

The same correct answer can appear under different option letters.

For example:

Training:
A → Correct Answer

Test:
C → Same Correct Answer

The hybrid matching pipeline therefore compares the actual option text rather than relying only on fixed answer letters.

💡 Key Learnings
1. Dataset Structure Can Be Valuable

The discovery that the longest option was correct approximately 40.55% of the time provided a useful structural signal.

2. Transformer Embeddings Perform Strongly

The pretrained E5-small-v2 model produced the best individual validation MAP@3:

0.99433

This demonstrates the value of contextual semantic embeddings for MCQ answer ranking.

3. Model Diversity Improves Ensembles

The three models capture different types of information:

PyTorch NN
    ↓
Engineered numerical features

E5-small-v2 + MLP
    ↓
Contextual semantic embeddings

LightGBM
    ↓
Tree-based non-linear patterns

Combining them resulted in:

Optimized OOF MAP@3 = 0.99808
4. Retrieval Complements Machine Learning

When a test question is highly similar to a historical training question, text matching can provide a strong signal.

The hybrid retrieval system matched:

374 / 500

test questions.

5. Text Matching Helps With Option Reordering

Matching the actual answer-option text rather than relying only on option letters helps handle candidate option reordering.

🔮 Future Work
1. Fine-Tune Larger Cross-Encoders

Future work could fine-tune larger models such as:

DeBERTa-v3-large
ModernBERT

using pairwise question-option scoring.

This could improve the model's ability to directly learn:

Question ↔ Candidate Answer

relationships.

2. Retrieval-Augmented Generation

A RAG pipeline could be integrated using a vector database containing:

Wikipedia
Domain-specific documents
Reference material

Potential pipeline:

Question
   ↓
Retriever
   ↓
Relevant Documents
   ↓
Context + Question + Options
   ↓
Answer Scoring Model
   ↓
Top-3 Answers
3. Automated Hyperparameter Optimization

Future experiments could use:

Optuna
Bayesian Optimization

to optimize:

Learning rate
Dropout
Batch size
Hidden dimensions
Network architecture
LightGBM parameters
Ensemble weights
📚 References
Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. NeurIPS 30.
Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, G., Chanan, G., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12.
Biewald, L. (2020). Experiment Tracking with Weights & Biases.
👨‍💻 Author
Prince Patel

IIT Madras
BS in Data Science

Student ID: 23F3002830
🎓 Project Information
Property	Details
Project	Smart MCQ Solver Challenge
Course	Deep Learning & Generative AI
Institute	IIT Madras
Competition	Kaggle
Task	Top-3 MCQ Answer Prediction
Metric	MAP@3
Final Kaggle Score	0.75602
Optimized OOF MAP@3	0.99808
⭐ Key Features
✅ Custom PyTorch Neural Network
✅ Pretrained E5-small-v2 Transformer
✅ MLP Classification Head
✅ LightGBM Gradient Boosting
✅ TF-IDF Feature Engineering
✅ Truncated SVD
✅ 262-Dimensional Question-Option Features
✅ 5-Fold Stratified Cross Validation
✅ Powell Ensemble Optimization
✅ Jaccard Similarity Retrieval
✅ SequenceMatcher Option Matching
✅ Hybrid ML + Retrieval Pipeline
✅ Weights & Biases Experiment Tracking
✅ CUDA GPU Support
✅ Top-3 Answer Ranking
✅ Kaggle Submission
✅ Final Kaggle Score: 0.75602
🏁 Conclusion

This project demonstrates a hybrid approach to multiple-choice question answering by combining NLP feature engineering, deep learning, pretrained Transformer representations, gradient boosting, ensemble optimization, and text retrieval.

The final system achieved:

Kaggle Score
    ↓
0.75602

Target Cutoff
    ↓
0.73000

Optimized OOF MAP@3
    ↓
0.99808

Historical Match Coverage
    ↓
74.8%

The results demonstrate that combining semantic understanding, engineered features, model diversity, and historical text matching can provide a strong solution for Top-3 MCQ answer prediction.
