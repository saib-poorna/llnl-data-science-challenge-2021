# Detection and Classification of Objects in Telescope Data Using Machine Learning

**Program:** Lawrence Livermore National Laboratory (LLNL) Data Science Challenge 2021 — Team 3
**Team:** Alexander Nguyen, Nathan Tran, Sai Poornasree Balamurugan, Severin Field, Rutuja Gurav
**Mentor:** Kerianne Pruett (UC Riverside)

## Overview
A national-lab-sponsored data science challenge applying machine learning to two telescope-imaging problems: classifying stars vs. galaxies, and detecting asteroids in difference images.

## Star vs. Galaxy Classification
- Performed EDA and PCA on vectorized image data (>70% of variance explained by the first principal component, indicating low-rank structure).
- Built a data pipeline (reshaping, normalization, 75/15/10 train/validation/test split) and trained multiple model families for comparison:
  - A simple CNN, evaluated per imaging band (g, r, i, z) and on all bands combined
  - Classical ML baselines: Naive Bayes, Gaussian Process, Decision Trees, Gradient Boosting
  - LeNet5 and AlexNet architectures
- Diagnosed *why* certain models underperformed — e.g. Naive Bayes's feature-independence assumption doesn't hold for image data, which explained its poor results relative to the CNN.
- Compared training time vs. performance tradeoffs (AlexNet: 2h38m to train vs. 6–49 min for simpler models) to evaluate whether the added complexity was worth it.
- Found that simple neural networks outperformed classical ML models, and that galaxies were harder to classify than stars.

## Asteroid Detection
- Framed asteroid detection as binary image classification from telescope "difference images."
- Built a custom data-generation pipeline to create labeled positive (asteroid) and negative (background) samples from raw bounding-box data.
- Iterated through several rounds of data-cleaning issues (dead pixels, inconsistent borders, class imbalance) that were causing near-random performance, diagnosing and fixing each.
- Final LeNet5 model reached 96% accuracy on a cleaned, balanced dataset (precision 0.97 / 0.948, recall 0.94 / 0.97 across the two classes).

## Tools
Python, TensorFlow, scikit-learn (Naive Bayes, Gaussian Process, Decision Trees, Gradient Boosting), CNN architectures (simple CNN, LeNet5, AlexNet)

## Skills demonstrated
Deep learning (CNN architecture design and training), classical ML model comparison, model evaluation (ROC, precision-recall, F1, confusion matrices), data pipeline design, dataset generation and cleaning, error/failure-mode analysis, cross-functional team collaboration

## Files
- `LLNL_DSC_2021_Team_3.pdf` — full presentation slides covering methodology, results, and findings for both challenge problems
