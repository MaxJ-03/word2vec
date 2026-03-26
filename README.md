# Word2Vec in Pure NumPy

## Table of Contents
* [Overview](#overview)
* [Model Architectures](#model-architectures)
* [Literature and References](#literature-and-references)
* [Implementation Details](#implementation-details)
* [Architectures Implemented](#architectures-implemented)
* [Usage: Web Dashboard & Terminal UI](#usage-web-dashboard--terminal-ui)
* [Testing Suite](#testing-suite)
* [Evaluation and Analogy Testing](#evaluation-and-analogy-testing)
* [Benchmark Results](#benchmark-results)

## Overview
This repository contains a custom implementation of the Word2Vec algorithm. The primary goal of this project was to build the core training loop and optimization procedures entirely in pure Python and NumPy, without relying on high-level machine learning frameworks like PyTorch or TensorFlow. 

The full optimization procedure, including the forward pass, loss calculation, gradient derivations, and parameter updates, was implemented for both standard Word2Vec variants (Continuous Bag-of-Words and Skip-Gram). A modular and object-oriented architecture was chosen to allow for easy testing and comparison between different architectural setups.

## Model Architectures
The standard Word2Vec models learn distributed representations of words by predicting context within a specified window. Two primary architectures are implemented in this repository:

### Continuous Bag-of-Words (CBOW)
The CBOW architecture is designed to predict a target word based on its surrounding context words. The context word vectors are averaged (or summed) in the hidden layer before being used for prediction. This approach is computationally efficient and smooths over distributional information, making it effective for smaller datasets.

### Skip-Gram
The Skip-Gram architecture reverses the CBOW approach by using a single target word to predict the surrounding context words. By forcing the model to predict multiple context words from a single input, finer-grained vectors are created, which typically perform better for infrequent words and larger datasets.

## Literature and References
The development and mathematical derivations in this repository rely on two main sources:

* **Conceptual Understanding:** The core concepts behind the model and the resulting vector space properties are based on Tomas Mikolov's original Word2Vec publications:
  * Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781.
  * Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. Advances in Neural Information Processing Systems, 26.
* **Mathematical Implementation:** The explicit formulas for the loss functions, gradient derivations, backpropagation steps, and matrix updates used in this pure NumPy codebase are directly referenced (with numbers in the comments of the models) and derived from Xin Rong's detailed breakdown:
  * Rong, X. (2014). *word2vec Parameter Learning Explained*. arXiv preprint arXiv:1411.2738.

## Implementation Details
To achieve a pure NumPy implementation, several foundational optimization and data management techniques were built from scratch:

* **Learning Rate Decay:** An exponential learning rate decay mechanism is implemented to ensure the model converges smoothly. While the original Word2Vec publication utilized a linear decay strategy based on the volume of processed words, this implementation applies an exponential decay based on the number of completed training epochs. The learning rate starts at a configured hyperparameter and gradually decreases, allowing for large adjustments early in training and fine-tuned, microscopic steps as the model approaches the local minimum.
* **Saving Weights:** Upon completion of the training loop, the final word embeddings are extracted from the hidden layer weights, which serve as the primary embedding matrix. These vectors are normalized to unit length to optimize downstream cosine similarity calculations and are saved to disk as a `.txt` file. The first line denotes the vocabulary size and vector dimensions, followed by one line per word containing the token and its respective vector components.

## Architectures Implemented
The repository includes the major variations of the Word2Vec model, supporting both standard softmax and the optimized training techniques:

### Continuous Bag-of-Words (CBOW)
Located in `src/cbow/`:
* `cbow.py`: Standard CBOW using a full softmax output layer.
* `cbow_hier_softmax.py`: CBOW with a binary Huffman Tree for Hierarchical Softmax.
* `cbow_neg_sample.py`: CBOW with Negative Sampling.

### Skip-Gram
Located in `src/skip_gram/`:
* `skip_gram.py`: Standard Skip-Gram model.
* `skip_gram_hier_softmax.py`: Skip-Gram with a binary Huffman Tree for Hierarchical Softmax.
* `skip_gram_neg_sample.py`: Skip-Gram with Negative Sampling.

## Usage: Web Dashboard & Terminal UI
This project offers two ways to interact with the models: a centralized Terminal UI and a full-featured, real-time Web Dashboard powered by Flask.

### 1. Web Dashboard (GUI)
To launch the interactive web interface, run:
`python web/app.py`

* **Model Explorer:** Load saved `.txt` embeddings into memory to interactively search for closest word neighbors. It includes an Analogy Calculator that projects high-dimensional vector math onto a 2D coordinate system using pure NumPy PCA for visual analysis.
* **Training Hub:** Configure hyperparameters (Epochs, LR, Dimensions, Context Window, Negative Samples) and launch background training protocols across multiple architectures simultaneously.
* **Live Leaderboard:** A real-time ranking table that automatically updates with semantic and syntactic accuracy scores as soon as background training threads complete their evaluation phase.

### 2. Terminal UI
To run the standard command-line hub, execute:
`python src/run_training.py`

* **Interactive Menus:** Prompts for dataset selection and hyperparameter configuration directly in the terminal.
* **Batch Mode:** Sequentially train all 6 architectures automatically and log the results to the central metrics file.

## Testing Suite
The repository includes a testing suite that validates the data processing pipeline, Huffman tree generation, and the exact calculations of the forward and backward passes. 

To run the test suite locally, execute:
`python run_tests.py`

## Evaluation and Analogy Testing
The models are evaluated using the standard `word-test.v1.txt` dataset, originally introduced alongside the Word2Vec architecture in Mikolov's 2013 papers. The evaluation script (`evaluation.py`) processes the trained weight matrix (normalized to unit length) and tests the vector space for:
* **Semantic accuracy** (e.g., Athens : Greece :: Oslo : Norway)
* **Syntactic accuracy** (e.g., apparent : apparently :: rapid : rapidly)

### Accuracy Calculation
Vector offsets are computed using cosine similarity. For an analogy question such as "A is to B as C is to D", the target vector is calculated as:

$$\vec{v}_{target} = \vec{v}_B - \vec{v}_A + \vec{v}_C$$

The vocabulary is then searched for the vector mathematically closest to this result. If the closest word matches the target word D, it is marked as correct. If any of the four words in the analogy are missing from the model's vocabulary, the question is skipped. The final accuracy is calculated as the percentage of correctly answered, non-skipped analogies.

## Benchmark Results
Below is the performance comparison of the different architectures tested on the analogy benchmark (legend below the table).

<!-- BENCHMARK_TABLE_START -->
| Dataset | Architecture | Epochs | LR | Dim | Window | NS | Sem. Eval | Sem. Acc | Syn. Eval | Syn. Acc | Skipped | Total Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| text8 | Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 150 | 4 | - | 8561 | 15.96% | 10545 | 22.69% | 438 | 19.67% |
| text8 | Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 150 | 4 | - | 8561 | 15.96% | 10545 | 22.69% | 438 | 19.67% |
| dummy | Skip-Gram with Negative Sampling | 40 | 0.025 | 40 | 4 | 5 | 45 | 13.33% | 24 | 29.17% | 19475 | 18.84% |
| dummy | Skip-Gram with Negative Sampling | 40 | 0.025 | 40 | 4 | 5 | 45 | 13.33% | 24 | 29.17% | 19475 | 18.84% |
| dummy | Skip-Gram with Hierarchical Softmax | 40 | 0.025 | 10 | 3 | - | 45 | 11.11% | 24 | 25.00% | 19475 | 15.94% |
| dummy | Skip-Gram with Negative Sampling | 40 | 0.025 | 10 | 3 | 5 | 45 | 6.67% | 24 | 33.33% | 19475 | 15.94% |
| dummy | Skip-Gram with Hierarchical Softmax | 40 | 0.025 | 10 | 3 | - | 45 | 11.11% | 24 | 25.00% | 19475 | 15.94% |
| dummy | Skip-Gram with Negative Sampling | 40 | 0.025 | 10 | 3 | 5 | 45 | 6.67% | 24 | 33.33% | 19475 | 15.94% |
| text8 | Skip-Gram with Negative Sampling | 3 | 0.025 | 90 | 4 | 20 | 8561 | 8.60% | 10545 | 21.55% | 438 | 15.74% |
| text8 | Skip-Gram with Negative Sampling | 3 | 0.025 | 90 | 4 | 20 | 8561 | 8.60% | 10545 | 21.55% | 438 | 15.74% |
| dummy | Skip-Gram with Negative Sampling | 50 | 0.025 | 10 | 4 | 5 | 45 | 8.89% | 24 | 20.83% | 19475 | 13.04% |
| dummy | Standard Skip-Gram | 50 | 0.025 | 10 | 4 | - | 45 | 11.11% | 24 | 16.67% | 19475 | 13.04% |
| dummy | Skip-Gram with Hierarchical Softmax | 50 | 0.025 | 10 | 4 | - | 45 | 13.33% | 24 | 8.33% | 19475 | 11.59% |
| dummy | Skip-Gram with Hierarchical Softmax | 40 | 0.025 | 40 | 4 | - | 45 | 11.11% | 24 | 12.50% | 19475 | 11.59% |
| dummy | Skip-Gram with Hierarchical Softmax | 40 | 0.025 | 40 | 4 | - | 45 | 11.11% | 24 | 12.50% | 19475 | 11.59% |
| dummy | Skip-Gram with Hierarchical Softmax | 100 | 0.025 | 50 | 5 | - | 45 | 8.89% | 24 | 8.33% | 19475 | 8.70% |
| dummy | Skip-Gram with Negative Sampling | 100 | 0.025 | 50 | 5 | 5 | 45 | 8.89% | 24 | 8.33% | 19475 | 8.70% |
| dummy | CBOW with Hierarchical Softmax | 50 | 0.025 | 10 | 4 | - | 45 | 0.00% | 24 | 16.67% | 19475 | 5.80% |
| WikiText2train | Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 100 | 4 | - | 2704 | 4.25% | 7506 | 4.49% | 9334 | 4.43% |
| dummy | CBOW with Negative Sampling | 50 | 0.025 | 10 | 4 | 5 | 45 | 4.44% | 24 | 4.17% | 19475 | 4.35% |
| dummy | Standard CBOW | 50 | 0.025 | 10 | 4 | - | 45 | 4.44% | 24 | 4.17% | 19475 | 4.35% |
| WikiText2train | Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 80 | 4 | - | 2704 | 3.96% | 7506 | 3.73% | 9334 | 3.79% |
| text8 | Skip-Gram with Negative Sampling | 3 | 0.025 | 90 | 4 | 20 | 8561 | 2.35% | 10545 | 2.79% | 438 | 2.59% |
| OutlineOfHistory | Skip-Gram with Hierarchical Softmax | 10 | 0.05 | 100 | 4 | - | 343 | 12.54% | 5795 | 1.17% | 13406 | 1.81% |
| OutlineOfHistory | Skip-Gram with Hierarchical Softmax | 10 | 0.025 | 80 | 4 | - | 343 | 8.75% | 5795 | 1.26% | 13406 | 1.68% |
| WikiText2train | CBOW with Hierarchical Softmax | 3 | 0.025 | 100 | 4 | - | 2704 | 3.29% | 7506 | 1.03% | 9334 | 1.63% |
| text8 | Skip-Gram with Hierarchical Softmax | 1 | 0.05 | 150 | 4 | - | 8561 | 2.04% | 10545 | 1.20% | 438 | 1.58% |
| text8 | Skip-Gram with Negative Sampling | 1 | 0.05 | 150 | 4 | 20 | 8561 | 1.59% | 10545 | 1.48% | 438 | 1.53% |
| WikiText2train | Skip-Gram with Negative Sampling | 3 | 0.025 | 100 | 4 | 10 | 2704 | 2.48% | 7506 | 0.56% | 9334 | 1.07% |
| OutlineOfHistory | Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 100 | 4 | - | 343 | 3.50% | 5795 | 0.72% | 13406 | 0.88% |
| OutlineOfHistory | CBOW with Hierarchical Softmax | 10 | 0.05 | 100 | 4 | - | 343 | 3.21% | 5795 | 0.66% | 13406 | 0.80% |
| OutlineOfHistory | Skip-Gram with Negative Sampling | 10 | 0.05 | 100 | 4 | 15 | 343 | 5.83% | 5795 | 0.45% | 13406 | 0.75% |
| text8 | CBOW with Negative Sampling | 1 | 0.05 | 150 | 4 | 20 | 8561 | 0.69% | 10545 | 0.36% | 438 | 0.51% |
| OutlineOfHistory | Skip-Gram with Negative Sampling | 10 | 0.025 | 80 | 4 | 15 | 343 | 5.25% | 5795 | 0.21% | 13406 | 0.49% |
| OutlineOfHistory | CBOW with Hierarchical Softmax | 3 | 0.025 | 100 | 4 | - | 343 | 2.92% | 5795 | 0.29% | 13406 | 0.44% |
| OutlineOfHistory | CBOW with Hierarchical Softmax | 10 | 0.025 | 80 | 4 | - | 343 | 1.75% | 5795 | 0.33% | 13406 | 0.41% |
| OutlineOfHistory | Skip-Gram with Negative Sampling | 3 | 0.025 | 100 | 4 | 20 | 343 | 1.75% | 5795 | 0.05% | 13406 | 0.15% |
| text8 | CBOW with Negative Sampling | 1 | 0.025 | 70 | 4 | 20 | 8561 | 0.18% | 10545 | 0.11% | 438 | 0.14% |
| WikiText2train | CBOW with Negative Sampling | 3 | 0.025 | 100 | 4 | 10 | 2704 | 0.22% | 7506 | 0.01% | 9334 | 0.07% |
| OutlineOfHistory | CBOW with Negative Sampling | 3 | 0.025 | 100 | 4 | 20 | 343 | 0.00% | 5795 | 0.02% | 13406 | 0.02% |
| OutlineOfHistory | CBOW with Negative Sampling | 10 | 0.05 | 100 | 4 | 15 | 343 | 0.00% | 5795 | 0.02% | 13406 | 0.02% |
| OutlineOfHistory | CBOW with Negative Sampling | 10 | 0.025 | 80 | 4 | 15 | 343 | 0.00% | 5795 | 0.00% | 13406 | 0.00% |

<!-- BENCHMARK_TABLE_END -->

* **Dataset:** The text corpus used for training (e.g., `text8`, `dummy`).
* **Architecture:** The specific Word2Vec optimization strategy employed.
* **Epochs:** The number of complete passes through the training dataset.
* **LR (Learning Rate):** The initial learning rate provided before exponential decay.
* **Dim:** The dimensionality of the word embedding vectors (hidden layer size).
* **Window:** The maximum distance of context words before and after the target word.
* **NS:** The number of negative samples drawn per positive context pair (applicable only to Negative Sampling architectures).
* **Sem. Eval:** The number of semantic analogies (e.g., capitals, family) successfully evaluated.
* **Sem. Acc:** The accuracy percentage for evaluated semantic analogies.
* **Syn. Eval:** The number of syntactic analogies (e.g., plurals, comparatives) successfully evaluated.
* **Syn. Acc:** The accuracy percentage for evaluated syntactic analogies.
* **Skipped:** The total number of test questions discarded because one or more words were missing from the model's vocabulary.
* **Total Acc:** The overall accuracy across all evaluated semantic and syntactic questions.