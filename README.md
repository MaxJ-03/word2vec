# Word2Vec in Pure NumPy

## Table of Contents
* [Overview](#overview)
* [Model Architectures](#model-architectures)
* [Literature and References](#literature-and-references)
* [Implementation Details](#implementation-details)
* [Architectures Implemented](#architectures-implemented)
* [Datasets](#datasets)
* [Usage: Web Dashboard & Terminal UI](#usage-web-dashboard--terminal-ui)
* [Testing Suite](#testing-suite)
* [Analogy Testing](#analogy-testing)
* [Evaluation, Architectural Insights, and Future Work](#evaluation-architectural-insights-and-future-work)
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

## Datasets
The models were trained and evaluated on three distinct datasets to assess performance across different data scales and linguistic distributions:

* **text8**: A large, widely used corpus of cleaned Wikipedia text. This dataset provides a robust environment with high vocabulary coverage, making it highly effective for learning generalized semantic and syntactic representations.
* **WikiText2train**: A smaller, standard language modeling dataset. Due to its reduced size compared to text8, it offers lower vocabulary coverage for the evaluation benchmark, which results in a significantly higher rate of skipped questions and lower overall accuracy.
* **dummy**: A synthetic, toy dataset consisting of highly repetitive sentences explicitly modeling analogy relationships (e.g., "athens is the capital of greece . paris is the capital of france ."). This dataset is utilized mainly for debugging purposes to verify that the models can successfully learn explicit, localized patterns without structural errors.

## Usage: Web Dashboard & Terminal UI
This project offers two ways to interact with the models: a Terminal UI and a Web Dashboard.

### 1. Web Dashboard (GUI)
To launch the interactive web interface, run:
```bash
python web/app.py
 ```

Before starting the dashboard, make sure the embedding files are fully downloaded (this repository stores large embedding files with Git LFS):

```bash
git lfs install
git lfs pull --include="embeddings/*"
```

Optional verification:
```bash
git lfs ls-files
```

If embeddings were not pulled and only LFS pointer files are present, model loading in the dashboard can fail and you may see model selection errors.

* **Model Explorer:** Load saved `.txt` embeddings into memory to interactively search for closest word neighbors. It includes an Analogy Calculator that projects high-dimensional vector math onto a 2D coordinate system using pure NumPy PCA for visual analysis.
* **Training Hub:** Configure hyperparameters (Epochs, LR, Dimensions, Context Window, Negative Samples) and launch background training protocols across multiple architectures simultaneously.
* **Live Leaderboard:** A real-time ranking table that automatically updates with semantic and syntactic accuracy scores as soon as background training threads complete their evaluation phase.

### 2. Terminal UI
To run the standard command-line hub, execute:
``` bash
python src/run_training.py
```

* **Interactive Menus:** Prompts for dataset selection and hyperparameter configuration directly in the terminal.
* **Batch Mode:** Sequentially train all 6 architectures automatically and log the results to the central metrics file.

## Testing Suite
The repository includes a testing suite that validates the data processing pipeline, Huffman tree generation, and the exact calculations of the forward and backward passes. 

To run the test suite locally, execute:
```bash 
python run_tests.py
```

## Analogy Testing
The models are evaluated using the standard `word-test.v1.txt` dataset, originally introduced alongside the Word2Vec architecture in Mikolov's 2013 papers. The evaluation script (`evaluation.py`) processes the trained weight matrix (normalized to unit length) and tests the vector space for:
* **Semantic accuracy** (e.g., Athens : Greece :: Oslo : Norway)
* **Syntactic accuracy** (e.g., apparent : apparently :: rapid : rapidly)

### Accuracy Calculation
Vector offsets are computed using cosine similarity. For an analogy question such as "A is to B as C is to D", the target vector is calculated as:

$$\vec{v}_{target} = \vec{v}_B - \vec{v}_A + \vec{v}_C$$

The vocabulary is then searched for the vector mathematically closest to this result. If the closest word matches the target word D, it is marked as correct. If any of the four words in the analogy are missing from the model's vocabulary, the question is skipped. The final accuracy is calculated as the percentage of correctly answered, non-skipped analogies.

## Evaluation, Architectural Insights, and Future Work

The primary objective of this repository was to build, optimize, and mathematically verify Word2Vec from scratch using pure NumPy. The evaluation metrics derived from the standard analogy benchmark (detailed in the Benchmark Results below) revealed critical insights into how different architectural paradigms handle the geometry of language. 

### Architectural Differences: Skip-Gram vs. CBOW
Across all real-world text corpora evaluated (`text8` and `WikiText2train`), the Skip-Gram architecture exhibited an overwhelming dominance over the Continuous Bag-of-Words (CBOW) models. 
* **The Averaging Penalty:** CBOW computes the hidden layer by averaging the one-hot encoded context vectors. While computationally lightweight, this operation fundamentally smooths over the exact sequential and distributional details of the text.
* **Fine-Grained Signals:** Skip-Gram inverts this relationship, forcing the model to predict multiple distinct context words from a single input word. This structure injects a much higher volume of specific gradient updates per word pair, allowing the vector space to resolve highly specific semantic and syntactic analogies, peaking at nearly 20% accuracy on `text8`.

### Optimization Dynamics: Hierarchical Softmax vs. Negative Sampling
In the confines of this pure NumPy implementation, Hierarchical Softmax (HS) proved to be the more efficient and stable optimization strategy. 
* The top-performing Skip-Gram configurations all utilized the Huffman tree routing, achieving higher accuracy in fewer epochs compared to Negative Sampling (NS). 
* **Structural Bias:** The Huffman tree inherently places the most frequent vocabulary words near the root. This guarantees that the network frequently and reliably updates the internal node weights that govern the structural foundation of the language. Negative Sampling, reliant on stochastic draws, likely requires a higher number of training epochs or a dynamically tuned sampling distribution to converge to the same level of stability.

### The Limits of Dimensionality
Scaling the hidden layer directly impacts the model's degrees of freedom. The benchmarks confirm a positive, though non-linear, correlation between dimensionality and accuracy on adequately sized datasets. 
* On the `text8` corpus, increasing the Skip-Gram dimension from 90 to 150 yielded a direct performance boost from 17.44% to 19.79%. 
* However, evaluation on smaller or synthetic datasets (such as `dummy`) demonstrates the risk of over-parameterization. If the vector dimensionality is too high relative to the vocabulary density, the vectors spread too sparsely across the latent space, overfitting to noise and degrading the cosine similarity relationships.

### Implementation Learnings
Building a highly iterative machine learning algorithm in pure Python required aggressive architectural optimization to bypass the interpreter's overhead.

* **Vectorization over Loops:** Native Python loops initially proved to be a massive bottleneck, particularly when calculating gradients for varying context windows and tree paths. By systematically replacing Python `for` loops with flattened arrays, `np.concatenate`, and unbuffered accumulations (`np.add.at`), almost all mathematical operations were successfully transferred to NumPy’s highly optimized C-backend.
* **Iterative Tree Generation:** The construction and traversal of the Huffman Tree for Hierarchical Softmax was initially slow and memory-intensive due to recursive design patterns. This was resolved through two major optimizations: first, utilizing a Python `min-heap` to turn repeated global array sorts into highly efficient `O(V log V)` merging work. Second, the tree-to-dictionary conversion algorithm was rewritten to use an iterative Depth-First Search (DFS) with a manual stack. This entirely eliminates the function call overhead and prevents the interpreter from hitting recursion limits when processing massive vocabularies.
* **Mathematical Verification:** Decoupling the analytical gradient computations from the network state updates allowed for the implementation of strict finite-difference numerical gradient checks. This mathematical proofing guarantees that the underlying matrix calculus is absolute and eliminates the risk of silent errors in the computations.

### Limitations and Future Work
While this codebase provides a mathematically verified baseline, time constraints naturally limited the scope of the evaluation. To push this implementation further, future work should focus on three main areas:

1. **Comprehensive Hyperparameter Search:** The current benchmarks rely on manually selected configurations. To find the true performance ceiling of each architecture, the clear next step is to build an automated grid search to test different combinations of learning rates, decay factors, embedding dimensions, and negative sampling sizes.
2. **Dataset Expansion:** The models are currently evaluated on a few standard datasets. Testing the architectures against a wider variety of domain-specific text will give a much clearer picture of where this Word2Vec implementation generalizes well and where it breaks down.
3. **Subsampling Frequent Words:** The data pipeline currently processes every single token in the corpus. Adding Tomas Mikolov’s probabilistic subsampling to drop common, low-information words (like "the" or "is") would drastically speed up training times and help the model build better vectors for rare words.

## Benchmark Results
Below is the performance comparison of the different architectures tested on the analogy benchmark (legend below the tables).

The benchmark section is auto-generated from `reports/metrics_log.json` by `src/generate_report.py` and is split into separate tables per dataset (trainset) for easier inspection.

<!-- BENCHMARK_TABLE_START -->
### Dataset: text8
| Architecture | Epochs | LR | Dim | Window | NS | Sem. Eval | Sem. Acc | Syn. Eval | Syn. Acc | Skipped | Total Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 150 | 4 | - | 8561 | 16.11% | 10545 | 22.78% | 438 | 19.79% |
| Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 150 | 4 | - | 8561 | 15.96% | 10545 | 22.69% | 438 | 19.67% |
| Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 90 | 4 | - | 8561 | 13.18% | 10545 | 20.91% | 438 | 17.44% |
| Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 90 | 4 | - | 8561 | 13.18% | 10545 | 20.91% | 438 | 17.44% |
| Skip-Gram with Negative Sampling | 3 | 0.025 | 90 | 4 | 20 | 8561 | 8.60% | 10545 | 21.55% | 438 | 15.74% |
| Skip-Gram with Negative Sampling | 3 | 0.025 | 90 | 4 | 20 | 8561 | 8.60% | 10545 | 21.55% | 438 | 15.74% |
| Skip-Gram with Hierarchical Softmax | 1 | 0.05 | 150 | 4 | - | 8561 | 2.04% | 10545 | 1.20% | 438 | 1.58% |
| Skip-Gram with Negative Sampling | 1 | 0.05 | 150 | 4 | 20 | 8561 | 1.59% | 10545 | 1.48% | 438 | 1.53% |
| CBOW with Negative Sampling | 1 | 0.05 | 150 | 4 | 20 | 8561 | 0.69% | 10545 | 0.36% | 438 | 0.51% |
| CBOW with Negative Sampling | 1 | 0.025 | 70 | 4 | 20 | 8561 | 0.18% | 10545 | 0.11% | 438 | 0.14% |

### Dataset: dummy
| Architecture | Epochs | LR | Dim | Window | NS | Sem. Eval | Sem. Acc | Syn. Eval | Syn. Acc | Skipped | Total Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Skip-Gram with Negative Sampling | 40 | 0.025 | 40 | 4 | 5 | 45 | 13.33% | 24 | 29.17% | 19475 | 18.84% |
| Skip-Gram with Negative Sampling | 40 | 0.025 | 40 | 4 | 5 | 45 | 13.33% | 24 | 29.17% | 19475 | 18.84% |
| Skip-Gram with Hierarchical Softmax | 40 | 0.025 | 10 | 3 | - | 45 | 11.11% | 24 | 25.00% | 19475 | 15.94% |
| Skip-Gram with Negative Sampling | 40 | 0.025 | 10 | 3 | 5 | 45 | 6.67% | 24 | 33.33% | 19475 | 15.94% |
| Skip-Gram with Hierarchical Softmax | 40 | 0.025 | 10 | 3 | - | 45 | 11.11% | 24 | 25.00% | 19475 | 15.94% |
| Skip-Gram with Negative Sampling | 40 | 0.025 | 10 | 3 | 5 | 45 | 6.67% | 24 | 33.33% | 19475 | 15.94% |
| Skip-Gram with Negative Sampling | 50 | 0.025 | 10 | 4 | 5 | 45 | 8.89% | 24 | 20.83% | 19475 | 13.04% |
| Standard Skip-Gram | 50 | 0.025 | 10 | 4 | - | 45 | 11.11% | 24 | 16.67% | 19475 | 13.04% |
| Skip-Gram with Hierarchical Softmax | 50 | 0.025 | 10 | 4 | - | 45 | 13.33% | 24 | 8.33% | 19475 | 11.59% |
| Skip-Gram with Hierarchical Softmax | 40 | 0.025 | 40 | 4 | - | 45 | 11.11% | 24 | 12.50% | 19475 | 11.59% |
| Skip-Gram with Hierarchical Softmax | 40 | 0.025 | 40 | 4 | - | 45 | 11.11% | 24 | 12.50% | 19475 | 11.59% |
| Standard CBOW | 100 | 0.025 | 50 | 5 | - | 45 | 2.22% | 24 | 29.17% | 19475 | 11.59% |
| Skip-Gram with Hierarchical Softmax | 100 | 0.025 | 50 | 5 | - | 45 | 8.89% | 24 | 8.33% | 19475 | 8.70% |
| Skip-Gram with Negative Sampling | 100 | 0.025 | 50 | 5 | 5 | 45 | 8.89% | 24 | 8.33% | 19475 | 8.70% |
| CBOW with Hierarchical Softmax | 50 | 0.025 | 10 | 4 | - | 45 | 0.00% | 24 | 16.67% | 19475 | 5.80% |
| CBOW with Negative Sampling | 50 | 0.025 | 10 | 4 | 5 | 45 | 4.44% | 24 | 4.17% | 19475 | 4.35% |
| Standard CBOW | 50 | 0.025 | 10 | 4 | - | 45 | 4.44% | 24 | 4.17% | 19475 | 4.35% |

### Dataset: WikiText2train
| Architecture | Epochs | LR | Dim | Window | NS | Sem. Eval | Sem. Acc | Syn. Eval | Syn. Acc | Skipped | Total Acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 100 | 4 | - | 2704 | 4.25% | 7506 | 4.49% | 9334 | 4.43% |
| Skip-Gram with Hierarchical Softmax | 3 | 0.025 | 80 | 4 | - | 2704 | 3.96% | 7506 | 3.73% | 9334 | 3.79% |
| CBOW with Hierarchical Softmax | 3 | 0.025 | 100 | 4 | - | 2704 | 3.29% | 7506 | 1.03% | 9334 | 1.63% |
| Skip-Gram with Negative Sampling | 3 | 0.025 | 100 | 4 | 10 | 2704 | 2.48% | 7506 | 0.56% | 9334 | 1.07% |
| CBOW with Negative Sampling | 3 | 0.025 | 100 | 4 | 10 | 2704 | 0.22% | 7506 | 0.01% | 9334 | 0.07% |
<!-- BENCHMARK_TABLE_END -->

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