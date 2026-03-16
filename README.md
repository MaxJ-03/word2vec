# Word2Vec in Pure NumPy

## Overview
This repository contains a custom implementation of the Word2Vec algorithm. The primary goal of this project was to build the core training loop and optimization procedures entirely in pure Python and NumPy, without relying on high-level machine learning frameworks like PyTorch or TensorFlow. 

The full optimization procedure—including the forward pass, loss calculation, gradient derivations, and parameter updates—was implemented for both standard Word2Vec variants (Continuous Bag-of-Words and Skip-Gram). A modular and object-oriented architecture was chosen to allow for easy testing and comparison between different architectural setups.

## Literature and References
The development and mathematical derivations in this repository rely on two main sources:

* **Conceptual Understanding:** The core concepts behind the model and the resulting vector space properties are based on Tomas Mikolov's original Word2Vec publications:
  * Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781.
  * Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. Advances in Neural Information Processing Systems, 26.
* **Implementation Details:** The mathematical implementation closely follows Xin Rong's detailed breakdown. The equations provided in this paper were used directly to write the matrix updates and backpropagation steps across all model variants:
  * Rong, X. (2014). *word2vec Parameter Learning Explained*. arXiv preprint arXiv:1411.2738.

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
* `skip_gram_hier_softmax.py`: Skip-Gram a binary Huffman Tree for Hierarchical Softmax.
* `skip_gram_neg_sample.py`: Skip-Gram with Negative Sampling.

## Usage and Terminal UI
A centralized Terminal User Interface (UI) is provided to easily manage training and evaluation. 

To start the hub, execute:
    python src/main.py

### Functionalities
1. **Train New Models:** * Allows selection of a specific text dataset from the `trainsets/` directory.
   * Prompts for hyperparameter configuration (epochs, learning rate, window size, embedding dimension, negative sampling size).
   * Supports a Batch Mode to sequentially train all 6 architectures automatically.
   * Automatically saves trained weights to the `embeddings/` folder.
2. **Evaluate Saved Models:** * Loads saved `.txt` embedding matrices into memory.
   * Allows interactive closest-neighbor word queries via cosine similarity.
   * Executes the standardized analogy benchmark.
3. **Generate Markdown Report:** * Compiles the analogy accuracy results from all trained models into a clean Markdown table.

## Testing Suite
The repository includes a testing suite that validates the data processing pipeline, Huffman tree generation, and the exact calculations of the forward and backward passes. 

To run the test suite locally, execute:
    python run_tests.py

## Evaluation and Analogy Testing
The models are evaluated using the standard `word-test.v1.txt` dataset, originally introduced alongside the Word2Vec architecture in Mikolov's 2013 papers. The evaluation script (`evaluation.py`) processes the trained weight matrix (normalized to unit length) and tests the vector space for:
* **Semantic accuracy** (e.g., Athens : Greece :: Oslo : Norway)
* **Syntactic accuracy** (e.g., apparent : apparently :: rapid : rapidly)

## Benchmark Results
Below is the performance comparison of the different architectures tested on the analogy benchmark.
<!-- BENCHMARK_TABLE_START -->
| Model Architecture | Semantic Accuracy | Syntactic Accuracy | Total Accuracy |
|---|---|---|---|
| `dummy_Skip-Gram_with_Negative_Sampling_dim10_w2` | 33.33% | 0.00% | 33.33% |
| `dummy_CBOW_with_Hierarchical_Softmax_dim10_w2` | 8.33% | 0.00% | 8.33% |
| `dummy_CBOW_with_Negative_Sampling_dim10_w2` | 8.33% | 0.00% | 8.33% |
| `dummy_Skip-Gram_with_Hierarchical_Softmax_dim10_w2` | 8.33% | 0.00% | 8.33% |
| `dummy_Standard_Skip-Gram_dim10_w2` | 8.33% | 0.00% | 8.33% |
| `WikiText2train_Skip-Gram_with_Negative_Sampling_dim100_w10` | 3.25% | 2.37% | 2.61% |
| `text8_CBOW_with_Negative_Sampling_ep1_lr0.025_dim70_w4_ns20` | 0.18% | 0.11% | 0.14% |
| `WikiText2train_CBOW_with_Negative_Sampling_dim100_w10` | 0.26% | 0.05% | 0.11% |
| `WikiText2train_CBOW_with_Negative_Sampling_dim300_w10` | 0.00% | 0.01% | 0.01% |
| `dummy_CBOW_with_Hierarchical_Softmax_dim10_w10` | 0.00% | 0.00% | 0.00% |
| `dummy_CBOW_with_Negative_Sampling_dim10_w10` | 0.00% | 0.00% | 0.00% |
| `dummy_Standard_CBOW_dim10_w10` | 0.00% | 0.00% | 0.00% |
| `dummy_Standard_CBOW_dim10_w2` | 0.00% | 0.00% | 0.00% |

<!-- BENCHMARK_TABLE_END -->