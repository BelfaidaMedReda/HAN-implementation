# 🧠 Hierarchical Attention Network (HAN) Using GRU  
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)  
[![Google Colab](https://img.shields.io/badge/Open%20In-Colab-yellow.svg)](https://colab.research.google.com/drive/159eQ1XZjG74ITsGEqG6fN-Yzb5nFN3mc)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains a **PyTorch implementation** of a **Hierarchical Attention Network (HAN)** for document classification — built and trained in **Google Colab**.  
The model follows the approach from *Yang et al., 2016*, combining **bi-directional GRUs** with **attention mechanisms** at both the word and sentence levels to produce interpretable document representations.

---

## 🧩 Project Overview

The HAN architecture mimics the hierarchical structure of language:  
- **Word Encoder** → encodes sequences of words into sentence representations.  
- **Sentence Encoder** → encodes sentences into document representations.  
- **Attention Layers** → compute context-aware weights at both levels for interpretability.  

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:800/1*0UMnxoQqDkzH7ij_iVj8_g.png" width="600" alt="HAN Architecture Diagram">
</p>

---

## 🧠 Model Components

### 🔹 1. Self-Attention Layer (`AttentionWithContext`)
Implements the attention mechanism as described by Yang et al. (2016):  
- Computes weighted averages of hidden states using a trainable context vector.  
- Supports extraction of **attention coefficients** for interpretability.  

### 🔹 2. Sentence Encoder (`AttentionBiGRU`)
- Embedding layer (trainable word vectors)  
- Bidirectional GRU to capture context in both directions  
- Word-level attention layer  

### 🔹 3. Document Encoder (`HAN`)
- Time-distributed sentence encoder  
- Bidirectional GRU for sentence sequence modeling  
- Sentence-level attention  
- Linear + sigmoid output layer for classification  

---

## ⚙️ Training Configuration

| Parameter | Description | Value |
|------------|-------------|-------|
| Optimizer | Adam | `lr=0.001` |
| Loss | Binary Cross-Entropy | BCELoss |
| Batch Size | 64 | |
| Epochs | 15 | |
| Dropout | 0.5 | |
| Hidden Units | 50 | |
| Patience (Early Stopping) | 2 | |
| Device | GPU (if available) | |

### ✅ Results  
Training achieved a **validation accuracy of ≈85%**, with strong interpretability through attention visualization.

---

## 📂 Project Structure

```
├── data/
│ ├── docs_train.npy
│ ├── labels_train.npy
│ ├── docs_test.npy
│ ├── labels_test.npy
│ └── word_to_index.json
│
├── Lab.ipynb # Main notebook (implementation & training)
├── mini-report.pdf
└── README.md # Project documentation
```

## 🚀 How to Run

### ▶️ Option 1: Open in Google Colab  
Click below to open the notebook directly:  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/159eQ1XZjG74ITsGEqG6fN-Yzb5nFN3mc)

### ▶️ Option 2: Run Locally  

```bash
git clone https://github.com/<your-username>/hierarchical-attention-network.git
cd hierarchical-attention-network
pip install torch numpy tqdm
```
Then launch the notebook or run:

```python
!unzip data.zip
train()
```

## 🔍 Attention Visualization

The model provides both word-level and sentence-level attention weights, enabling explainable NLP.
Example of sentence-level attention output:

```pgsql
27.04 ; ) First of all , Mulholland Drive is downright brilliant .
24.69 A masterpiece .
18.43 This is the kind of movie that refuse to leave your head .
```

## 🧪 Key Takeaways

- Hands-on implementation of Hierarchical Attention Networks with PyTorch.

- Design and use of custom attention layers.

- Exploration of explainable deep learning for NLP.

- Training optimization using early stopping and gradient clipping.

## 📖 Reference

> **Yang, Zichao, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy.**  
> *"Hierarchical Attention Networks for Document Classification."*  
> Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2016).  
> [[Read Paper]](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)


## 🧾 Additional Discussion: *mini-report.pdf*

The file **`mini-report.pdf`** included in this repository presents a concise theoretical discussion about:
- The **evolution of attention mechanisms** and their improvements beyond basic self-attention.  
- The **motivations for replacing recurrent operations** with self-attention, as introduced in *“Attention Is All You Need”* by Vaswani et al. (2017).  
- A detailed analysis of the **Hierarchical Attention Network (HAN)** architecture, highlighting its **strengths**, **limitations**, and **contextual dependencies** as discussed in later works such as *“Bidirectional Context-Aware Hierarchical Attention Network for Document Understanding” (Remy et al., 2019)*.

This report complements the implementation in the notebook by connecting **practical experimentation** with **academic insights** from seminal papers in NLP attention modeling.