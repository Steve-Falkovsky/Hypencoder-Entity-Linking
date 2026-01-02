# Hypencoder for Entity Linking

This repository explores the application of **Hypernetworks** to the task of **Entity Linking (EL)**. 
By moving beyond the limitations of traditional dense vector retrieval, this project implements and evaluates the **Hypencoder** architecture 
to determine how it compares to encoder models traditionally used for tasks such as entity linking.

## What is a Hypencoder?

Traditional Entity Linking relies on **Bi-Encoders** and simple vector similarities (dot-product or cosine). While efficient, this approach creates a "representation bottleneck" where the nuances of a query must be compressed into a single fixed-length vector.

Based on the research in *["Hypencoder: Hypernetworks for Information Retrieval"](https://arxiv.org/abs/2502.05364)*, this project shifts the paradigm:

* **Query as a Network:** Instead of a vector, a query is encoded into a small, specialized neural network (a **Q-Net**).
* **Dynamic Relevance:** Document embeddings are processed through this Q-Net to produce a relevance score, allowing for much higher expressivity than a simple inner product.
* **Domain Adaptation:** The implementation in this repository specifically adapts this technique for **Medical Entity Linking**, utilizing datasets like **BC5CDR** and **MeSH** to test the model in specialized domains.

---

## Technical Highlights
This project demonstrates the following:

* **Architecture Modififcation:** Adapted the official Hypencoder framework to be accessible on low-compute resources such as free-tier cloud compute services (e.g. Google Colab). In addition, it was adapted to support Entity Linking tasks.
* **Full ML Pipeline:** Built an end-to-end pipeline. This repository provides you with all steps from extracting the data, processing it, to training models, to finally evaluating them.
* **Comparative Benchmarking:** Evaluated Hypencoder performance against industry-standard fine-tuned Bi-Encoders to analyze trade-offs in accuracy and latency.

---

## 📁 Repository Structure

```text
├── hypencoder-paper/          # Core Hypencoder implementation and research code
├── src/entity_linking/        # Custom EL logic, models, and benchmarking tools
├── scripts/etl/               # Data processing for BC5CDR and MeSH sources
├── notebooks/                 # Experimental results and SapBERT vs. Hypencoder comparisons
└── train/configs/             # YAML configurations for various layer depths (2 to 8 layers)

```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Steve-Falkovsky/hypencoder-entity-linking.git
cd hypencoder-entity-linking

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e ./hypencoder-paper

```

---

## Quick Start: Using the Model

You can experiment with pre-trained configurations or your own fine-tuned versions using the provided notebooks.
To find datasets and trained/fine-tuned hypencoder models you can check out my profile on Hugging Face: https://huggingface.co/Stevenf232

### Comparing Models

To see the performance difference between a standard fine-tuned SapBERT and the Hypencoder approach, refer to the relevant notebooks

### Training a Prototype

Configuration files for different architectural depths are located in `hypencoder_cb/train/configs/`. For example, to train a 2-layer Hypencoder:

```bash
python hypencoder_cb/train/train.py --config hypencoder_cb/train/configs/hypencoder.2_layer.yaml

```

Also, refer to the original paper's repo for more details: https://github.com/jfkback/hypencoder-paper

---

## 📊 Key Findings & Results

In progress...
---

## 📜 Citation & Credits

This project builds upon the work of Julian Killingback, Hansi Zeng, and Hamed Zamani.

```bibtex
@misc{killingback2025hypencoderhypernetworksinformationretrieval,
      title={Hypencoder: Hypernetworks for Information Retrieval}, 
      author={Julian Killingback and Hansi Zeng and Hamed Zamani},
      year={2025},
      eprint={2502.05364},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2502.05364}, 
}

```

---
