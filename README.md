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
├── hypencoder-paper/          # Core Hypencoder implementation, training configs (for creating your own models)
├── src/entity_linking/        # Custom EL logic, models, and benchmarking tools
├── scripts/etl/               # Data processing for BC5CDR and MeSH sources
├── notebooks/                 # Experimental results and SapBERT vs. Hypencoder comparisons

```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Steve-Falkovsky/hypencoder-entity-linking.git
cd hypencoder-entity-linking

```

### Create and activate your virtual Environment (Optional, but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate 
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
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

## End to end pipeline
### Get BC5CDR data
```bash
./scripts/etl/fetch_bc5cdr_sources.sh
```

### Process and format the data (incuding MeSH2015)
```bash
python scripts/etl/prepare_bc5cdr_and_mesh.py --bc5cdr_dir data/raw/corpora_sources/CDR_Data/CDR.Corpus.v010516 --mesh2015_dir data/raw/corpora_sources/mesh2015 --out_train data/processed/bc5cdr_train.bioc.xml.gz --out_val data/processed/bc5cdr_val.bioc.xml.gz --out_test data/processed/bc5cdr_test.bioc.xml.gz --out_ontology data/processed/mesh2015.json.gz
```

### Now you can create your own datasets:
Examples for scripts are:
`src/entity_linking/nameonly_to_jsonl.py`
`src/entity_linking/contrastive_loss_jsonl.py`

### Create your own models by:
1. Creating a configuration by modifying a .YAML config file such as `hypencoder-paper/hypencoder_cb/train/configs/hypencoder.2_layer_finetuned_BC5CDR.yaml`
2. Training the model using the relevant notebook such as `notebooks/fine_tune_Hypencoder_on_BC5CDR.ipynb`


---


## 📊 Key Findings & Results

In progress...
---

## 📜 Citation & Credits

This project builds upon the work of Julian Killingback, Hansi Zeng, and Hamed Zamani on Hypencoders.

```bibtex
@inproceedings{hypencoder,
    author = {Killingback, Julian and Zeng, Hansi and Zamani, Hamed},
    title = {Hypencoder: Hypernetworks for Information Retrieval},
    year = {2025},
    isbn = {9798400715921},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3726302.3729983},
    doi = {10.1145/3726302.3729983},
    booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {2372–2383},
    numpages = {12},
    keywords = {learning to rank, neural ranking models, retrieval models},
    location = {Padua, Italy},
    series = {SIGIR '25}
}

```

This project uses data extraction pipelines from work of Javier Sanz-Cruzado and Jake Lever. 
```bibtex
@inproceedings{
    title = {Accelerating Cross-Encoders in Biomedical Entity Linking},
    author = {Sanz-Cruzado, Javier and Lever, Jake},
    booktitle = {Proceedings of the 24th Workshop on Biomedical Language Processing},
    month = {jul},
    year = {2025},
    address = {Vienna, Austria},
    publisher = {Association for Computational Linguistics},
}

```

---
