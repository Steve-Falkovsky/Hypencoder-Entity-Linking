## System Requirements

- Google Colab

## Environment Configuration

Hypencoder notebooks begins with a setup cell that:

1. Installs required dependencies
2. Downgrades specific libraries to compatible versions:
   - transformers == 4.50.0
   - sentence-transformers == 2.2.2

This step is necessary due to compatibility issues with newer versions.

## External Dependencies

### [HuggingFace Repository](https://huggingface.co/Stevenf232)

All trained models and processed datasets are hosted on HuggingFace:
RQ1 and RQ3 - [hypencoder_2layer_SapBERT_context](https://huggingface.co/Stevenf232/hypencoder_2layer_SapBERT_context)

RQ2 (Different number of hidden layers) - `huggingface.co/Stevenf232/hypencoder_<number of layers>layer_SapBERT_context`
