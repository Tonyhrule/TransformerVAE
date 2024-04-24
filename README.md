# TransformerVAE: Music Generation with Transformer Variational Autoencoder

## Overview
TransformerVAE explores the integration of transformer architectures with Variational Autoencoders (VAE) to evaluate their effectiveness in music generation. This project utilizes transformers' structure-aware capabilities and VAEs' generative potential, known for their efficient latent representations. The objective is to further understand VAEs, compare their performance with RNNs, identify limitations, and discover potential enhancements.

## Project Status

 **[Status]**: Just started

## Next Steps

### Model Enhancement
- **Algorithm**: Refine the Transformer and VAE components to improve learning efficiency and output quality, potentially adjusting the architecture and layer configurations.
- **Dataset**: Currently, the dataset includes corrupt and unprocessable files. Expanding the dataset to include more reliable data that is compatible with the model will greatly benefit the training.

## Installation

### Prerequisites
- Python 3.6 or newer
- PyTorch
- NumPy
- pretty_midi

### Setup
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/Tonyhrule/TransformerVAE.git
cd TransformerVAE
pip install -r requirements.txt
```

## Project Structure

- **'generate.py'**: Script for generating MIDI files from preprocessed NPY data.
- **'model.py'**: Contains the TransformerVAE model architecture.
- **'postprocessing.py'**: Script to convert NPY files back into MIDI format.
- **'preprocessing.py'**: Utility script to convert MIDI files to NPY format.
- **'train.py'**: Contains the training routine for the TransformerVAE model.
- **'utils.py'**: Helper functions and classes for dataset handling and loading.
