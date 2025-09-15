# Brain MRI Scanner Domain Adaptation

This project explores methods to build machine learning models that can transfer and adapt well between brain MRI scans from two different scanners. The goal is to improve model generalization and robustness across different scanner domains.

## Problem Overview

MRI scans from different scanners often vary in appearance and characteristics, making it challenging for models trained on one scanner’s data to perform well on another. This project investigates techniques to bridge this domain gap. This is a binary classification problem predicting whether schizophrenia treatment will be effective.

## Data

- Brain MRI scans collected from **Scanner EP1** and **Scanner EP2**.

## Dataset

The project uses anonymized brain MRI scans from two scanners collected under approved protocols.  
Due to privacy and ethical considerations, the raw data cannot be publicly shared.  
Researchers interested in collaboration or data access may contact alexanderdanieldsouza@gmail.com.

## Methods

- Frozen Decoder - Train a two-headed model on EP1, transform EP2 using the saved decoder, train a new encoder for EP2, and reuse the existing prediction head.
- Frozen Predictor - Where we first train an autoencoder and predictor on EP1. Then, we freeze the predictor and train a new autoencoder on EP2. This process creates a new latent space optimized using the prediction loss, enabling accurate predictions on EP2 based on this latent representation.

## Results

- Accuracy measures of how each method performed on cross-scanner transfer tasks.
- Visualizations.


## License

This project is licensed under the MIT License. See the LICENSE file for details.
