# EKG-compression
Program that can compress EKG data using a neural network

## Description
This project implements a Multi-Layer Perceptron (MLP) Autoencoder for the compression and decompression of ECG (electrocardiogram) signals.

The architecture relies on a "bottleneck" — a middle layer that is significantly smaller than the input and output dimensions. The model is trained to reconstruct its input; however, the reduced dimensionality of the middle layer forces the network to learn a compressed representation of the data. Consequently, the reconstructed output is a slightly distorted version of the original signal.

The primary objective is to evaluate the trade-off between the compression ratio (the size of the middle layer) and the quality of the reconstructed signal, identifying the smallest latent space that yields an acceptable level of distortion.

### Workflow:
- Compression (Encoding): An ECG signal is passed through the network up to the middle layer. The activations of these middle neurons are extracted and saved as the compressed data.
- Decompression (Decoding): The compressed data is injected directly into the middle layer. The subsequent layers process this data to reconstruct the original ECG signal.

## members
- Radosław Ciepał
- Kacper Mazurek
- Krzysztof Kowalik

