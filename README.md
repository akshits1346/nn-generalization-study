# An Empirical Study of Overfitting and Generalization in Neural Networks

## Overview
This project empirically studies overfitting and generalization behavior in neural networks through controlled experiments on model capacity, data size, and regularization.

## Motivation
Neural networks often achieve high training accuracy while generalizing poorly. Understanding this gap is critical for designing robust models.

## Experimental Setup
Experiments are conducted on MNIST using logistic regression and multi-layer perceptrons of varying depth. Training data size and regularization are systematically varied.

## Results
Increasing model capacity improves training performance but does not consistently improve validation accuracy.
Reducing dataset size significantly amplifies the generalization gap.
Dropout regularization improves validation stability at the cost of lower training accuracy.

## Limitations
Experiments are limited to small-scale datasets and simple architectures. Results may not directly extrapolate to large-scale or highly structured data.

