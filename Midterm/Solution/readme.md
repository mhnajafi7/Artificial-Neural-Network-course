# Neural Network Midterm - Spring 2024

## Mohammad Hossein Najafi - 97103938

---

This repository contains the implementation and analysis of the Hopfield network for memory association, as well as work with the Self-Organizing Map (SOM) network in image segmentation.

### Table of Contents

1. [Introduction](#introduction)
2. [Part 1 - Hopfield Network](#part-1---hopfield-network)
   - [Prerequisites](#prerequisites)
   - [Patterns](#patterns)
   - [Display the Patterns](#display-the-patterns)
   - [Hopfield Network](#hopfield-network)
   - [Training](#training)
   - [Display Corrupted Images](#display-corrupted-images)
   - [Restoring](#restoring)
   - [Animating Restoring Noisy Patterns](#animating-restoring-noisy-patterns)
   - [Energy](#energy)
   - [Sketch](#sketch)
3. [Part 2 - SOM Networks](#part-2---som-networks)
4. [How to Run](#how-to-run)

## Introduction

This project focuses on the Hopfield network and the Self-Organizing Map (SOM) network. The Hopfield network is used for memory association, while the SOM network is applied in image segmentation.

## Part 1 - Hopfield Network

### Prerequisites

First, we import the `numpy` library for working with arrays and the `matplotlib` library for displaying images. The `HTML` library is used for animation purposes.

### Patterns

We define three different patterns in a list of 225 elements, which can be displayed as a 15x15 2D representation.

### Display the Patterns

The patterns are displayed in a 15x15 2D representation.

### Hopfield Network

A model based on the Hopfield network is defined to learn these patterns.

### Training

We initialize the model with 225 neurons and train the model with the patterns. For validation, the same patterns are fed back to the model to generate predictions.

### Display Corrupted Images

### Restoring

For each noise-corrupted pattern, it is fed into the Hopfield network. The network will converge to one of the original patterns (or maybe other things), effectively restoring it.

### Animating Restoring Noisy Patterns

Examples:
- Batman 10% Noise
- Batman 25% Noise
- Batman 50% Noise
- Batman 75% Noise
- A 10% noise
- A 25% noise
- A 50% noise
- A 75% noisy
- Z 10% noisy
- Z 25% noisy
- Z 50% noisy
- Z 75% noisy

### Energy

The energy function in Hopfield networks is defined as follows:

\[ E = -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} w_{ij} s_i s_j + \sum_{i=1}^{N} \theta_i s_i \]

where \(\theta\) is the threshold of each neuron. For simplicity, we assume \(\theta = 0\), so we do not use the second part of this formula to calculate Energy.

### Sketch

Sketching energy versus step for each noisy pattern until restoration.

## Part 2 - SOM Networks

Details of the SOM network implementation and its application in image segmentation are provided in the notebook.

## How to Run

1. Ensure you have the necessary libraries installed: `numpy`, `matplotlib`, and `HTML`.
2. Open the Jupyter notebook `Midterm.ipynb`.
3. Execute the cells in sequence to reproduce the results.

## Code Snippets

Here are a few code snippets from the notebook:

```python
# Hopfield Network Implementation Example
model = Hopfield_Network(225) # Create a Hopfield Network with 225 neurons
model.train(patt) # Train the network with the patterns

(output,m) = model.predict(patt[0]) # Predict the output of the network given the Batman pattern
plt.imshow(np.array(output).reshape(15,15).T,cmap='binary') # Display the output
plt.show()

(output,n) = model.predict(patt[1]) # Predict the output of the network given the a pattern
plt.imshow(np.array(output).reshape(15,15).T,cmap='binary') # Display the output
plt.show()

(output,p) = model.predict(patt[2]) # Predict the output of the network given the z pattern
plt.imshow(np.array(output).reshape(15,15).T,cmap='binary') # Display the output
plt.show()
```
For more details, please refer to the Jupyter notebook 'Codes/Midterm.ipynb'.