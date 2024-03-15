# OutFit Compatibility Prediction and Diagnosis


In this project we aim to predict OutFit Compatibility Prediction and Diagnosis using Polyvore Dataset.

Model Explanation:

1. We aim to learn compatibility between clothing items using an end-to-end comparison network which consists of four steps. 
Extracting features for different concepts like color, style etc. using a multi-layer feature extractor with ResNet-50 CNN architecture mentioned in section 3.1. 

2. Creating enumerated pairwise similarity between features using comparison modules in section 3.3.

3. Computing the compatibility score of the comparison modules using a Multi-layered perceptron network (MCN) to compute in section 3.2. The MCN is first used to predict outfit compatibility and then by backpropagating the gradients from the output score to input to determine the relation of each similarity to the compatibility score.

4. Enabling multi-modal inputs using a visual semantic embedding.

Model Architecture:

![alt text](https://github.com/ganeshparsads/OutFitCompatibility/blob/main/data/images/Model.jpeg?raw=true)

