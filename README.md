# DeepLearning
This repository contains HW and projects of IE 534 in 18 Fall and other deep learning materials.

**HW1**: Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch). The neural network should be trained on the Training Set using stochastic gradient descent. It should achieve 97-98% accuracy on the Test Set.

**HW2**: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch). You should write your own code for convolutions (e.g., do not use SciPy’s convolution function). The convolution network should have a single hidden layer with multiple channels.

**HW3**: Train a deep convolution network on a GPU with PyTorch for the CIFAR10 dataset. The convolution network should use (A) dropout, (B) trained with RMSprop or ADAM, and (C) data augmentation. For 10% extra credit, compare dropout test accuracy (i) using the heuristic prediction rule and (ii) Monte Carlo simulation.

**HW4**: Build the Residual Network specified in Figure 1 and achieve at least 60% test accuracy./
     Fine-tune a pre-trained ResNet-18 model and achieve at least 70% test accuracy
     
**HW5**: Image Ranking Project

**HW6**: GAN Project

**HW7&HW8**: NLP Project

**Project**: 

The Show-and-Tell paper proposed in 2015 makes a progress on automatically describing the content of an image. In this paper, they present a generative model based on a deep recurrent aneural network that combines a recent advance in computer vision and machine translation. The model is trained to maximize the likelihood of the target description sentence given the training image. They apply the model on different datasets and evaluated with different metrics like BLEU, METEOR and CIDER. The results show that there is a significant improvement.

In this project, we aim to implement the model in the paper with PyTorch framework.

