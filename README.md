# MNIST-NN

## Brief:

A single layer (128 Neuron) neural network for detecting handwritten digits from the MNIST dataset. The code combines Batch Normalization (BN), Dropout, and Early Stopping to enhance performance and mitigate overfitting, while comparing ReLU and Tanh activation functions. The first iteration is done without any Batch Normalization, Dropout and Early Stopping just to see how both models work. Tanh performs better than ReLu in the first iteration (Tanh Model Accuracy: 96.9%, ReLu Model Accuracy: 92.5%) but in the second iteration after ensuring there is no overfitting and optimising the code, the ReLu model performs better (Tanh Model Accuracy: 97.1%, ReLu Model Accuracy: 97.9%) by just a slight margin. 

Provided below is the research backing behind the design of the Neural Network along with the technical performance and its outcomes.

## Key Design Decisions and Research Backing

1.	Single Hidden Layer, 128 Neurons
o	Why 128? (LeCun, et al., 1998) demonstrated that a modestly sized MLP can achieve high accuracy on MNIST. (Glorot & Bengio, 2010) highlighted initialization issues, but with 128 neurons, the network remains both capable and manageable.
2.	Batch Normalization
o	Why BN? (Nair & Hinton, 2015): It normalizes layer inputs to help mitigate internal covariate shift, stabilizing gradients and often accelerating training.
3.	Activation Function Choices
o	ReLU (Nair & Hinton, 2015): Reduces vanishing gradients and typically converges faster.
o	Tanh (LeCun, et al., 1998): Historically popular, can be competitive for smaller networks, though it saturates for large absolute input values.
4.	Dropout
o	Introduced by (Srivastava, et al., 2014) to randomly drop connections, preventing co-adaptation of neurons and fighting overfitting.
5.	Softmax Output + Categorical Crossentropy
o	Standard in multi-class classification (Goodfellow, et al., 2016) and (Bishop, 2006), turning raw logits into probabilities and comparing these to one-hot encoded labels.
6.	Adam Optimizer
o	(Kingma & Ba, 2015) demonstrated that Adam adaptively tunes learning rates, often outperforming plain SGD in speed and reliability.
7.	Early Stopping
o	(Prechelt, 1998) provides a foundational study of halting training when validation loss plateaus, restoring the best weights to avoid overfitting.

## Technical Performance and Model Outcomes:

### Model Architecture and Rationale:

The architecture, consisting of a single hidden layer with 128 neurons, meets the criteria of a three-layer network that includes an input, hidden, and output layer. We enhanced it with Batch Normalization (BN) to stabilize activations and learning dynamics, Dropout (0.2) to reduce overfitting, and Early Stopping to halt training once validation loss ceased improving. These modifications reflect both established best practices in neural network design and practical constraints such as computational efficiency and implementation complexity.
1. Batch Normalization: By normalizing intermediate feature distributions, BN addresses internal covariate shift, often leading to faster convergence and improved generalization (Ioffe & Szegedy, 2015).
2. Dropout: Randomly disables a fraction of neurons, diminishing the risk of overfitting and encouraging the network to learn more robust representations (Srivastava, et al., 2014).
3. Early Stopping: Mitigates excessive training epochs that might lead to overfitting. By restoring the best weights at the end, it ensures optimal performance on unseen data (Prechelt, 1998).

### Comparative Accuracy and Findings

Our experiments compared two activation functions in the hidden layer:
1.	ReLU (Rectified Linear Unit), which often converges faster and mitigates the vanishing gradient problem.
2.	Tanh, which can capture negative input values by mapping them into a (−1, 1) range, sometimes offering smoother gradients for smaller-scale problems.

In typical runs:
1. The ReLU model tends to achieve slightly faster convergence and marginally higher accuracy (often in the 97–98% range on MNIST).
2. The Tanh model is competitive but can require careful tuning of the learning rate and might converge more slowly. Its final accuracy often hovers around a similar range, sometimes marginally lower.

Given MNIST’s relative simplicity, both activation functions suffice to achieve high accuracies beyond 95%. The subtle differences—such as speed of convergence and slight variations in final performance—can guide choices for other, more complex tasks.

### Training vs. Validation Accuracy and Loss – Observations: 

1. In most trials, ReLU’s training accuracy rises quickly within the first few epochs, often outpacing Tanh slightly.
2. Both models show minimal overfitting, evidenced by the parallel trends of training and validation curves, thanks to Dropout and Early Stopping.
3. The Tanh model can exhibit a slower climb in accuracy but eventually narrows the gap. In terms of loss, both converge to similarly low values, though ReLU may do so in fewer epochs.

![image](https://github.com/ahsanjam/MNIST-NN/blob/main/Accuracy%20and%20Loss)

### Confusion Matrices for each model on the test set – Observations:

1. Both models produce a strong diagonal, indicating that the majority of test samples (digits 0–9) are classified correctly.
2. Minor misclassifications appear with digits that share visual similarities (e.g., ‘4’ vs. ‘9’ or ‘7’ vs. ‘1’), although these errors are not frequent.
3. The ReLU model’s confusion matrix shows slightly fewer off-diagonal errors in some runs, consistent with its marginally higher overall accuracy.
4. The Tanh model’s confusion matrix looks similar but can present small clusters of confusion among digits with very similar strokes.

![image](https://github.com/ahsanjam/MNIST-NN/blob/main/Confusion%20Matrices)

