# MNIST-NN
A single layer (128 Neuron) neural network for detecting handwritten digits from the MNIST dataset. The code combines Batch Normalization (BN), Dropout, and Early Stopping to enhance performance and mitigate overfitting, while comparing ReLU and Tanh activation functions.


Key Design Decisions and Research Backing
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

Training vs. Validation Accuracy and Loss – Observations:
•	In most trials, ReLU’s training accuracy rises quickly within the first few epochs, often outpacing Tanh slightly.
•	Both models show minimal overfitting, evidenced by the parallel trends of training and validation curves, thanks to Dropout and Early Stopping.
•	The Tanh model can exhibit a slower climb in accuracy but eventually narrows the gap. In terms of loss, both converge to similarly low values, though ReLU may do so in fewer epochs.



