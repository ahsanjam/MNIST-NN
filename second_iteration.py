import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def general_model(hidden_activation="relu"):
    model = keras.Sequential([
        # Single Hidden Layer
        layers.Dense(128, input_shape=(784,)),
        layers.BatchNormalization(),          # Normalize the hidden layer activations
        layers.Activation(hidden_activation), # Flexible choice of activation (ReLU, Tanh, etc.)
        layers.Dropout(0.2),                  # Dropout for regularization

        # Output Layer
        layers.Dense(10, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype("float32") / 255.0
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype("float32") / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create Early Stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,     # stop after 3 epochs of no improvement
    restore_best_weights=True
)

# Build and train model with ReLU
relu_model = general_model(hidden_activation="relu")
history_relu = relu_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on test set
test_loss_relu, test_acc_relu = relu_model.evaluate(x_test, y_test, verbose=0)
print(f"\n[ReLU Model] Test Loss: {test_loss_relu:.4f}, Test Accuracy: {test_acc_relu:.4f}")

# Build and train model with Tanh
tanh_model = general_model(hidden_activation="tanh")
history_tanh = tanh_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluating on the test set
test_loss_tanh, test_acc_tanh = tanh_model.evaluate(x_test, y_test, verbose=0)
print("\n[Tanh Model] Test Loss:", round(test_loss_tanh, 2), "Test Accuracy:", round(test_acc_tanh, 2))


# Compare final accuracies
print("\n--- Model Comparison ---")
print("Test Accuracy for ReLu:", round(test_acc_relu, 2))
print("Test Accuracy for Tanh:", round(test_acc_tanh, 2))

# Visualization

# Create a figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# We want to plot accuracy on the first subplot, and loss on the second.
for ax, metric in zip(axes, ['accuracy', 'loss']):
    # Plot ReLU training and validation
    ax.plot(history_relu.history[metric], 'r', label=f'ReLU Train {metric.title()}')
    ax.plot(history_relu.history['val_' + metric], 'r--', label=f'ReLU Val {metric.title()}')
    
    # Plot Tanh training and validation
    ax.plot(history_tanh.history[metric], 'b', label=f'Tanh Train {metric.title()}')
    ax.plot(history_tanh.history['val_' + metric], 'b--', label=f'Tanh Val {metric.title()}')
    
    # Common labels and legend
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.title())
    ax.legend()

# Set distinct titles for each subplot
axes[0].set_title("Accuracy Over Epochs")
axes[1].set_title("Loss Over Epochs")

plt.tight_layout()
plt.show()

# B) Confusion Matrices
# Convert predictions from one-hot to integer labels
y_pred_relu = np.argmax(relu_model.predict(x_test), axis=1)
y_pred_tanh = np.argmax(tanh_model.predict(x_test), axis=1)
y_true      = np.argmax(y_test, axis=1)

# Compute confusion matrices
cm_relu = confusion_matrix(y_true, y_pred_relu)
cm_tanh = confusion_matrix(y_true, y_pred_tanh)

plt.figure(figsize=(12, 5))

# Plot CM for ReLU Model
plt.subplot(1, 2, 1)
sns.heatmap(cm_relu, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - ReLU")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Plot CM for Tanh Model
plt.subplot(1, 2, 2)
sns.heatmap(cm_tanh, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Tanh")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()