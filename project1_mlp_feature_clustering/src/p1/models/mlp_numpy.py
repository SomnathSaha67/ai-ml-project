"""MLP Neural Network implementation from scratch using NumPy."""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import json


class MLPNumpy:
    """
    Multi-layer Perceptron from scratch using NumPy.
    
    Supports:
    - Configurable hidden layers with ReLU or tanh activations
    - Binary (sigmoid) or multiclass (softmax) output
    - Cross-entropy or binary cross-entropy loss
    - Backpropagation algorithm
    - Mini-batch gradient descent
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        hidden_activation: str = 'relu',
        output_activation: str = 'sigmoid',
        seed: int = 42,
    ):
        """
        Initialize MLP.
        
        Args:
            input_size: Input feature dimension.
            hidden_sizes: List of hidden layer sizes.
            output_size: Output size.
            hidden_activation: 'relu' or 'tanh'.
            output_activation: 'sigmoid' (binary) or 'softmax' (multiclass).
            seed: Random seed.
        """
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        # Build network architecture
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize parameters
        self.params = {}
        self.caches = {}
        self._initialize_parameters(layer_sizes)
    
    def _initialize_parameters(self, layer_sizes: List[int]) -> None:
        """Initialize weights and biases using He initialization."""
        for i in range(1, len(layer_sizes)):
            W = np.random.randn(
                layer_sizes[i - 1], layer_sizes[i]
            ) * np.sqrt(2.0 / layer_sizes[i - 1])
            b = np.zeros((1, layer_sizes[i]))
            
            self.params[f'W{i}'] = W
            self.params[f'b{i}'] = b
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def _relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(z, 0.0)
    
    def _relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return (z > 0).astype(float)
    
    def _tanh(self, z: np.ndarray) -> np.ndarray:
        """Tanh activation."""
        return np.tanh(z)
    
    def _tanh_derivative(self, z: np.ndarray) -> np.ndarray:
        """Tanh derivative."""
        return 1.0 - np.tanh(z) ** 2
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        z = z - np.max(z, axis=1, keepdims=True)  # For numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.caches.clear()
        A = X
        
        for i in range(1, self.n_layers + 1):
            Z = np.dot(A, self.params[f'W{i}']) + self.params[f'b{i}']
            
            # Store cache for backprop
            self.caches[f'A{i-1}'] = A
            self.caches[f'Z{i}'] = Z
            
            # Apply activation
            if i == self.n_layers:
                # Output layer
                if self.output_activation == 'sigmoid':
                    A = self._sigmoid(Z)
                elif self.output_activation == 'softmax':
                    A = self._softmax(Z)
            else:
                # Hidden layer
                if self.hidden_activation == 'relu':
                    A = self._relu(Z)
                elif self.hidden_activation == 'tanh':
                    A = self._tanh(Z)
        
        return A
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute loss."""
        m = y_true.shape[0]
        
        if self.output_activation == 'sigmoid':
            # Binary cross-entropy
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            loss = -np.mean(
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            )
        else:
            # Categorical cross-entropy
            y_pred = np.clip(y_pred, 1e-7, 1)
            loss = -np.mean(y_true * np.log(y_pred))
        
        return loss
    
    def _backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass with backpropagation."""
        m = y_true.shape[0]
        grads = {}
        
        # Output layer gradient
        if self.output_activation == 'sigmoid':
            dA = y_pred - y_true
        else:
            dA = y_pred - y_true
        
        # Backprop through layers
        for i in range(self.n_layers, 0, -1):
            A_prev = self.caches[f'A{i-1}']
            Z = self.caches[f'Z{i}']
            
            # Gradient of loss w.r.t. Z
            if i < self.n_layers:
                # Hidden layer
                if self.hidden_activation == 'relu':
                    dZ = dA * self._relu_derivative(Z)
                elif self.hidden_activation == 'tanh':
                    dZ = dA * self._tanh_derivative(Z)
            else:
                dZ = dA
            
            # Gradients w.r.t. W and b
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            grads[f'W{i}'] = dW
            grads[f'b{i}'] = db
            
            # Gradient w.r.t. previous layer
            if i > 1:
                dA = np.dot(dZ, self.params[f'W{i}'].T)
        
        return grads
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_name: str = 'sgd',
    ) -> Dict[str, Any]:
        """
        Train the MLP.
        
        Args:
            X: Training features (n_samples, n_features).
            y: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            epochs: Number of epochs.
            batch_size: Mini-batch size.
            learning_rate: Learning rate.
            optimizer_name: 'sgd' or 'adam'.
            
        Returns:
            Dictionary with training history.
        """
        # Handle binary vs multiclass
        if y.ndim == 1:
            n_classes = len(np.unique(y))
            if n_classes == 2:
                y_train = y.reshape(-1, 1).astype(float)
            else:
                # One-hot encode
                y_train = np.eye(n_classes)[y]
        else:
            y_train = y.astype(float)
        
        if y_val is not None:
            if y_val.ndim == 1:
                n_classes = len(np.unique(y_val))
                if n_classes == 2:
                    y_val = y_val.reshape(-1, 1).astype(float)
                else:
                    y_val = np.eye(n_classes)[y_val]
            else:
                y_val = y_val.astype(float)
        
        # Initialize optimizer
        if optimizer_name == 'adam':
            from .optimizers import AdamOptimizer
            optimizer = AdamOptimizer(learning_rate=learning_rate)
        else:
            from .optimizers import SGDOptimizer
            optimizer = SGDOptimizer(learning_rate=learning_rate)
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch gradient descent
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                X_batch = X_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]
                
                # Forward
                y_pred = self._forward(X_batch)
                loss = self._compute_loss(y_pred, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # Backward
                grads = self._backward(y_pred, y_batch)
                
                # Update
                optimizer.update(self.params, grads)
            
            # Record metrics
            history['train_loss'].append(epoch_loss / n_batches)
            
            # Train accuracy
            y_pred_train = self._forward(X)
            if self.output_activation == 'sigmoid':
                y_pred_train_class = (y_pred_train > 0.5).astype(int)
                y_true_class = y_train.astype(int)
            else:
                y_pred_train_class = np.argmax(y_pred_train, axis=1)
                y_true_class = np.argmax(y_train, axis=1)
            train_acc = np.mean(y_pred_train_class == y_true_class)
            history['train_accuracy'].append(train_acc)
            
            # Validation metrics
            if X_val is not None:
                y_pred_val = self._forward(X_val)
                val_loss = self._compute_loss(y_pred_val, y_val)
                history['val_loss'].append(val_loss)
                
                if self.output_activation == 'sigmoid':
                    y_pred_val_class = (y_pred_val > 0.5).astype(int)
                    y_true_val_class = y_val.astype(int)
                else:
                    y_pred_val_class = np.argmax(y_pred_val, axis=1)
                    y_true_val_class = np.argmax(y_val, axis=1)
                val_acc = np.mean(y_pred_val_class == y_true_val_class)
                history['val_accuracy'].append(val_acc)
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted class labels.
        """
        y_pred = self._forward(X)
        if self.output_activation == 'sigmoid':
            # Binary: output shape (N, 1) or (N,)
            if y_pred.ndim == 2 and y_pred.shape[1] == 1:
                return (y_pred > 0.5).astype(int).ravel()
            return (y_pred > 0.5).astype(int)
        else:
            return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        return self._forward(X)
