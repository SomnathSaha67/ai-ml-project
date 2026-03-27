"""MLP optimizer implementations."""

import numpy as np
from typing import Dict, Any


class SGDOptimizer:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate.
            momentum: Momentum parameter.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Update parameters.
        
        Args:
            params: Parameter dictionary.
            grads: Gradient dictionary.
        """
        if self.velocity is None:
            self.velocity = {}
            for key in params:
                self.velocity[key] = np.zeros_like(params[key])
        
        for key in params:
            self.velocity[key] = (
                self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            )
            params[key] += self.velocity[key]


class AdamOptimizer:
    """Adam optimizer."""
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate.
            beta1: Exponential decay rate for 1st moment.
            beta2: Exponential decay rate for 2nd moment.
            epsilon: Small value for numerical stability.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Update parameters.
        
        Args:
            params: Parameter dictionary.
            grads: Gradient dictionary.
        """
        if self.m is None:
            self.m = {}
            self.v = {}
            for key in params:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
        
        self.t += 1
        
        for key in params:
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
