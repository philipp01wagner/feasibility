import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


class GPModelWrapper:
    """
    Wrapper around BoTorch's SingleTaskGP to provide the interface expected by
    DiscreteSafeAcquisitionOptimizer (lower_confidence_bound, add_observations, etc.)
    """
    
    def __init__(self, train_X, train_Y, beta=2.0):
        """
        Initialize GP model wrapper.
        
        Parameters:
        -----------
        train_X : torch.Tensor
            Training inputs, shape (n, d)
        train_Y : torch.Tensor
            Training outputs, shape (n, 1)
        beta : float
            Confidence bound parameter (larger = more conservative)
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.beta = beta
        
        # Create and fit the GP model
        self._update_model()
    
    def _update_model(self):
        """Refit the GP model with current training data."""
        self.model = SingleTaskGP(self.train_X, self.train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
    
    def lower_confidence_bound(self, X):
        """
        Compute lower confidence bound: mean - beta * std_dev
        
        Parameters:
        -----------
        X : torch.Tensor
            Points to evaluate, shape (n, d)
            
        Returns:
        --------
        lcb : torch.Tensor
            Lower confidence bound values, shape (n,)
        """
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            std = torch.sqrt(variance)
            lcb = mean - self.beta * std
        return lcb
    
    def add_observations(self, X, Y):
        """
        Add new observations and refit the model.
        
        Parameters:
        -----------
        X : torch.Tensor
            New input observations, shape (n, d)
        Y : torch.Tensor
            New output observations, shape (n, 1) or (n,)
        """
        # Ensure Y has correct shape
        if Y.dim() == 1:
            Y = Y.unsqueeze(-1)
        
        # Append to training data
        self.train_X = torch.cat([self.train_X, X])
        self.train_Y = torch.cat([self.train_Y, Y])
        
        # Refit model
        self._update_model()
    
    def predict(self, X):
        """
        Get mean predictions.
        
        Parameters:
        -----------
        X : torch.Tensor
            Points to evaluate, shape (n, d)
            
        Returns:
        --------
        mean : torch.Tensor
            Predicted mean values, shape (n,)
        """
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X)
            mean = posterior.mean.squeeze(-1)
        return mean
    
    @property
    def covar_module(self):
        """Access to the underlying covariance module."""
        return self.model.covar_module
    
    @property
    def likelihood(self):
        """Access to the underlying likelihood."""
        return self.model.likelihood
