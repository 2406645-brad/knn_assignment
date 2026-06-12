import numpy as np
import pandas as pd

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits the dataset into random train and test subsets.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training vectors.
    y : array-like of shape (n_samples,)
        Target values.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int or None, default=None
        Controls the shuffling applied to the data before applying the split.
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # Convert to numpy arrays for reliable indexing
    X_arr = np.array(X)
    y_arr = np.array(y)
    
    n_samples = X_arr.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_samples = int(n_samples * test_size)
    
    test_idx = indices[:test_samples]
    train_idx = indices[test_samples:]
    
    # Return split data, maintaining original types if possible (or returning arrays)
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


class MinMaxScaler:
    """
    Transforms features by scaling each feature to a given range [0, 1].
    Implemented from scratch using NumPy.
    """
    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.range_ = None

    def fit(self, X):
        """Compute the minimum and maximum to be used for later scaling."""
        X_arr = np.array(X, dtype=np.float64)
        self.min_ = np.min(X_arr, axis=0)
        self.max_ = np.max(X_arr, axis=0)
        self.range_ = self.max_ - self.min_
        # Avoid division by zero for constant features
        self.range_[self.range_ == 0.0] = 1.0
        return self

    def transform(self, X):
        """Scale features of X according to feature_range."""
        X_arr = np.array(X, dtype=np.float64)
        return (X_arr - self.min_) / self.range_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range."""
        X_arr = np.array(X, dtype=np.float64)
        return X_arr * self.range_ + self.min_


class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Implemented from scratch using NumPy.
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """Compute the mean and std to be used for later scaling."""
        X_arr = np.array(X, dtype=np.float64)
        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)
        # Avoid division by zero for features with zero variance
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        """Perform standardization by centering and scaling."""
        X_arr = np.array(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Scale back the data to the original representation."""
        X_arr = np.array(X, dtype=np.float64)
        return X_arr * self.scale_ + self.mean_
