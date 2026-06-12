import numpy as np
from collections import Counter

class KNNClassifier:
    """
    K-Nearest Neighbors (KNN) Classifier implemented from scratch.
    
    Parameters:
    -----------
    k : int, default=5
        Number of neighbors to use.
    metric : str, default='euclidean'
        Distance metric to use. Options: 'euclidean', 'manhattan', 'minkowski'
    p : int, default=3
        Power parameter for the Minkowski distance metric. Only used if metric='minkowski'.
    """
    def __init__(self, k=5, metric='euclidean', p=3):
        self.k = k
        self.metric = metric.lower()
        self.p = p
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target values.
        Since KNN is a lazy learner, this step just stores the training data.
        """
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def _compute_distances(self, x):
        """
        Compute the distances between a single test sample x and all training samples.
        """
        if self.metric == 'euclidean':
            # Euclidean distance: sqrt(sum((x - x_train)^2))
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            # Manhattan distance: sum(|x - x_train|)
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif self.metric == 'minkowski':
            # Minkowski distance: (sum(|x - x_train|^p))^(1/p)
            return np.power(np.sum(np.abs(self.X_train - x) ** self.p, axis=1), 1.0 / self.p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}. Supported metrics: 'euclidean', 'manhattan', 'minkowski'.")

    def _predict_single(self, x, current_k=None):
        """
        Predict the class label for a single query vector x.
        Supports recursive tie-breaking by decreasing k.
        """
        if current_k is None:
            current_k = self.k

        # 1. Compute distances from x to all training samples
        distances = self._compute_distances(x)

        # 2. Get the indices of the current_k nearest neighbors
        # partition is O(N) compared to argsort which is O(N log N)
        k_indices = np.argpartition(distances, current_k - 1)[:current_k]
        
        # Sort these nearest indices specifically by distance
        k_indices_sorted = k_indices[np.argsort(distances[k_indices])]
        
        # 3. Retrieve the labels of the k nearest neighbors
        k_nearest_labels = self.y_train[k_indices_sorted]

        # 4. Perform a majority vote
        counts = Counter(k_nearest_labels)
        most_common = counts.most_common()

        # Check if there is a tie for the most common label
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Tie breaking:
            # If k is already 1, we can't decrease it further; return the label of the closest neighbor
            if current_k <= 1:
                return k_nearest_labels[0]
            # Otherwise, recursively call with current_k - 1
            return self._predict_single(x, current_k - 1)

        # Return the label with the highest vote count
        return most_common[0][0]

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X_arr = np.array(X, dtype=np.float64)
        predictions = [self._predict_single(x) for x in X_arr]
        return np.array(predictions)

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns:
        --------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        X_arr = np.array(X, dtype=np.float64)
        probabilities = []
        
        for x in X_arr:
            distances = self._compute_distances(x)
            # Find K nearest neighbors
            k_indices = np.argpartition(distances, self.k - 1)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Compute probabilities based on frequency of each class in neighbors
            counts = Counter(k_nearest_labels)
            prob = [counts.get(cls, 0) / self.k for cls in self.classes_]
            probabilities.append(prob)
            
        return np.array(probabilities)
