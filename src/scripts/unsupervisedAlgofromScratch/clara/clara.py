# scripts/unsupervisedAlgofromScratch/clara/clara.py

import numpy as np


class CLARAFromScratch:
    """
    CLARA (Clustering LARge Applications)
    Implementation from scratch
    
    CLARA uses sampling to handle large datasets by applying PAM on samples.
    """
    def __init__(self, n_clusters=3, n_sampling=40, n_sampling_iter=5, random_state=42):
        """
        Parameters:
        -----------
        n_clusters : int, number of clusters (k)
        n_sampling : int, sample size for each iteration
                     Should be larger than k, typically 40 + 2*k
        n_sampling_iter : int, number of sampling iterations
                          Higher values = better solution but slower
        random_state : int, seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.n_sampling = n_sampling
        self.n_sampling_iter = n_sampling_iter
        self.random_state = random_state
        self.medoid_indices_ = None
        self.labels_ = None
        self.inertia_ = None
        self.best_cost_ = None
        
    def _compute_distance(self, X, medoid_indices):
        """Compute distances from all points to medoids"""
        distances = np.zeros((X.shape[0], len(medoid_indices)))
        for i, medoid_idx in enumerate(medoid_indices):
            distances[:, i] = np.linalg.norm(X - X[medoid_idx], axis=1)
        return distances
    
    def _assign_clusters(self, distances):
        """Assign each point to nearest medoid"""
        return np.argmin(distances, axis=1)
    
    def _compute_cost(self, distances, labels):
        """Compute total cost (sum of distances to assigned medoids)"""
        cost = 0
        for i in range(len(labels)):
            cost += distances[i, labels[i]]
        return cost
    
    def _pam_on_sample(self, X_sample, sample_indices):
        """
        Apply PAM (Partitioning Around Medoids) on a sample
        
        Returns medoid indices in the ORIGINAL dataset
        """
        n_sample = X_sample.shape[0]
        
        # Initialize: select k random points as initial medoids
        local_medoid_indices = np.random.choice(n_sample, size=self.n_clusters, replace=False)
        
        # PAM iterations
        improved = True
        max_pam_iter = 100
        iteration = 0
        
        while improved and iteration < max_pam_iter:
            improved = False
            iteration += 1
            
            # Compute current cost
            distances = self._compute_distance(X_sample, local_medoid_indices)
            labels = self._assign_clusters(distances)
            current_cost = self._compute_cost(distances, labels)
            
            # Try swapping each medoid with each non-medoid
            for i in range(self.n_clusters):
                non_medoids = np.setdiff1d(np.arange(n_sample), local_medoid_indices)
                
                for j in non_medoids:
                    # Create new configuration
                    new_medoids = local_medoid_indices.copy()
                    new_medoids[i] = j
                    
                    # Compute new cost
                    new_distances = self._compute_distance(X_sample, new_medoids)
                    new_labels = self._assign_clusters(new_distances)
                    new_cost = self._compute_cost(new_distances, new_labels)
                    
                    # If better, update
                    if new_cost < current_cost:
                        local_medoid_indices = new_medoids
                        current_cost = new_cost
                        improved = True
                        break
                
                if improved:
                    break
        
        # Convert local indices to original dataset indices
        medoid_indices_original = sample_indices[local_medoid_indices]
        
        return medoid_indices_original, current_cost
    
    def fit(self, X):
        """
        Perform CLARA clustering
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        self
        """
        np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot be larger than n_samples ({n_samples})")
        
        if self.n_sampling > n_samples:
            self.n_sampling = n_samples
            print(f"⚠️  n_sampling adjusted to {n_samples}")
        
        if self.n_sampling < self.n_clusters:
            raise ValueError(f"n_sampling ({self.n_sampling}) must be >= n_clusters ({self.n_clusters})")
        
        best_medoids = None
        best_cost = np.inf
        
        # Perform n_sampling_iter sampling iterations
        for iter_num in range(self.n_sampling_iter):
            # Draw random sample
            sample_indices = np.random.choice(n_samples, size=self.n_sampling, replace=False)
            X_sample = X[sample_indices]
            
            # Apply PAM on sample
            medoids, sample_cost = self._pam_on_sample(X_sample, sample_indices)
            
            # Evaluate on FULL dataset
            distances = self._compute_distance(X, medoids)
            labels = self._assign_clusters(distances)
            full_cost = self._compute_cost(distances, labels)
            
            # Update best solution
            if full_cost < best_cost:
                best_cost = full_cost
                best_medoids = medoids
        
        # Final assignment with best medoids
        distances = self._compute_distance(X, best_medoids)
        best_labels = self._assign_clusters(distances)
        
        self.medoid_indices_ = best_medoids
        self.labels_ = best_labels
        self.inertia_ = best_cost
        self.best_cost_ = best_cost
        
        return self
    
    def fit_predict(self, X):
        """Perform clustering and return cluster labels"""
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        """Assign new points to nearest medoid"""
        if self.medoid_indices_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        distances = self._compute_distance(X, self.medoid_indices_)
        return self._assign_clusters(distances)
