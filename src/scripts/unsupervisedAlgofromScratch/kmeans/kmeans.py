import numpy as np

class KMeansFromScratch:
    """
    Implémentation K-means from scratch OPTIMISÉE
    Compatible avec l'API sklearn pour être utilisé dans vos fonctions existantes
    """
    def __init__(self, n_clusters=8, max_iter=300, n_init=10, random_state=None):
        """
        Parameters:
        -----------
        n_clusters : int, nombre de clusters
        max_iter : int, nombre maximum d'itérations
        n_init : int, nombre de fois qu'on relance l'algorithme
        random_state : int, pour la reproductibilité
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X, seed):
        """Initialisation aléatoire des centroïdes (méthode Forgy)"""
        np.random.seed(seed)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[random_indices].copy()
    
    def _compute_distances(self, X, centroids):
        """
        ⚡ VERSION OPTIMISÉE avec broadcasting vectorisé
        Calcule la distance euclidienne entre points et centroïdes
        """
        # Broadcasting : (n_samples, 1, n_features) - (1, n_clusters, n_features)
        # Résultat : (n_samples, n_clusters, n_features)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        
        # Calcul vectorisé des distances euclidiennes
        distances = np.sqrt(np.sum(diff**2, axis=2))  # (n_samples, n_clusters)
        
        return distances
    
    def _assign_clusters(self, distances):
        """Assigne chaque point au cluster le plus proche"""
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """
        ⚡ VERSION OPTIMISÉE avec opérations vectorisées
        Met à jour les centroïdes (moyenne des points de chaque cluster)
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            mask = labels == k
            cluster_points = X[mask]
            
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Cluster vide: réinitialiser avec un point aléatoire
                centroids[k] = X[np.random.randint(0, X.shape[0])]
        
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """
        ⚡ VERSION OPTIMISÉE vectorisée
        Calcule l'inertie (WCSS)
        """
        # Vectorisation : calculer les distances au carré pour tous les points
        distances_sq = np.sum((X - centroids[labels])**2, axis=1)
        return np.sum(distances_sq)
    
    def fit(self, X, y=None):
        """
        Entraîne le modèle K-means
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : ignoré (pour compatibilité sklearn)
        
        Returns:
        --------
        self
        """
        X = np.asarray(X, dtype=np.float64)  # ✅ Conversion optimisée
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_n_iter = 0
        
        # Plusieurs initialisations pour éviter les minima locaux
        for init_idx in range(self.n_init):
            seed = self.random_state + init_idx if self.random_state is not None else None
            
            # Initialisation
            centroids = self._initialize_centroids(X, seed)
            
            # Boucle principale
            for iteration in range(self.max_iter):
                # E-step: Assigner les clusters
                distances = self._compute_distances(X, centroids)
                labels = self._assign_clusters(distances)
                
                # M-step: Mettre à jour les centroïdes
                new_centroids = self._update_centroids(X, labels)
                
                # ✅ Vérifier la convergence de manière optimisée
                if np.allclose(centroids, new_centroids, rtol=1e-6, atol=1e-8):
                    break
                
                centroids = new_centroids
            
            # Calculer l'inertie
            inertia = self._compute_inertia(X, labels, centroids)
            
            # Garder la meilleure solution
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = iteration + 1
        
        # Sauvegarder les meilleurs résultats
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        
        return self
    
    def predict(self, X):
        """
        Prédit les clusters pour de nouvelles données
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        labels : array, shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)
    
    def fit_predict(self, X, y=None):
        """Entraîne et retourne les labels"""
        self.fit(X, y)
        return self.labels_
