# scripts/unsupervisedAlgofromScratch/hdbscan/hdbscan.py

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


class HDBSCANFromScratch:
    """
    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
    Simplified implementation from scratch
    """
    def __init__(self, min_cluster_size=5, min_samples=None):
        """
        Parameters:
        -----------
        min_cluster_size : int, minimum size of clusters
        min_samples : int, number of samples in neighborhood for core points
                      If None, defaults to min_cluster_size
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.labels_ = None
        self.probabilities_ = None
        
    def _compute_core_distances(self, X):
        """
        Compute core distance for each point (distance to k-th nearest neighbor)
        """
        n_samples = X.shape[0]
        core_distances = np.zeros(n_samples)
        
        # Compute pairwise distances
        distances = squareform(pdist(X))
        
        # For each point, find distance to min_samples-th nearest neighbor
        for i in range(n_samples):
            # Sort distances (excluding self)
            sorted_dists = np.sort(distances[i])
            # k-th nearest neighbor (k = min_samples)
            core_distances[i] = sorted_dists[self.min_samples]
        
        return core_distances, distances
    
    def _compute_mutual_reachability(self, distances, core_distances):
        """
        Compute mutual reachability distance matrix
        mutual_reach(a,b) = max(core_dist(a), core_dist(b), dist(a,b))
        """
        n_samples = len(core_distances)
        mutual_reach = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    mutual_reach[i, j] = max(
                        core_distances[i],
                        core_distances[j],
                        distances[i, j]
                    )
        
        return mutual_reach
    
    def _build_mst(self, mutual_reach):
        """Build Minimum Spanning Tree from mutual reachability distances"""
        # Use scipy's minimum_spanning_tree
        mst = minimum_spanning_tree(mutual_reach)
        return mst
    
    def _single_linkage_tree(self, mst):
        """
        Build single linkage dendrogram from MST
        Returns: hierarchy as list of (point1, point2, distance, cluster_size)
        """
        n_samples = mst.shape[0]
        mst_array = mst.toarray()
        
        # Extract edges from MST
        edges = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if mst_array[i, j] > 0:
                    edges.append((i, j, mst_array[i, j]))
                elif mst_array[j, i] > 0:
                    edges.append((i, j, mst_array[j, i]))
        
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])
        
        return edges
    
    def _extract_clusters(self, edges, n_samples):
        """
        Extract clusters from hierarchy using min_cluster_size
        Simplified version: uses connected components at different thresholds
        """
        # Union-Find data structure for connected components
        parent = list(range(n_samples))
        rank = [0] * n_samples
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
        
        # Process edges to build clusters
        labels = np.full(n_samples, -1)
        cluster_id = 0
        
        # Gradually add edges (starting from smallest distances)
        for i, j, dist in edges:
            union(i, j)
        
        # Extract final clusters
        cluster_map = {}
        for i in range(n_samples):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(i)
        
        # Assign labels only to clusters >= min_cluster_size
        for members in cluster_map.values():
            if len(members) >= self.min_cluster_size:
                for member in members:
                    labels[member] = cluster_id
                cluster_id += 1
        
        return labels
    
    def fit(self, X):
        """
        Perform HDBSCAN clustering
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        
        Returns:
        --------
        self
        """
        n_samples = X.shape[0]
        
        print(f"   🔄 Calcul des core distances (min_samples={self.min_samples})...")
        core_distances, distances = self._compute_core_distances(X)
        
        print(f"   🔄 Calcul de la mutual reachability...")
        mutual_reach = self._compute_mutual_reachability(distances, core_distances)
        
        print(f"   🔄 Construction du MST...")
        mst = self._build_mst(mutual_reach)
        
        print(f"   🔄 Construction de la hiérarchie...")
        edges = self._single_linkage_tree(mst)
        
        print(f"   🔄 Extraction des clusters (min_cluster_size={self.min_cluster_size})...")
        labels = self._extract_clusters(edges, n_samples)
        
        self.labels_ = labels
        
        # Compute simple probabilities (1.0 for cluster members, 0.0 for noise)
        self.probabilities_ = np.where(labels == -1, 0.0, 1.0)
        
        return self
    
    def fit_predict(self, X):
        """Perform clustering and return cluster labels"""
        self.fit(X)
        return self.labels_


# Version optimisée utilisant la bibliothèque hdbscan (recommandée)
try:
    import hdbscan
    
    class HDBSCANWrapper:
        """Wrapper pour utiliser la vraie implémentation HDBSCAN"""
        def __init__(self, min_cluster_size=5, min_samples=None):
            self.min_cluster_size = min_cluster_size
            self.min_samples = min_samples if min_samples is not None else min_cluster_size
            self.model = None
            self.labels_ = None
            self.probabilities_ = None
            
        def fit(self, X):
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples
            )
            self.labels_ = self.model.fit_predict(X)
            self.probabilities_ = self.model.probabilities_
            return self
        
        def fit_predict(self, X):
            self.fit(X)
            return self.labels_
    
    # Utiliser la vraie implémentation par défaut
    HDBSCANOptimized = HDBSCANWrapper
    
except ImportError:
    print("⚠️  Package hdbscan non installé. Utilisez: pip install hdbscan")
    HDBSCANOptimized = HDBSCANFromScratch
