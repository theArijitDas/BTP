import numpy as np
from sklearn.decomposition import SparsePCA
from Utils.utils import seed_everything, one_hot_encode

class FTPLModel:
    def __init__(self, N: int, k: int, eta: float, feature_extraction: str="pca", sample_gamma_once: bool=True, seed: int=42):
        """
        Follow-The-Perturbed-Leader (FTPL) Model

        Parameters:
        N                    : Size of the universe
        k                    : Cache capacity (number of files to cache)
        eta                  : Learning rate
        feature_extraction   : Method to use for feature extraction ("pca" or "sparse_pca")
        sample_gamma_once    : Whether to sample \( \Gamma \) once or at every instant
        seed                 : Random seed for reproducibility
        """
        self.N = N
        self.k = k
        self.eta = eta
        self.sample_gamma_once = sample_gamma_once
        self.R = np.zeros((N, N))  # Reward matrix
        seed_everything(seed)

        feature_extraction_methods = {
            "pca": self._find_max_eigenvector,
            "sparse_pca": self._find_sparse_pca_vector
        }

        if feature_extraction not in feature_extraction_methods:
            raise ValueError("Invalid feature_extraction method. Choose either 'pca' or 'sparse_pca'.")
        
        self.feature_extraction_func = feature_extraction_methods[feature_extraction]
        
        if sample_gamma_once:
            self.Gamma = np.random.normal(0, 1, (N, N))  # Sample once

    def _find_max_eigenvector(self, R: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eig(R)
        max_index = np.argmax(eigenvalues)
        return eigenvectors[:, max_index]

    def _find_sparse_pca_vector(self, R: np.ndarray, n_components: int=1, alpha: float=1) -> np.ndarray:
        spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
        spca.fit(R)
        return spca.components_[0]
    
    def _top_k_binary_vector(self, v: np.ndarray, k: int) -> np.ndarray:
        """Returns a binary vector y, where the indices corresponding to the top-k elements of v are set to 1."""
        if k <= 0:
            return np.zeros_like(v, dtype=int)
        
        # Get indices of the top-k elements
        top_k_indices = np.argpartition(-v, k)[:k]  # argpartition for efficient selection
        
        # Create binary vector
        y = np.zeros_like(v, dtype=int)
        y[top_k_indices] = 1
        
        return y

    def _madow_sampling(self, v: np.ndarray) -> np.ndarray:
        """Madow's Sampling Scheme to obtain y with ||y||_1 = k from v"""
        v = v - np.max(v)  # Stabilize before applying softmax
        p = self.k  * np.exp(v) / np.sum(np.exp(v))  # Softmax normalization scaled by self.k
        Pi = np.cumsum(p)
        U = np.random.uniform(0, 1)
        S = set()
        for i in range(self.k):
            j = np.searchsorted(Pi, U + i)
            S.add(j)

        y = np.zeros(self.N)
        y[list(S)] = 1
        
        return y

    def _validate_l1_norm(self, y: np.ndarray) -> np.ndarray:
        """Ensure L1 norm of y is exactly k"""
        while np.sum(y) != self.k:
            diff = self.k - int(np.sum(y))
            if diff > 0:
                additional_indices = np.random.choice(np.where(y == 0)[0], size=diff, replace=False)
                y[additional_indices] = 1
            else:
                remove_indices = np.random.choice(np.where(y == 1)[0], size=-diff, replace=False)
                y[remove_indices] = 0
        return y

    def update(self, x_t1: int|np.ndarray, x_t2: int|np.ndarray):
        """Update reward matrix R"""
        if isinstance(x_t1, int): x_t1 = one_hot_encode(x_t1, self.N)
        if isinstance(x_t2, int): x_t2 = one_hot_encode(x_t2, self.N)

        self.R += np.outer(x_t1, x_t2) + np.outer(x_t2, x_t1) - np.diag(x_t1 * x_t2)

    def get_cache(self) -> np.ndarray:
        """Select the top k elements based on perturbed reward matrix"""
        if not self.sample_gamma_once:
            self.Gamma = np.random.normal(0, 1, (self.N, self.N))  # Resample every step
        perturbed_R = self.R + self.eta * self.Gamma
        
        v = self.feature_extraction_func(perturbed_R)
        y = self._madow_sampling(v)
        return self._validate_l1_norm(y)
    
    def __call__(self) -> np.ndarray:
        return self.get_cache()
    
    def get_best_offline_stationary(self) -> np.ndarray:
        v = self.feature_extraction_func(self.R)
        # y = self._madow_sampling(v)
        y = self._top_k_binary_vector(v, self.k)
        return self._validate_l1_norm(y)
