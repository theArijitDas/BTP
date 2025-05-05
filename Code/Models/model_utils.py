from itertools import combinations
import numpy as np
from sklearn.decomposition import SparsePCA
from Utils.utils import seed_everything, one_hot_encode

class RewardMatrixHandler:
    def __init__(self, N: int):
        """
        Base class for managing and updating the reward matrix R.

        Parameters:
        N : int
            Size of the square matrix R (NxN).
        """
        self.N = N
        self.R = np.zeros((N, N))  # Reward matrix

    def update(self, x_t1: int | np.ndarray, x_t2: int | np.ndarray):
        """
        Update the reward matrix R based on observed interactions.

        Parameters:
        x_t1 : int | np.ndarray
            Either an integer (one-hot encoded internally) or a binary vector.
        x_t2 : int | np.ndarray
            Either an integer (one-hot encoded internally) or a binary vector.
        """
        if isinstance(x_t1, int):
            x_t1 = one_hot_encode(x_t1, self.N)
        if isinstance(x_t2, int):
            x_t2 = one_hot_encode(x_t2, self.N)

        self.R += np.outer(x_t1, x_t2) + np.outer(x_t2, x_t1) - np.diag(x_t1 * x_t2)


class ModifiedSPCA:
    def __init__(self, K=1.0, max_iter=100, tol=1e-5):
        """
        Parameters:
        - eta: regularization parameter (ηₜ)
        - K: scaling parameter (K)
        - s: thresholding parameter (s)
        - max_iter: max number of iterations
        - tol: tolerance for convergence
        """
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, R_perturbed):
        """
        R: matrix Rₜ (reward matrix)
        gamma: matrix gamma (same shape as R)
        
        Returns the real vector v_t after convergence.
        """
        # Step 1: Initialize A_t
        A_t = R_perturbed

        # Initialize v_t randomly
        v_t = np.random.rand(A_t.shape[0])
        v_t /= np.linalg.norm(v_t, 2)

        for iteration in range(self.max_iter):
            # Step 3: Compute u_t
            At_vt = A_t @ v_t
            norm_At_vt = np.linalg.norm(A_t @ v_t, 1)
            u_t = (self.K * At_vt) / norm_At_vt

            # Step 4: Update v_t with thresholding
            v_new = np.zeros_like(v_t)
            for i in range(len(u_t)):
                if u_t[i] > 1:
                    v_new[i] = 1
                elif u_t[i] <= 0:
                    v_new[i] = 0
                else:
                    v_new[i] = u_t[i]

            # Check for convergence
            if np.linalg.norm(v_new - v_t, 2) < self.tol:
                break

            v_t = v_new

        return v_t


class FTPLModel:
    def __init__(self, N: int, k: int, eta: float,
                 feature_extraction: str="modified_spca", sampling: str="madow",
                 sample_gamma_once: bool=True, seed: int=42):
        """
        Follow-The-Perturbed-Leader (FTPL) Model

        Parameters:
        N                    : Size of the universe
        k                    : Cache capacity (number of files to cache)
        eta                  : Learning rate
        feature_extraction   : Method to use for feature extraction ("pca" or "sparse_pca" or "modified_spca")
        sample_gamma_once    : Whether to sample gamma once or at every instant
        seed                 : Random seed for reproducibility
        """
        self.N = N
        self.k = k
        self.eta = eta
        self.sample_gamma_once = sample_gamma_once
        seed_everything(seed)

        feature_extraction_methods = {
            "pca": self._find_max_eigenvector,
            "sparse_pca": self._find_sparse_pca_vector,
            "modified_spca": self._find_modified_spca_vector,
        }

        sampling_methods = {
            "madow": self._madow_sampling,
            "top_k": self._top_k_binary_vector,
        }

        softmax = {
            "pca": True,
            "sparse_pca": True,
            "modified_spca": False,
        }

        if feature_extraction not in feature_extraction_methods:
            raise ValueError("Invalid feature_extraction method. Choose either 'pca', 'sparse_pca', or 'modified_spca'.")
        
        if sampling not in sampling_methods:
            raise ValueError("Invalid sampling method. Choose either 'madow' or 'top_k'.")
        
        self.feature_extraction_func = feature_extraction_methods[feature_extraction]
        self.sampling_func = sampling_methods[sampling]
        self.softmax = softmax[feature_extraction]
        
        if sample_gamma_once:
            self.Gamma = np.random.normal(0, 1, (N, N))  # Sample once

    #=========== Methods to get real valued inclusion probability vector ==========#
    def _find_max_eigenvector(self, R: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eig(R)
        max_index = np.argmax(eigenvalues)
        return eigenvectors[:, max_index]

    def _find_sparse_pca_vector(self, R: np.ndarray, n_components: int=1, alpha: float=1) -> np.ndarray:
        spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
        spca.fit(R)
        return spca.components_[0]
    
    def _find_modified_spca_vector(self, R: np.ndarray):
        modified_spca = ModifiedSPCA()
        return_ = modified_spca.fit(R)
        return return_
    
    #=========== Methods to sample cache vector from inclusion probability vector ==========#
    def _top_k_binary_vector(self, v: np.ndarray, k: int = None) -> np.ndarray:
        """Returns a binary vector y, where the indices corresponding to the top-k elements of v are set to 1."""
        k = self.k if k is None else k

        if k <= 0:
            return np.zeros_like(v, dtype=int)
        
        # Get indices of the top-k elements
        top_k_indices = np.argpartition(-v, k)[:k]  # argpartition for efficient selection
        
        # Create binary vector
        y = np.zeros_like(v, dtype=int)
        y[top_k_indices] = 1
        assert np.sum(y) == self.k, f"L1 norm of 'y' must be {self.k}"
        return y

    def _madow_sampling(self, v: np.ndarray) -> np.ndarray:
        """Madow's Sampling Scheme to obtain y with ||y||_1 = k from v"""
        if self.softmax:
            v = v - np.max(v)  # Stabilize before applying softmax
            p = self.k  * np.exp(v) / np.sum(np.exp(v))  # Softmax normalization scaled by self.k
        else:
            p = self.k * v / np.sum(v)
        Pi = np.cumsum(p)
        U = np.random.uniform(0, 1)
        S = set()
        for i in range(self.k):
            j = np.searchsorted(Pi, U + i)
            S.add(j)

        y = np.zeros(self.N)
        y[list(S)] = 1
        return y

    #============= Method to ensure k items are selected for cache =============#
    def _validate_l1_norm(self, y: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Ensure L1 norm of y is exactly k, guided by probability vector v."""
        while np.sum(y) != self.k:
            diff = self.k - int(np.sum(y))

            if diff > 0:
                # Need to add items: pick top `diff` indices from v where y == 0
                candidate_indices = np.where(y == 0)[0]
                candidate_probs = v[candidate_indices]
                top_indices = candidate_indices[np.argsort(-candidate_probs)[:diff]]
                y[top_indices] = 1

            else:
                # Need to remove items: pick bottom `-diff` indices from v where y == 1
                candidate_indices = np.where(y == 1)[0]
                candidate_probs = v[candidate_indices]
                bottom_indices = candidate_indices[np.argsort(candidate_probs)[: -diff]]
                y[bottom_indices] = 0
        assert np.sum(y) == self.k, f"L1 norm of 'y' must be {self.k}"
        return y


    def get_cache(self,R) -> np.ndarray:
        """Select the top k elements based on perturbed reward matrix"""
        if not self.sample_gamma_once:
            self.Gamma = np.random.normal(0, 1, (self.N, self.N))  # Resample every step
        perturbed_R = R + self.eta * self.Gamma
        
        v = self.feature_extraction_func(perturbed_R)
        y = self.sampling_func(v)
        return self._validate_l1_norm(y, v)
    
    def __call__(self, R) -> np.ndarray:
        return self.get_cache(R)


class BestStationaryOptimal:
    def __init__(self, N, k):
        """
        Initialize the optimizer with the matrix size (N) and number of ones (K) in Y.

        Parameters:
        N (int): Size of the square matrix R (NxN).
        K (int): Number of ones in the binary vector Y.
        """
        if k > N or k < 0:
            raise ValueError("K must be between 0 and N (inclusive).")
        
        self.N = N
        self.K = k

    def get_best_Y(self, R):
        """
        Given an NxN matrix R, find the binary vector Y that maximizes Y' R Y.

        Parameters:
        R (numpy.ndarray): The input square matrix of shape (N, N).

        Returns:
        numpy.ndarray: The optimal binary vector Y of shape (N,).
        """
        max_value = float('-inf')
        best_Y = None

        # Iterate over all subsets of size K
        for indices in combinations(range(self.N), self.K):
            Y = np.zeros(self.N)
            Y[list(indices)] = 1

            # Compute Y'RY
            value = Y @ R @ Y

            # Update max_value and best_Y
            if value > max_value:
                max_value = value
                best_Y = Y.copy()

        return best_Y
    
    def __call__(self, R):
        return self.get_best_Y(R)
