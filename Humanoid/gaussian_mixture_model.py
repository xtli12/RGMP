import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
import pickle
import os

class GaussianMixtureModel:
    """
    Gaussian Mixture Model for robotic action optimization
    
    Addresses the issue where purely visual training leads to network collapse 
    towards the mean of action distribution. GMM enables modeling of separate 
    action clusters with distinct means and covariances for accurate action representation.
    """
    
    def __init__(self, n_components=6, n_features=6, covariance_type='full'):
        """
        Initialize GMM with K=6 components for 6-DOF robotic arm
        
        Args:
            n_components: Number of Gaussian components (K=6)
            n_features: Dimension of joint space (6 for 6-DOF arm)
            covariance_type: Type of covariance parameters
        """
        self.K = n_components  # K=6 components
        self.n_features = n_features  # 6-DOF joint space
        self.covariance_type = covariance_type
        
        # GMM parameters: {α_k, μ_k, Σ_k}
        self.priors = None  # α_k: prior probabilities
        self.means = None   # μ_k: cluster means
        self.covariances = None  # Σ_k: covariance matrices
        
        # Sklearn GMM for parameter estimation via EM algorithm
        self.gmm_model = GaussianMixture(
            n_components=self.K,
            covariance_type=self.covariance_type,
            random_state=42,
            max_iter=200,
            tol=1e-6
        )
        
        self.is_fitted = False
        
    def fit(self, joint_configurations):
        """
        Fit GMM parameters using EM algorithm to maximize data likelihood
        
        Args:
            joint_configurations: Ground-truth joint configurations (N, 6)
                                 where N is number of demonstrations
        
        Returns:
            self: Fitted GMM model
        """
        if isinstance(joint_configurations, torch.Tensor):
            joint_configurations = joint_configurations.cpu().numpy()
        
        # Ensure input shape is (N, 6)
        if joint_configurations.ndim == 1:
            joint_configurations = joint_configurations.reshape(1, -1)
        
        print(f"Fitting GMM with {len(joint_configurations)} demonstrations")
        print(f"Joint configuration shape: {joint_configurations.shape}")
        
        # Fit GMM using EM algorithm
        self.gmm_model.fit(joint_configurations)
        
        # Extract fitted parameters
        self.priors = self.gmm_model.weights_  # α_k
        self.means = self.gmm_model.means_     # μ_k
        self.covariances = self.gmm_model.covariances_  # Σ_k
        
        self.is_fitted = True
        
        print(f"GMM fitted successfully:")
        print(f"  - Priors (α_k): {self.priors}")
        print(f"  - Means shape: {self.means.shape}")
        print(f"  - Covariances shape: {self.covariances.shape}")
        print(f"  - Log likelihood: {self.gmm_model.score(joint_configurations):.4f}")
        
        return self
    
    def predict_action(self, initial_prediction):
        """
        Predict final action using GMM optimization
        
        Process:
        1. Compare initial prediction a_in to GMM clusters using Mahalanobis distance
        2. Select closest cluster center as final action a*
        
        Args:
            initial_prediction: Initial action prediction from ARGN (6,)
        
        Returns:
            final_action: Optimized action a* (6,)
            cluster_id: ID of selected cluster
            distances: Mahalanobis distances to all clusters
        """
        if not self.is_fitted:
            raise ValueError("GMM must be fitted before prediction")
        
        if isinstance(initial_prediction, torch.Tensor):
            initial_prediction = initial_prediction.cpu().numpy()
        
        # Ensure input is 1D array of length 6
        if initial_prediction.ndim > 1:
            initial_prediction = initial_prediction.flatten()
        
        assert len(initial_prediction) == self.n_features, \
            f"Expected {self.n_features} features, got {len(initial_prediction)}"
        
        # Calculate Mahalanobis distance to each cluster
        distances = []
        
        for k in range(self.K):
            mean_k = self.means[k]  # μ_k
            cov_k = self.covariances[k]  # Σ_k
            
            # Mahalanobis distance: l_k = sqrt((a_in - μ_k)^T * Σ_k^(-1) * (a_in - μ_k))
            try:
                # Compute inverse of covariance matrix
                cov_inv = np.linalg.inv(cov_k)
                
                # Calculate Mahalanobis distance
                diff = initial_prediction - mean_k
                distance = np.sqrt(diff.T @ cov_inv @ diff)
                distances.append(distance)
                
            except np.linalg.LinAlgError:
                # Handle singular covariance matrix
                print(f"Warning: Singular covariance matrix for cluster {k}")
                # Use regularized covariance
                reg_cov = cov_k + np.eye(self.n_features) * 1e-6
                cov_inv = np.linalg.inv(reg_cov)
                diff = initial_prediction - mean_k
                distance = np.sqrt(diff.T @ cov_inv @ diff)
                distances.append(distance)
        
        distances = np.array(distances)
        
        # Select cluster with minimum Mahalanobis distance
        # a* = arg min_{μ_k} l_k
        closest_cluster = np.argmin(distances)
        final_action = self.means[closest_cluster].copy()
        
        return final_action, closest_cluster, distances
    
    def sample_action(self, n_samples=1):
        """
        Sample actions from the fitted GMM distribution
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            samples: Generated action samples (n_samples, 6)
        """
        if not self.is_fitted:
            raise ValueError("GMM must be fitted before sampling")
        
        samples, _ = self.gmm_model.sample(n_samples)
        return samples
    
    def score_samples(self, actions):
        """
        Compute log-likelihood of actions under the GMM
        
        Args:
            actions: Action samples to score (N, 6)
            
        Returns:
            log_likelihood: Log-likelihood scores (N,)
        """
        if not self.is_fitted:
            raise ValueError("GMM must be fitted before scoring")
        
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        return self.gmm_model.score_samples(actions)
    
    def get_cluster_info(self):
        """
        Get detailed information about fitted clusters
        
        Returns:
            cluster_info: Dictionary with cluster statistics
        """
        if not self.is_fitted:
            return None
        
        cluster_info = {
            'n_components': self.K,
            'priors': self.priors.tolist(),
            'means': self.means.tolist(),
            'covariances': self.covariances.tolist(),
            'converged': self.gmm_model.converged_,
            'n_iter': self.gmm_model.n_iter_,
            'lower_bound': self.gmm_model.lower_bound_
        }
        
        return cluster_info
    
    def save_model(self, filepath):
        """Save fitted GMM model to file"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'K': self.K,
            'n_features': self.n_features,
            'covariance_type': self.covariance_type,
            'priors': self.priors,
            'means': self.means,
            'covariances': self.covariances,
            'gmm_model': self.gmm_model,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"GMM model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load fitted GMM model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.K = model_data['K']
        self.n_features = model_data['n_features']
        self.covariance_type = model_data['covariance_type']
        self.priors = model_data['priors']
        self.means = model_data['means']
        self.covariances = model_data['covariances']
        self.gmm_model = model_data['gmm_model']
        self.is_fitted = model_data['is_fitted']
        
        print(f"GMM model loaded from {filepath}")
        return self

class GMMActionOptimizer:
    """
    Action optimizer that integrates GMM with ARGN predictions
    
    This class handles the complete pipeline from initial ARGN prediction
    to final optimized action using GMM clustering.
    """
    
    def __init__(self, gmm_model_path=None):
        """
        Initialize GMM action optimizer
        
        Args:
            gmm_model_path: Path to pre-trained GMM model (optional)
        """
        self.gmm = GaussianMixtureModel()
        
        if gmm_model_path and os.path.exists(gmm_model_path):
            self.gmm.load_model(gmm_model_path)
            print("Loaded pre-trained GMM model")
        else:
            print("No pre-trained GMM model found, will need to train")
    
    def train_gmm(self, demonstration_data):
        """
        Train GMM on human demonstration data
        
        Args:
            demonstration_data: List of joint configurations from demonstrations
                               Each element should be (6,) array for 6-DOF arm
        """
        # Convert demonstration data to numpy array
        if isinstance(demonstration_data, list):
            joint_configs = np.array(demonstration_data)
        elif isinstance(demonstration_data, torch.Tensor):
            joint_configs = demonstration_data.cpu().numpy()
        else:
            joint_configs = demonstration_data
        
        # Ensure proper shape (N, 6)
        if joint_configs.ndim == 1:
            joint_configs = joint_configs.reshape(1, -1)
        
        print(f"Training GMM with {len(joint_configs)} demonstrations")
        
        # Fit GMM using EM algorithm
        self.gmm.fit(joint_configs)
        
        return self.gmm
    
    def optimize_action(self, initial_prediction, return_details=False):
        """
        Optimize initial ARGN prediction using GMM
        
        Args:
            initial_prediction: Initial action from ARGN network
            return_details: Whether to return detailed optimization info
            
        Returns:
            optimized_action: Final optimized action
            details: Optimization details (if return_details=True)
        """
        if not self.gmm.is_fitted:
            raise ValueError("GMM must be trained before optimization")
        
        # Get optimized action from GMM
        final_action, cluster_id, distances = self.gmm.predict_action(initial_prediction)
        
        if return_details:
            details = {
                'initial_prediction': initial_prediction,
                'final_action': final_action,
                'selected_cluster': cluster_id,
                'mahalanobis_distances': distances,
                'cluster_means': self.gmm.means,
                'improvement': np.linalg.norm(final_action - initial_prediction)
            }
            return final_action, details
        
        return final_action
    
    def save_gmm(self, filepath):
        """Save trained GMM model"""
        self.gmm.save_model(filepath)
    
    def load_gmm(self, filepath):
        """Load pre-trained GMM model"""
        self.gmm.load_model(filepath)

def create_demonstration_dataset(joint_trajectories):
    """
    Create demonstration dataset for GMM training
    
    Args:
        joint_trajectories: List of joint angle trajectories from human demonstrations
                           Each trajectory is (T, 6) where T is trajectory length
    
    Returns:
        demonstration_data: Flattened joint configurations for GMM training
    """
    demonstration_data = []
    
    for trajectory in joint_trajectories:
        if isinstance(trajectory, torch.Tensor):
            trajectory = trajectory.cpu().numpy()
        
        # Add all joint configurations from trajectory
        for joint_config in trajectory:
            demonstration_data.append(joint_config)
    
    return np.array(demonstration_data)

# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic demonstration data for testing
    np.random.seed(42)
    
    # Simulate 6 different manipulation skills with distinct joint configurations
    skill_centers = [
        [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],    # Skill 1: Home position
        [1.57, -1.0, 0.5, -2.0, 1.0, 0.5],     # Skill 2: Side grasp
        [-1.57, -0.5, -0.5, -1.0, -1.0, -0.5], # Skill 3: Lift up
        [0.5, -2.0, 1.0, -0.5, 0.5, 1.0],      # Skill 4: Top pinch
        [-0.5, -1.5, -1.0, -1.5, -0.5, -1.0],  # Skill 5: Push
        [1.0, -0.8, 0.8, -1.8, 1.2, 0.3]       # Skill 6: Pull
    ]
    
    # Generate demonstration data around each skill center
    demonstration_data = []
    for center in skill_centers:
        # Generate 50 demonstrations per skill with small variations
        for _ in range(50):
            noise = np.random.normal(0, 0.1, 6)  # Small Gaussian noise
            demo = np.array(center) + noise
            demonstration_data.append(demo)
    
    demonstration_data = np.array(demonstration_data)
    print(f"Generated {len(demonstration_data)} demonstration samples")
    
    # Test GMM training and optimization
    print("\n=== Testing GMM Action Optimizer ===")
    
    # Initialize optimizer
    optimizer = GMMActionOptimizer()
    
    # Train GMM
    optimizer.train_gmm(demonstration_data)
    
    # Test action optimization
    print("\n=== Testing Action Optimization ===")
    
    # Simulate initial ARGN predictions (slightly noisy)
    test_predictions = [
        [0.1, -1.6, 0.05, -1.55, 0.02, 0.01],  # Close to skill 1
        [1.6, -0.95, 0.52, -1.98, 1.02, 0.48], # Close to skill 2
        [-1.55, -0.48, -0.52, -0.98, -1.02, -0.48] # Close to skill 3
    ]
    
    for i, initial_pred in enumerate(test_predictions):
        print(f"\nTest {i+1}:")
        print(f"Initial prediction: {initial_pred}")
        
        optimized_action, details = optimizer.optimize_action(
            initial_pred, return_details=True
        )
        
        print(f"Optimized action: {optimized_action}")
        print(f"Selected cluster: {details['selected_cluster']}")
        print(f"Improvement (L2 norm): {details['improvement']:.4f}")
        print(f"Min Mahalanobis distance: {np.min(details['mahalanobis_distances']):.4f}")
    
    # Save trained model
    optimizer.save_gmm('gmm_model.pkl')
    print("\nGMM model saved successfully")
    
    # Test model loading
    new_optimizer = GMMActionOptimizer('gmm_model.pkl')
    test_action = new_optimizer.optimize_action([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    print(f"Loaded model test result: {test_action}")