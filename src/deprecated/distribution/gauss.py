from src.deprecated.distribution.base_dist import BaseDistribution
from src.deprecated.distribution.utils import _gaussian_pdf 
import jax.numpy as jnp
import jax.scipy.stats.norm as norm 
import jax



class Gaussian(BaseDistribution): 
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def density(self, x): 
        """Compute Gaussian density for numerical stability."""
        return _gaussian_pdf(x, self.mean, self.std)
     



    

if __name__ == "__main__":
        # Example usage
    
    gaussian = Gaussian(mean=0, std=1)
    x = jnp.arange(0, 100, 0.1)  # Or jnp.linspace(0, 5, 50)

    print("Density:", gaussian.density(x))

