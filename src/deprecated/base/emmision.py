import equinox as eqx


class Emission(eqx.Module): 
    def __init__(self):
        pass 

    def step(self, xt = None):
        """
        xt is the covarites at time step t. 
        Returns the the emission parameters at time step t. 
        """
        raise ValueError("Emission step function not implemented. Please implement the step function to return the emission parameters at time step t given the covariates xt.") 
    
    def density(self, yt, xt = None):
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission density p(y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        raise ValueError("Emission density function not implemented. Please implement the density function to return the emission density p(y_t | z_t, x_t) at time step t given the observation y and covariates x.")  
    
    def cdf(self, yt, xt = None):
        """
        y is the observation at time step t. 
        x is the covariates at time step t. 
        Returns the emission cdf P(Y_t <= y_t | z_t, x_t) at time step t with dimensions (num_states,).
        """
        raise ValueError("Emission cdf function not implemented. Please implement the cdf function to return the emission cdf P(Y_t <= y_t | z_t, x_t) at time step t given the observation y and covariates x.")



    
