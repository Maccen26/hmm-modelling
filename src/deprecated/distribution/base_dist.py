class BaseDistribution:
    def __init__(self, **kwargs):
        pass 
    
    def density(self, x):
        raise NotImplementedError("The density method must be implemented by subclasses.") 
    