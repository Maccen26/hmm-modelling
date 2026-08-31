from src.deprecated.base.hmm import HMM
import jax
import jax.numpy as jnp
import dataclasses
import equinox as eqx


class HMMParams:
    """Extracts and exposes the actual (untransformed) parameters from an HMM model."""

    def __init__(self, model: HMM):
        self.model_name = type(model).__name__
        if hasattr(model, 'emission'):
            self.emission_type = type(model.emission).__name__

            if hasattr(model.emission, 'mu'):
                self.mu = model.emission.mu
        
            if hasattr(model.emission, 'log_sigma'):
                self.sigma = jnp.exp(model.emission.log_sigma)

            if hasattr(model.emission, 'phi'):
                self.phi = model.emission.phi
        
        if hasattr(model, 'params'):
            self.params = model.params
        
        if hasattr(model, 'transition'):
            self.transition_logits = model.transition.transition_logits
            if hasattr(model.transition, 'beta'):
                self.beta = model.transition.beta
                
            else: 
                self.transition_matrix = model.transition.transition_matrix()

            if hasattr(model.transition, 'initial_state_dist'):
                self.initial_state_dist = model.transition.initial_state_dist
        






    def __repr__(self):
        lines = [f"model: {self.model_name}",
                 f"mu:    {self.mu}",
                 f"sigma: {self.sigma}"]
        
        if hasattr(self, 'phi'):
            lines.append(f"phi:   {self.phi}")
        if hasattr(self, 'beta'):
            lines.append(f"beta:  {self.beta}")
        if hasattr(self, 'initial_state_dist'):
            lines.append(f"initial_state_dist:  {self.initial_state_dist}")
        
        lines.append(f"transition_matrix:\n{self.transition_matrix}")
        return "\n".join(lines)





def get_trainable_params_by_class(model):
    """Return: {ClassName: [ {field_name: array, ...}, ... ]}"""
    out = {}

    def walk(node):
        if isinstance(node, eqx.Module):
            cls = type(node).__name__
            instance_params = {}

            for f in dataclasses.fields(node):
                v = getattr(node, f.name)

                if eqx.is_inexact_array(v):  # float/complex JAX arrays
                    instance_params[f.name] = v
                else:
                    walk(v)

            if instance_params:
                out.setdefault(cls, []).append(instance_params)

        elif isinstance(node, (list, tuple)):
            for x in node:
                walk(x)
        elif isinstance(node, dict):
            for x in node.values():
                walk(x)

    walk(model)
    return out


def flat_trainable_params(model):
    out = {}

    def walk(node):
        if isinstance(node, eqx.Module):
            for f in dataclasses.fields(node):
                v = getattr(node, f.name)

                if eqx.is_inexact_array(v):
                    key = f.name.lstrip("_")   # "_mu" -> "mu", "_log_sigma" -> "log_sigma"
                    out[key] = v
                elif isinstance(v, eqx.Module):
                    walk(v)
                elif isinstance(v, (list, tuple)):
                    for x in v:
                        walk(x)
                elif isinstance(v, dict):
                    for x in v.values():
                        walk(x)

    walk(model)
    return out


if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)

    from src.models.v1.stationary_hmm import StationaryHMM
    from src.models.v1.ar_hmm import ArHMM

    print("=== StationaryHMM ===")
    transition_logits = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    mu = jnp.array([500.0, 800.0, 1200.0])
    log_sigma = jnp.array([2.0, 3.0, 4.0, 5.0])

    stationary_model = StationaryHMM(transition_logits, mu, log_sigma)

    params = HMMParams(stationary_model)
    print(params)

    print("\n=== ArHMM ===")
    phi = jnp.array([0.9, 0.5, 0.7, 0.3])
    ar_model = ArHMM(transition_logits, mu, log_sigma, phi)
    ar_params = HMMParams(ar_model)
    print(ar_params)