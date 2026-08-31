from src.api.v4.hmm_models.results import HMMResults, StateResults
import jax.numpy as jnp
import jax.scipy.stats as jstats

def AIC(hmm_results: HMMResults) -> float:
    """Akaike Information Criterion for model selection."""
    return 2 * hmm_results.num_params - 2 * hmm_results.log_likelihood

def BIC(hmm_results: HMMResults, num_data_points: int) -> float:
    """Bayesian Information Criterion for model selection."""
    return float(hmm_results.num_params * jnp.log(num_data_points) - 2 * hmm_results.log_likelihood)

def LRT(base_model: HMMResults, expanded_model: HMMResults) -> tuple[float, float]:
    """Likelihood Ratio Test p-value comparing a base model to a nested expanded model.

    The test statistic -2 * (LL_base - LL_expanded) is chi-squared distributed
    with degrees of freedom equal to the difference in number of parameters.
    """
    test_statistic = 2 * (expanded_model.log_likelihood - base_model.log_likelihood)
    df = expanded_model.num_params - base_model.num_params
    
    p_value = 1.0 - jstats.chi2.cdf(test_statistic, df=df)
    return float(test_statistic), float(p_value)