
from dataclasses import dataclass


@dataclass(frozen=True)
class MODELTYPES: 
    StationaryHMM = "stationary_hmm_4_state" 
    AR1 = "ar1_model" 
    AR2 = "ar2_model" 
    AR_HMM = "ar_hmm_model"
    COVARIATE_HMM = "covariate_hmm_model"
    COVARIATE_AR_HMM = "covariate_ar_hmm_model"  
    AR_HMM_UNCONSTRAINED = "ar_hmm_unconstrained-v2"
    AR_HMM_UNCONSTRAINED_PRIOR = "ar_hmm_unconstrained-v2" 
    AR_HMM_v2 = "ar_hmm_v2" 

    