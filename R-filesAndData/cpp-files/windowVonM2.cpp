#include <TMB.hpp>     // Links in the TMB libraries

template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(y);        // Data vector from R
  DATA_VECTOR(n);        // Data vector from R
  DATA_VECTOR(h);        // Data vector from R
  DATA_VECTOR(lco2);     // Data vector from R
  DATA_FACTOR(dwel);     // Data vector from R

  PARAMETER_VECTOR(u1);  // Random effects
  PARAMETER_VECTOR(u2);  // Random effects
  // Parameters
  PARAMETER_VECTOR(beta);// Parameter value from R
  PARAMETER(sigma_1);    // Parameter value from R
  PARAMETER(lkappa);    // Parameter value from R
  
  
  int nobs = y.size();
  int nsub = u1.size();
  int j;
  Type mean_ran1 = Type(0);
  //  Type mean_ran2 = Type(0);
  Type p = Type(0);
  Type c1;
  Type c2;
  Type c3;
  Type c4;

  Type f = 0;      // Declare neg. log. likelihood

  c2 = log(2*M_PI) + log(besselI(exp(lkappa), Type(0)));
  for(int j=0; j < nsub; j++){
    f -= dnorm(u1[j], mean_ran1, exp(sigma_1), true);
    c1 = exp(lkappa) * cos(u2[j] - beta[3]);
    f -= c1 - c2;
  }
  
  for(int i =0; i < nobs; i++){
    j = dwel[i];
    c3 = beta[1] * cos(h[i] - u2[j]);
    c4 =  beta[0] + c3 + beta[2] * lco2[i] + u1[j];
    p  = invlogit(c4);
    f -= dbinom(y[i],n[i],p, true);
  }
  return f;
}
