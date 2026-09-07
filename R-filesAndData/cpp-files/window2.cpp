#include <TMB.hpp>     // Links in the TMB libraries

template<class Type>
Type objective_function<Type>::operator() ()
{
  // Data
  DATA_MATRIX(Y);         // Observation matrix from R
  DATA_MATRIX(X);         // Regression matrix from R
  DATA_FACTOR(dwel);      // Data vector from R

  // Random Effects Parameters
  PARAMETER_MATRIX(u);    

  // Fixed Effects Parameters
  PARAMETER_VECTOR(pars);  
  
  int nobs = Y.col(0).size(), nsub = u.col(0).size();
  int j;
  Type mean_u1 = Type(0); 
  Type mean_u2 = Type(0);
  Type p = Type(0);
  Type c;
  Type c2;

  Type f = 0;      // Declare neg. log. likelihood
  for(int j=0; j < nsub; j++){
    f -= dnorm(u(j,0), mean_u1 , exp(pars(0)), true);
    f -= dnorm(u(j,1), mean_u2 , exp(pars(1)), true);
  }
  
  for(int i =0; i < nobs; i++){
    j  = dwel[i];
    c  = cos(X(i,0) - pars(5) - u(j,1));
    c2  = pars(2) + pars(3) * c + pars(4) * X(i,1);
    p  = invlogit(c2 + u(j,0));
    f -= dbinom(Y(i,0),Y(i,1),p, true);
  }
  return f;
}
