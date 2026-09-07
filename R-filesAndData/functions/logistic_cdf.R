# function facilitating writing of logit function
logistic_cdf <- function(x) {
  return( 1/(1+exp(-x) ) )
}

