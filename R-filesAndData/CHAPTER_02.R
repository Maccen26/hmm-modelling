##################################################
## Example 2.1

dbinom(1, size = 2, prob = c(0.25, 0.5, 0.6))

##################################################
## Section 2.1
##################################################

##################################################
## Section 2.2
##################################################

##################################################
## Example 2.2

library(MASS)
par(bg="white")
y<-c(0.63,0.58,4.71,0.58,115.75,0.71,0.04,0.67,0.58,115.75,0.71,0.04,14.13,14.08,27.04,4.92,36.83,7,1.63,28,6.92,1.71,1.75,6.88,0.63,0.58,0.54,0.54,0.5,0.5,0.46,0.63,0.58,0.46,0.54,6.88,3.04,3,18.83,18.79,5.75,0.67,2.13,2.08,8.75,26.5,1.83,6.88,13.83,13.79,18.83,18.79,5.75,0.67,2.13,2.08,8.75,26.5,1.83,6.88,2.83,0.71,0.04,0.88,0.83,0.79,4.71,4.67,4.63,0.92,0.88,6.88)
hist(y,breaks=15,xlab="Time (days)")

c(mean(y),sd(y))

par(bg="white")

theta <- seq(5,20,length=300)
n <- length(y)
ll <- theta^(-n)*exp(-n/theta*mean(y))
ll <- ll/max(ll)
plot(theta,ll,type="l",ylab="Normalized likelihood",xlab=expression(theta))

ll2.fun <- function(theta,y){sum(dchisq(y,df=theta,log=TRUE))}
opt <- optimize(ll2.fun,c(0,20),y=y,maximum=TRUE)

sum(dexp(y, rate = 1/9, log = TRUE))

sum(dchisq(y,df=3.4,log=TRUE))

##################################################
## Example 2.3

options(digits=3)
n <- length(y)
theta.hat <- mean(y)
ll.m <- - n * log(theta.hat) - n / theta.hat * mean(y)
c(theta.hat, ll.m)

##################################################
## Section 2.3
##################################################

##################################################
## Example 2.4

##################################################
## Section 2.4
##################################################

##################################################
## Example 2.5

##################################################
## Example 2.6

theta0 <- 12
(ll0 <- - n * log(theta0) - n / theta0 * mean(y))

Q <- (ll.m - ll0)
1 - pchisq(2 * Q, df = 1)

exp(-0.5*qchisq(0.95,df=1))

par(bg="white")
plot(theta,ll,type="l",xlim=c(6,13),ylab="Normalised Likelihood",
     xlab=expression(theta))
lines(range(theta),
exp(-0.5*qchisq(0.95,df=1))*c(1,1),col=1,lty=2,lwd=2)

## Negative log-likelihood function
nll.wei <- function(theta, y){
    - sum(dweibull(y, shape = theta[1], scale = theta[2],
                   log = TRUE))}

## Optimize parameters
fit.wei <- nlminb(c(1, 1), nll.wei, lower = 0.01, y = y)
## Parameter estimates
fit.wei$par
## negative log-likelihood
fit.wei$objective

## test statistics
Q <- -fit.wei$objective-ll.m
## p-value
1 - pchisq(2 * Q, df = 1)

##################################################
## Example 2.7

(j <- length(y) / mean(y) ^ 2)

##################################################
## Example 2.8

## Information
library(numDeriv)
H <- hessian(nll.wei, fit.wei$par, y = y)

se <- sqrt(diag(solve(H)))
tab <- cbind(fit.wei$par, fit.wei$par + qnorm(0.025) * se,
             fit.wei$par + qnorm(0.975) * se)
rownames(tab) <- c("shape", "scale")
colnames(tab) <- c("theta", "lowerb", "upperb")
round(tab, digits=2)

##################################################
## Section 2.5
##################################################

##################################################
## Example 2.9

pll.wei.gam <- function(gamma,y){
    ## function for inner optiisation
    fun.tmp <- function(lambda,gamma,y){
        theta <- c(gamma, lambda)
        nll.wei(theta, y)
        }
    ## inner optimazation
    optimize(fun.tmp, c(0.001,100), gamma = gamma, 
             y = y)$objective
}

## Profile likelihood over gamma
gamma <- seq(0.4,0.9,length=100)
pll.gam <- sapply(gamma,pll.wei.gam,y=y)


pll.wei.lam <- function(lambda,y){
    fun.tmp <- function(gamma,lambda,y){
        theta <- c(gamma, lambda)
        nll.wei(theta, y)
        }
    optimize(fun.tmp, c(0.001,100), lambda = lambda, 
             y = y)$objective
}

lambda <- seq(2,11,length=100)
pll.lam <- sapply(lambda,pll.wei.lam,y=y)

par(mfrow=c(1,2),bg="white")
plot(gamma,exp(-pll.gam+fit.wei$objective),type="l",ylab="Profile likelihood",xlab=expression(gamma))
lines(range(gamma),exp(-0.5*qchisq(0.95,df=1))*c(1,1),
      col=1,lwd=2,lty=2)
rug(tab[1,2:3],lwd=3,col=1)
plot(lambda,exp(-pll.lam+fit.wei$objective),type="l",ylab="Profile likelihood",xlab=expression(lambda))
lines(range(lambda),exp(-0.5*qchisq(0.95,df=1))*c(1,1),
      col=1,lwd=2,lty=2)
rug(tab[2,2:3],lwd=3,col=1)

##################################################
## Section 2.6
##################################################

##################################################
## Example 2.10

pll.wei.lam <- function(lambda,y){
    fun.tmp <- function(gamma,lambda,y){
        theta <- c(gamma, lambda)
        nll.wei(theta, y)
        }
    optimize(fun.tmp, c(0.001,100), lambda = lambda, 
             y = y)$objective
}


lambda <- seq(3.35,9,length=100)
pll.lam <- sapply(lambda,pll.wei.lam,y=y)

par(mfrow=c(1,2),bg="white")
plot(lambda,-pll.lam+fit.wei$objective,type="l",ylab="Profile log-likelihood",xlab=expression(lambda))
lines(range(lambda),-0.5*qchisq(0.95,df=1)*c(1,1),
      col=1,lwd=2,lty=2)
lines(lambda, -0.5*H[2,2]*(lambda-fit.wei$par[2])^2,
      lty=2)

plot(log(lambda),-pll.lam+fit.wei$objective,type="l",ylab="Profile log-likelihood",xlab=expression(log(lambda)))

nll.wei.log <- function(theta,y){
    -sum(dweibull(y,shape=exp(theta[1]),
                  scale=exp(theta[2]),
                  log = TRUE))}
fit.wei.log <- nlminb(c(0,0),nll.wei.log,y=y)
H.log <- hessian(nll.wei.log, fit.wei.log$par,y=y)
lines(log(lambda), -0.5*H.log[2,2]*(log(lambda)-fit.wei.log$par[2])^2,
      lty=2)
lines(range(log(lambda)),-0.5*qchisq(0.95,df=1)*c(1,1),
      col=1,lwd=2,lty=2)

##################################################
## Section 2.7
##################################################

##################################################
## Example 2.11

nll.loglogis <- function(theta,y){
   - sum(dlogis(y, location = theta[1], scale = theta[2],
                log=TRUE))}
fit.logis <- nlminb(c(0,1), nll.loglogis, y = log(y),
                    lower= c(-Inf, 0))
c(fit.logis$par,-fit.logis$objective)

c(-fit.wei$objective, ll.m)

(ll.logis <- - fit.logis$objective + sum(log(1/y)))

##################################################
## Section 2.8
##################################################

##################################################
## Example 2.12

AIC <- c(Wei = 2 * fit.wei$objective + 2 * 2, 
         logis=-2 * ll.logis + 2 * 2)
BIC <- c(Wei = 2 * fit.wei$objective + log(length(y)) * 2, 
         logis = -2 * ll.logis + log(length(y)) * 2)
cbind(AIC,BIC)

##################################################
## Figure 2.1
par(mfrow=c(1,1))
col.tmp <- gray(c(0.25,0.5,0.75))
plot(ecdf(y))
x <- seq(0,max(y),length=200)
lines(x, pexp(x, 1/mean(y)),col=col.tmp[1],lwd=3,lty=1)
lines(x, pweibull(x, shape=fit.wei$par[1], scale=fit.wei$par[2] ),col=col.tmp[2],lwd=3,lty=1)
lines(x, plogis(log(x), location=fit.logis$par[1], scale=fit.logis$par[2] ),col=col.tmp[3],lwd=3,lty=1)
lines(ecdf(y))
legend("bottomright",legend=c("Exponential","Weibull","log-logistic"),col=col.tmp,lty=1,lwd=3)

##################################################
## Section 2.9
##################################################

##################################################
## Figure 2.2
source("functions/flowChartDatNew.R")

##################################################
## Section 2.10
##################################################
