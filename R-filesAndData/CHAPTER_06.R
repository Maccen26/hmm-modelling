library(numDeriv)
library(glmmTMB)
library(ggplot2)
source("functions/logistic_cdf.R")

## from Chapter 4
dfcount <- read.csv(file = "data/clothing_count.csv",sep=";") 
dfcount$delta.t=dfcount$time/dfcount$nobs
fit.binom <- glm(cbind(clo, nobs - clo) ~ sex, data = dfcount, 
                 family = binomial)

fit.poisOff <- glm(clo ~ sex + offset(log(time)), data = dfcount, 
                   family = poisson)

## From Chapter 5
dfMEMEx1 <- read.csv(file = "data/clothing.csv")
 dfMEMEx1$subjId <- factor(dfMEMEx1$subjId)
dfMEMEx1$tOut <- dfMEMEx1$tOut-mean(dfMEMEx1$tOut)
  mtOut <- mean(dfMEMEx1$tOut)
  mtInOp <- mean(dfMEMEx1$tInOp)

  dfMEMEx1$tOutCen <- dfMEMEx1$tOut - mtOut
  dfMEMEx1$tInOpCen <- dfMEMEx1$tInOp - mtInOp
  dfMEMEx1$sexNew <- suppressWarnings(as.numeric("NA"))
  
  dfMEMEx1$sexNew[dfMEMEx1$sex=="female"] <- 1
  dfMEMEx1$sexNew[dfMEMEx1$sex=="male"] <- 2
  dfMEMEx1$sex <- as.numeric(dfMEMEx1$sexNew)


  ## insert obs. number within day
  lev.subjId <- as.numeric(levels(as.factor(dfMEMEx1$subjId)))
  for(i in 1:length(lev.subjId)){
      lev.day <- as.numeric(levels(as.factor(dfMEMEx1$day[dfMEMEx1$subjId==lev.subjId[i]])))
      for(j in 1:length(lev.day)){
          I <- dfMEMEx1$subjId==lev.subjId[i] & dfMEMEx1$day==lev.day[j]
         dfMEMEx1$obs.no[I] <- 1:sum(I)
      }
  }
  

modMEMEx18AR1 <- glmmTMB(clo ~ tOutCen + sex + (1 | subjId) + 
                             ar1(as.factor(obs.no) - 1 | 
                                 subjId:day), data = dfMEMEx1,
                         REML = FALSE)

##################################################
## Example 6.1

##################################################
## Example 6.2

##################################################
## Example 6.3

##################################################
## Example 6.4

##################################################
## Section 6.1
##################################################

##################################################
## Example 6.5

##################################################
## Example 6.6

##################################################
## Example 6.7

##################################################
## Example 6.8

nll.u <- function(u, eta, y, n, sigma){
    p <- exp(eta + u)/(1 + exp(eta + u))
    - sum(dbinom(y, size = n, prob = p, log =TRUE)) -
          dnorm(u, mean=0, sd = sigma, log = TRUE)
}

##################################################
## Example 6.9


## Laplace app. negative log-likelihood
nll.la <- function(theta, X, y, n, subj.lev, subjId, full = FALSE){
    beta <- theta[1:dim(X)[2]] ## Fixed effect parameters
    ## Random effect parameters
    sigma <- exp(theta[-(1:dim(X)[2])]/2) 
    eta <- as.numeric(X %*% beta) ## Linear predictor
    nll <- 0 ## Initalize negative marginal log-likelihood
    u <- numeric(length(subj.lev))
    H <- u
    ## For loop over random effects
    for(i in 1:length(subj.lev)){
        I <- subjId == subj.lev[i] ## Choose subject
        ## Optimise random effect for given parameters
        opt <- nlminb(0, nll.u, eta = eta[I], y = y[I], 
                      n = n[I], sigma = sigma)
        ## Extract random effect and Hessian 
        ## of random effect likelihood
        u[i] <- opt$par
        H[i] <- hessian(nll.u, u[i], eta = eta[I], y = y[I], 
                     n = n[I], sigma = sigma)
        ## Calculate marginal negative log-likelihood
        nll <- nll + opt$objective + 0.5 * log(as.numeric(H[i])) - 
            0.5 * log(2 * pi)
    }
    ## If full = TRUE return marginal likelihood and random effects
    ## including variance of random effects
    if(full){return(list(ll=-nll,u=u,Vu=1/H))}
    nll
}

##################################################
## Example 6.10

## Set up
dfcount$subjId <- as.factor(dfcount$subjId) ## Set subject Id as factor
subj.lev <- levels(dfcount$subjId) ## List of different subjects

## Observations:
y <- dfcount$clo
n <- dfcount$nobs

## Design matrix
X <- cbind(1,  dfcount$sex=="male") 

## Outer optimisation:
fit <- nlminb(c(0,0,0), nll.la, X = X, y = y, n = n,
              subj.lev = subj.lev, subjId=dfcount$subjId )

## Find standard errors and report results
H <- hessian(nll.la, fit$par,  X = X, y = y, n=n,
             subj.lev = subj.lev, subjId=dfcount$subjId)
se <- sqrt(diag(solve(H)))
sigma <- sqrt(exp(fit$par[3]+c(0, -1, 1) * qnorm(0.975)*se[3]))
partab <- cbind(fit$par[1:2], fit$par[1:2] - qnorm(0.975)*se[1:2],
                fit$par[1:2] + qnorm(0.975)*se[1:2])
partab <- rbind(partab,sigma)
partab <- cbind(partab, c(coef( fit.binom),NA))
rownames(partab) <- c("beta0","beta1","sigma")
colnames(partab) <- c("Est","lower","upper", "glm-pars")
partab

## Observed probabilities
p.obs <- aggregate(clo~sex,sum,data=dfcount)[ ,2]/
    aggregate(nobs~sex,sum,data=dfcount)[ ,2]
## glm-model 
p.glm <- 1/(1+exp(-cumsum(coef(fit.binom))))

## random effect model (only fixed effect pars
p.glmm <- 1/(1+exp(-cumsum(fit$par[1:2])))

set.seed(2455)
## Boot strap estimates of mean values
u <- rnorm(1e6,sd=exp(fit$par[3]/2))
p.glmm2 <- c(mean(1/(1+exp(-fit$par[1]-u))),
             mean(1/(1+exp(-sum(fit$par[1:2])-u))))
cbind(obs=p.obs, glmm.fix = p.glmm, glmm.boot = p.glmm2, glm = p.glm)


c(glm.logLik = logLik(fit.binom), glmm.logLik =-fit$objective)

##################################################
## Example 6.11

## derivative of nll.u
dnll.u <- function(u, eta, y, n, sigma){
    p <- exp(eta + u)/(1 + exp(eta + u))
    - sum(y - n * p) +  u / sigma^2
}

## Hessian matrix of nll.u
Hnll.u <- function(u, eta, y, n, sigma){
    p <- exp(eta + u)/(1 + exp(eta + u))
    matrix(sum( p * (1 - p) * n) +  1 / sigma^2)
}


nll.la.Deriv <- function(theta, X, y, n, subj.lev, subjId, 
                         full = FALSE){
    beta <- theta[1:dim(X)[2]] ## Fixed effect parameters
    ## Random effect parameters
    sigma <- exp(theta[-(1:dim(X)[2])]/2) 
    eta <- as.numeric(X %*% beta) ## Linear predictor
    nll <- 0 ## Initalize negative marginal log-likelihood
    u <- numeric(length(subj.lev))
    H <- u
    ## For loop over random effects
    for(i in 1:length(subj.lev)){
        I <- subjId == subj.lev[i] ## Choose subject
        ## Optimise random effect for given parameters
        opt <- nlminb(0, nll.u, gradient = dnll.u, 
                      hessian = Hnll.u, eta = eta[I], y = y[I], 
                      n = n[I], sigma = sigma)
        ## Extract random effect and Hessian 
        ## of random effect likelihood
        u[i] <- opt$par
        H[i] <-  Hnll.u(u[i], eta = eta[I], y = y[I], 
                        n = n[I], sigma = sigma)
        ## Calculate marginal negative log-likelihood
        nll <- nll + opt$objective + 
            0.5 * log(as.numeric(H[i])) - 0.5 * log(2 * pi)
    }
    ## If full return marginal likelihood and random effects
    ## including variance of random effects
    if(full){return(list(ll=-nll,u=u,Vu=1/H))}
    nll
}




##################################################
## Section 6.2
##################################################

##################################################
## Example 6.12

## library(glmmTMB)
summary(fit.tmb.binom <- glmmTMB(cbind(clo, nobs - clo) ~ sex + 
                               (1 | subjId), family = binomial, 
                           data = dfcount))

fit.tmb.pois <- glmmTMB(clo ~ sex + offset(log(time)) + (1 | subjId), 
                        family = poisson, data = dfcount)

c(l.binom.glmm = logLik(fit.tmb.binom),l.pois.glmm = logLik(fit.tmb.pois), l.binom.glm=logLik(fit.binom), l.pois.glm=logLik(fit.poisOff))

##################################################
## Figure 6.1
beta <- fit.tmb.binom$fit$par[1:2]
sigma <- exp(fit.tmb.binom$fit$par[3])
x <- seq(min(cumsum(beta)-2*sigma),max(cumsum(beta)+2*sigma),length=200)
subjId <- rownames(ranef(fit.tmb.binom)[[1]]$subjId)
subj.sex <- aggregate(sex~subjId,data=dfcount,function(x){x[1]})

par(mfrow=c(1,2),mai=c(0.5,0.5,0.1,0.1),oma=c(2,2,1,1))
matplot(x,cbind(dnorm(x,mean=beta[1],sd=sigma),
                dnorm(x,mean=sum(beta),sd=sigma)),type="l",
        xlab="u",ylab="density",lwd=2,col=gray(c(0,0.5)),lty=1:2)
lines(coef(fit.binom)[1]*c(1,1),c(-1,1),col=1,lwd=2,lty=4)
r.f <- ranef(fit.tmb.binom)[[1]]$subjId[ ,1][subj.sex[ ,2]=="female"]+ beta[1]
r.m <- ranef(fit.tmb.binom)[[1]]$subjId[ ,1][subj.sex[ ,2]=="male"]+  sum(beta)
rug(r.f,col=gray(0),lwd=2)
rug(r.m,col=gray(0.5),lwd=2,lty=2)
lines(sum(coef(fit.binom))*c(1,1),c(-1,1),col=gray(0.5),lwd=2,lty=4)

mtext("density",side=2,line=0,outer=TRUE)
mtext(expression(u+beta),side=1,line=3,outer=FALSE)

dlogit.norm <- function(x,mu,sigma){
    y <- exp(x)/(1+exp(x))
    1/(sigma*sqrt(2*pi))/(y*(1-y))*exp(-(x-mu)^2/(2*sigma^2))
}
u <- seq(min(cumsum(beta)-10*sigma),max(cumsum(beta)+1.5*sigma),length=200)
matplot(exp(u)/(1+exp(u)),cbind(dlogit.norm(u,mu=beta[1],sigma=sigma),
                                dlogit.norm(u,mu=sum(beta),sigma=sigma)),type="l",
        lty=1:2,ylab="",lwd=2,col=gray(c(0,0.5)))
lines(c(1,1)*exp(coef(fit.binom)[1])/(1+exp(coef(fit.binom)[1])),
     c(-1,20),col=1,lwd=2,lty=4)
lines(c(1,1)*exp(sum(coef(fit.binom)))/(1+exp(sum(coef(fit.binom)))),
     c(-1,20),col=gray(0.5),lwd=2,lty=4)

rug(exp(r.f)/(1+exp(r.f)),,col=gray(0),lwd=2)
rug(exp(r.m)/(1+exp(r.m)),col=gray(0.5),lwd=2,lty=2)

mtext(expression(exp(u+beta)/(1+exp(u+beta))),side=1,line=3,outer=FALSE)
legend("topright",legend=c("Females","Males","glm est."),col=gray(c(0,0.5,0)),lty=c(1,2,4),lwd=c(2,2,2))


##################################################
## Example 6.13

modMEMEx18AR1Gam <- glmmTMB(clo ~ tOutCen * sex + (1 | subjId) + 
                                ar1(as.factor(obs.no) -1 | subjId:day),
                            family = Gamma(log), data = dfMEMEx1,
                            disp = ~ sex) 
summary(modMEMEx18AR1Gam)

c(AIC(modMEMEx18AR1Gam),AIC(modMEMEx18AR1))

##################################################
## Figure 6.2
df.window <- read.table("data/windatGlmm.csv",sep=";")
dat.win.agg <- aggregate(cbind(y,lco2)~h+dwel,data=df.window,mean)
dat.win.agg2 <- aggregate(cbind(y,lco2)~h,data=df.window,mean)

# head(dat.win.agg)
par(mfrow=c(1,2),oma=c(4,4,2,4),mai=c(0,0.5,0,0))
plot(dat.win.agg$h/2/pi*24,dat.win.agg$y,pch=19,col="white",ylab="",xlab="")
mtext(side=1,line=3,"Hour of day")
mtext(side=2,line=3,"Probability")
n <- 0
col.tmp <- gray(seq(0.1,0.75,length=8))
pch.tmp <- 1:8
for(i in 0:7){
    I <- dat.win.agg$dwel>(i-0.1) & dat.win.agg$dwel<(i+0.1)
    n <- n+sum(I)
    lines(dat.win.agg$h[I]/2/pi*24,dat.win.agg$y[I],col=col.tmp[i+1],type="b",
          lty=i+1,pch=pch.tmp[i+1],cex=0.5)
}

lines(dat.win.agg2$h/2/pi*24,dat.win.agg2$y,col=1,pch=19,lwd=3)

legend("topright",ncol=2,legend=c("1","2","3","4"),col=col.tmp[1:4],lty=1:4,
       pch=pch.tmp[1:4],lwd=2,pt.cex=0.5)

plot(dat.win.agg$h/2/pi*24,exp(dat.win.agg$lco2),pch=19,col="white",log="y",
     lwd=2,axes=FALSE,ylab="",xlab="")
axis(4)
axis(1)
mtext(side=1,line=3,"Hour of day")
mtext(side=4,line=3,expression(CO[2]))
box()
n <- 0
for(i in 0:7){
    I <- dat.win.agg$dwel>(i-0.1) & dat.win.agg$dwel<(i+0.1)
    n <- n+sum(I)
    lines(dat.win.agg$h[I]/2/pi*24,exp(dat.win.agg$lco2[I]),col=col.tmp[i+1],
          type="b",pch=pch.tmp[i+1],lty=1:8,cex=0.5)
}
lines(dat.win.agg2$h/2/pi*24,exp(dat.win.agg2$lco2),col=1,pch=19,lwd=3)

legend("topright",ncol=2,legend=c("5","6","7","8"),col=col.tmp[5:8],lty=5:8,
       pch=pch.tmp[5:8],lwd=2,pt.cex=0.5)



##################################################
## Example 6.14

fit.window <- glmmTMB(y ~  I(cos(h)) + I(sin(h)) + I(lco2) + 
                          ( 1 | dwel), family = binomial, 
                      data = df.window)

summary(fit.window)

fit.window.glm <- glmmTMB(y ~  I(cos(h)) + I(sin(h)) +  I(lco2), 
                          family = binomial, data = df.window)

anova(fit.window.glm,fit.window)[ c(1,5:8)]

fit.window$sdr$par.random


h <- seq(0,2*pi,length=200)

X <- cbind(1,cos(h),sin(h),log(800))
eta <- X%*%t(coef(fit.window)[[1]]$dwel[1, ])
u <- fit.window$sdr$par.random
col.tmp <- gray(seq(0.1,0.75,length=8))
par(mfrow=c(2,2),mai=c(0.75,0.75,0.1,0.25),oma=c(0,0,0,0),bg="white")
plot(h*24/2/pi,eta,lwd=3,type="l",ylim=c(min(eta)+min(u),max(eta)+max(u)),xlab="Hour of day",ylab="Linear predictor")
rug(df.window$Hour)
for(i in 1:length(u)){
    lines(h*24/2/pi,eta+u[i],lwd=1,type="l",col=col.tmp[i],lty=i)
}


lco2 <- seq(min(df.window$lco2),max(df.window$lco2),length=200)
X <- cbind(1,cos(2*pi*8/24),sin(2*pi*8/24),lco2)
eta <- X%*%t(coef(fit.window)[[1]]$dwel[1, ])
u <- fit.window$sdr$par.random
plot(exp(lco2),eta,lwd=3,type="l",ylim=c(min(eta)+min(u),max(eta)+max(u)),xlab="CO2 concentration",ylab="Linear predictor",log="x")
rug(exp(df.window$lco2))
for(i in 1:length(u)){
    lines(exp(lco2),eta+u[i],lwd=1,type="l",,col=col.tmp[i],lty=i)
}

co2.fun <- approxfun(dat.win.agg2$h,dat.win.agg2$lco2,rule=2)
X <- cbind(1,cos(h),sin(h),co2.fun(h))
eta <- X%*%t(coef(fit.window)[[1]]$dwel[1, ])
plot(h*24/2/pi,eta,lwd=3,type="l",ylim=c(min(eta)+min(u),max(eta)+max(u)),xlab="Hour of day",ylab="Linear predictor")
for(i in 1:length(u)){
    co2.fun <- approxfun(dat.win.agg$h[dat.win.agg$dwel==(i-1)],dat.win.agg$lco2[dat.win.agg$dwel==(i-1)],rule=2)
    X <- cbind(1,cos(h),sin(h),co2.fun(h))
    eta <- X%*%t(coef(fit.window)[[1]]$dwel[1, ])
    lines(h*24/2/pi,eta+u[i],lwd=1,type="l",col=col.tmp[i],lty=i)
}

co2.fun <- approxfun(dat.win.agg2$h,dat.win.agg2$lco2,rule=2)
X <- cbind(1,cos(h),sin(h),co2.fun(h))
eta <- X%*%t(coef(fit.window)[[1]]$dwel[1, ])
plot(h*24/2/pi,exp(eta)/(1+exp(eta)),lwd=3,type="l",ylim=c(0,0.045),
     xlab="Hour of day",ylab="Probability")
for(i in 1:length(u)){
    co2.fun <- approxfun(dat.win.agg$h[dat.win.agg$dwel==(i-1)],dat.win.agg$lco2[dat.win.agg$dwel==(i-1)],rule=2)
    X <- cbind(1,cos(h),sin(h),co2.fun(h))
    eta <- X%*%t(coef(fit.window)[[1]]$dwel[1, ])
    lines(h*24/2/pi,exp(eta+u[i])/(1+exp(eta+u[i])),lwd=1,type="l",col=col.tmp[i],lty=i)
}



##################################################
## Example 6.15


# read data
dfCeilingFan <- read.csv(file = "data/exCeilingFan_data.csv") 

# tell R that its all factors
dfCeilingFan$fanSpeed <- factor(dfCeilingFan$fanSpeed, level=c("0","1","2"))
dfCeilingFan$subjId <- factor(dfCeilingFan$subjId) #21 subjects
dfCeilingFan$TSV <- factor(dfCeilingFan$TSV, level=c("0", "1", "2"))



  library(ordinal)
 ## Null model
 m0S0 <- clmm(fanSpeed ~ 1 + (1 | subjId), data = dfCeilingFan)
 ## Including TSV
 m0S1 <- clmm(fanSpeed ~ TSV + (1 | subjId), data = dfCeilingFan)
 ## Including random effect of TSV
 m0S2 <- clmm(fanSpeed ~ TSV + (1 + TSV | subjId),
              data = dfCeilingFan) 
 anova(m0S0,m0S1,m0S2)


 m00S1 <- clm(fanSpeed ~ 1 + TSV, data=dfCeilingFan)
anova(m0S1,m00S1)

 summary(m0S1)

par(mfrow=c(1,1),bg="white")
TSVf <- c(0, as.numeric(m0S1$coeff[3:4])) # coeffs from clmm for TSV
p1  <-  logistic_cdf( m0S1$coeff[1] - TSVf )
p2 <-  logistic_cdf( m0S1$coeff[2] - TSVf ) - p1
p3 <-  1 - logistic_cdf( m0S1$coeff[2] - TSVf )
pch <- c(19,17,15)
col.tmp <- gray(c(0.2,0.4,0.6))
matplot(1:3,cbind(p1,p2,p3),type="b",ylim=c(0,1),lwd=4,lty=2:4,pch=pch,ylab="Predicted probability of chosen fan speed",
        xlab="Thermal sensation vote",axes=FALSE,cex=1,col=col.tmp)
axis(at=c(1:3),labels=c("neutral","slightly warm","warm"),side=1)
axis(2)
box()
u <- m0S1$ranef
for(i in 1:length(u)){
    p1 <-  logistic_cdf( m0S1$coeff[1] - TSVf -u[i])
    p2 <-  logistic_cdf( m0S1$coeff[2] - TSVf -u[i]) - p1 
    p3 <-  1-logistic_cdf( m0S1$coeff[2] - TSVf -u[i]) 
    matlines(1:3,cbind(p1,p2,p3),lty=2:4,col=col.tmp,type="b",pch=pch,cex=0.5)
} 
TSVf <- c(0, as.numeric(m0S1$coeff[3:4])) # coeffs from clmm for TSV
p1  <-  logistic_cdf( m0S1$coeff[1] - TSVf )
p2 <-  logistic_cdf( m0S1$coeff[2] - TSVf ) - p1
p3 <-  1 - logistic_cdf( m0S1$coeff[2] - TSVf )

legend("topright",col=col.tmp,lwd=4,lty=2:4,pch=pch,legend=c("Speed 1","Speed 2","Speed 3"))
matlines(1:3,cbind(p1,p2,p3),type="b",lwd=4,pch=pch,cex=1.5,lty=2:4,
         col=col.tmp)






TSVf <- c(0, as.numeric(m0S1$coeff[3:4])) # coeffs from clmm for TSV


p1 <-  logistic_cdf( m0S1$coeff[1] - TSVf )
p2 <-  logistic_cdf( m0S1$coeff[2] - TSVf ) -logistic_cdf( m0S1$coeff[1] - TSVf )
p3 <-  1 - logistic_cdf( m0S1$coeff[2] - TSVf )

prob <- c(p1,p2,p3)
speed <- rep(c("Speed 1", "Speed 2", "Speed 3"),each = 3)
TSVp <- rep(c("neutral", "slightly warm", "warm"), 3)

dfPlot <- data.frame(TSVp, speed, prob)

ggplot(dfPlot, aes(TSVp, prob, group = speed, colour = speed,
                   shape = speed)) + geom_point(size=3) +
    geom_line() +
    xlab("Thermal sensation vote") + 
    ylab("Predicted probability of chosen fan speed") +
    ylim(c(0,1))+
    theme_bw() +	 
    theme(legend.justification=c(1,1),
          legend.position=c(.80,.95),
          axis.text.x=element_text(size=9),
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12)  		  
		)          

 m1S <- clmm(fanSpeed ~ TSV * fanType + (1 |subjId), 
             data=dfCeilingFan)
 anova(m0S1,m1S)

##################################################
## Section 6.3
##################################################

##################################################
## Example 6.16
df.window <- read.table("data/windatGlmm.csv",sep=";")
 library(TMB)
compile("cpp-files/window2.cpp")
dyn.load(dynlib("cpp-files/window2"))

## Set up parameter, notice random effects also parameters
tmbpars = list(
  u    = cbind(rep(0,8),rep(0,8)),
  pars = c(0,0,-12,2.2,1,2)
)

tmbdata = list(
  Y    = as.matrix(df.window[,c("y","n")]),
  X    = as.matrix(df.window[,c("h","lco2")]),
  dwel = as.vector(df.window$dwel)
)

## Define objective function
obj <- MakeADFun(data = tmbdata,
                 parameters = tmbpars,
                 random = "u",
                 DLL = "window2",
                 silent=FALSE)


opt3 <- nlminb(obj$par, obj$fn, obj$gr,
               lower=c(-Inf,-Inf,-Inf,0,-Inf,-Inf),
               upper=c(Inf,Inf,Inf,2*pi,Inf,Inf))


## Random effects
rap <- sdreport(obj,getJointPrecision = TRUE)
u1 <- rap$par.random[1:8]
u2 <- rap$par.random[9:16]

h <- seq(0,2*pi,by=0.01)
beta <- opt3$par[-(1:2)]
lco2 <- log(800)
eta <-   beta[1] + beta[2] * cos(h - beta[4] ) + beta[3] * lco2
par(mfrow=c(2,2),mai=c(0.75,0.75,0.1,0.25),oma=c(0,0,0,0),bg="white")

plot(24*h/2/pi,eta,type="l",ylim = c(-8.5,-2.5),lwd=3,xlab="Time of day",ylab=expression(eta))
legend("topright",legend=expression(paste(CO[2],"=",800)))
col.tmp=gray(seq(0.1,0.75,length=8))
for(i in 1:8){
    eta <-   beta[1] + beta[2] * cos(h - beta[4] - u2[i]) + beta[3] * lco2 + u1[i]
    lines(24*h/2/pi,eta,col=col.tmp[i],lty=i)
}

lco2 <- seq(min(df.window$lco2),max(df.window$lco2),length=200)
h <- 6/24*2*pi
eta <-   beta[1] + beta[2] * cos(h - beta[4] ) + beta[3] * lco2
plot(lco2,eta,type="l",ylim = c(-6.5,-1),lwd=3,xlab=expression(log(CO[2])),ylab=expression(eta))
legend("topleft",legend="Time of day 6AM")
for(i in 1:8){
    eta <-   beta[1] + beta[2] * cos(h - beta[4] - u2[i]) + beta[3] * lco2 + u1[i]
    lines(lco2,eta,col=col.tmp[i],lty=i)
}


co2.fun <- approxfun(dat.win.agg2$h,dat.win.agg2$lco2,rule=2)
h <- seq(0,2*pi,by=0.01)
eta <-   beta[1] + beta[2] * cos(h - beta[4] ) + beta[3] * co2.fun(h)
plot(24*h/2/pi,eta,type="l",ylim = c(-8.5,-2.5),lwd=3,xlab="Time of day",ylab=expression(eta))
legend("topright",legend=expression(paste(CO[2],"=",f(t))))
for(i in 1:8){
    co2.fun <- approxfun(dat.win.agg$h[dat.win.agg$dwel==(i-1)],dat.win.agg$lco2[dat.win.agg$dwel==(i-1)],rule=2)
    eta <-   beta[1] + beta[2] * cos(h - beta[4] - u2[i]) + beta[3] * co2.fun(h) + u1[i]
    lines(24*h/2/pi,eta,col=col.tmp[i],lty=i)
}

eta <-   beta[1] + beta[2] * cos(h - beta[4] ) + beta[3] * co2.fun(h)
plot(24*h/2/pi,exp(eta)/(1+exp(eta)),type="l",ylim = c(0,0.1),lwd=3,xlab="Time of day",ylab="Probability")
legend("topright",legend=expression(paste(CO[2],"=",f(t))))
for(i in 1:8){
    co2.fun <- approxfun(dat.win.agg$h[dat.win.agg$dwel==(i-1)],dat.win.agg$lco2[dat.win.agg$dwel==(i-1)],rule=2)
    eta <-   beta[1] + beta[2] * cos(h - beta[4] - u2[i]) + beta[3] * co2.fun(h) + u1[i]
    lines(24*h/2/pi,exp(eta)/(1+exp(eta)),col=col.tmp[i],lty=i)
}





##################################################
## Example 6.17

compile("cpp-files/windowVonM2.cpp")
dyn.load(dynlib("cpp-files/windowVonM2"))
parameters <- list(u1 = rep(0, 8), ## notice random effects also parameters
                   u2 = rep(0, 8),
                   beta = c(0,0,0,3.14),
                   sigma_1 = 0,
                   lkappa = 0)

## Define objective function
obj <- MakeADFun(data = df.window,
                 parameters = parameters,
                 random = c("u1","u2"),
                 DLL = "windowVonM2",
                 silent=TRUE)

optVM <- nlminb(obj$par, obj$fn, obj$gr,lower=c(-Inf,-Inf,-Inf,0,-Inf,-Inf),upper=c(Inf,Inf,Inf,2*pi,Inf,Inf))

rap <- sdreport(obj,getJointPrecision = TRUE)
u1VM <- rap$par.random[1:8]
u2VM <- rap$par.random[9:16]

t <- seq(0,2*pi,length=200)
f.vm <- exp((exp(optVM$par[6]) * cos(t - optVM$par[4]) - log(2*pi) - log(besselI(exp(optVM$par[6]), 0))))
par(mfrow=c(1,1),bg="white")
plot(24*t/2/pi,f.vm,type="l",ylim=c(0,0.65),xlab="Time of day",ylab="density",
     lwd=2)
lines(24*t/2/pi,dnorm(t,opt3$par[6],exp(opt3$par[1])),col=gray(0.5),lwd=2,lty=2)

legend("topright",legend=c("von Mises","Gausian"),lty=1:2,col=gray(c(0,0.5)),lwd=2)
rug(((opt3$par[6]+u2)%%(2*pi))*24/2/pi,col=gray(0.5),lwd=4,lty=2)
rug(u2VM*24/2/pi,lwd=2)


fit.window
opt3$par
optVM$par

logLik(fit.window)
opt3$objective


##################################################
## Section 6.4
##################################################

##################################################
## Example 6.18

##################################################
## Example 6.19

## Negative log-likelihood
nll.dnbinom <- function(theta, X, y, off.set, full = FALSE){
    mu <- exp(X %*% theta[1:dim(X)[2]] + off.set)
    phi <- exp(theta[dim(X)[2] + 1])
    ll <- sum(dnbinom(y, size =  phi, mu=mu, log = TRUE))
    if(full){
        E <- (phi + y) / (phi + mu)
        V <- E / (phi + mu)
        return(list(ll = ll, theta = theta, E = E, V = V))
    }
    - ll
}

## Estimate bench-mark models
## Poisson
pois0 <- glmmTMB(clo ~ offset(log(time)) + sex, data = dfcount,
                 family=poisson)
## Normal-Poisson
pois.norm1 <- glmmTMB(clo ~ offset(log(time)) + sex + (1|subjId:day),
                      data=dfcount,family=poisson)
## Gamma-Poisson
X <- model.matrix(fit.tmb.pois)
theta <- c(0,0,0)
opt.dnbinom1 <- nlminb(theta, nll.dnbinom, X = X, y = dfcount$clo,
                       off.set = log(dfcount$time))

## compare parameters and log-likelihood
tab <- rbind(cbind(opt.dnbinom1$par[1:2], fixef(pois0)[[1]],
      fixef(pois.norm1)[[1]]),c(-opt.dnbinom1$objective,logLik(pois0),logLik(pois.norm1)))
row.names(tab)[3] <- "log-Like."
colnames(tab) <- c("gamma-Pois","Pois","norm-Pois")
tab

summ.dbinom1 <- nll.dnbinom(theta = opt.dnbinom1$par, X,  
                            y = dfcount$clo,
                            off.set = log(dfcount$time), 
                            full = TRUE)
c(mean(summ.dbinom1$E), mean(exp(pois.norm1$sdr$par.random)))-1

par(bg="white")
plot(sort(summ.dbinom1$E), sort(exp(pois.norm1$sdr$par.random)),xlab="gamma-Poisson random effects",
     ylab="normal-Poisson random effects",pch=19)
abline(a=0,b=1)

##################################################
## Example 6.20

dfcount$sex<-factor(dfcount$sex)
dfcount$day<-factor(dfcount$day)
dfcount$subjId<-factor(dfcount$subjId)

pois.norm0 <- glmmTMB(clo ~ offset(log(time)) + sex, 
                      data = dfcount, family = poisson)
pois.norm1.subj <- glmmTMB(clo ~ offset(log(time)) + sex + 
                               (1|subjId),
                           data = dfcount, family = poisson)
pois.norm2.subj <- glmmTMB(clo ~ sex + (1|subjId),
                           data = dfcount, family = poisson)


 nll.hir <- function(theta,dat,X){
     lambda <- exp(X %*% theta[1:2] + log(dat$time))
     beta <- exp(-theta[3])
     alpha<-1 / beta
     l <- 0
     lev.sub <- levels(dat$subjId)
     for(i in 1:length(lev.sub)){
         yi <- dat$clo[dat$subjId == lev.sub[i]]
         lambdai <- lambda[dat$subjId == lev.sub[i]]
         l <- l  - (lgamma(alpha + sum(yi)) - lgamma(alpha) -
                    sum(lfactorial(yi)) + sum(yi * log(lambdai)) - 
                    alpha * log(1 + beta * sum(lambdai)) +
                    sum(yi) * log(beta) - 
                    sum(yi) * log(1 + beta * sum(lambdai)))
     }
     l
 }

X <- model.matrix(pois.norm0)
 
fit.hir <- nlminb(c(0,0,0),nll.hir,dat=dfcount,X=X,
                  control=list(trace=0))

tab <- cbind("gamma-Pois"=c(fit.hir$par,-fit.hir$objective),"norm-Pois"=c(fit.tmb.pois$fit$par,logLik(fit.tmb.pois)),
             "Pois"=c(fixef(pois0)[[1]],NA,logLik(pois0)))
row.names(tab) <- c("Intercept","sex","Disp","log-Lik")
tab

sig <- exp(fit.tmb.pois$fit$par[3])
tab<-rbind(cbind(exp(sig^2/2),1),
      sqrt(cbind((exp(sig^2)-1)*exp(sig^2),1/exp(fit.hir$par[3]))))
rownames(tab) <- c("E","sd")
colnames(tab) <- c("Pois-Norm","Pois-Gam")
tab

## Conditional mean and var of U
X <- model.matrix(pois.norm0)

  lambda <- exp(X%*%fit.hir$par[1:2]+log(dfcount$time))
  beta <- exp(-fit.hir$par[3])
  alpha<-1/beta
  dat.tmp <- cbind(dfcount,lambda=lambda)
  lam.sum <- aggregate(lambda~subjId,sum,dat=dat.tmp)[ ,2]
  y.sum <- aggregate(clo~subjId,sum,dat=dat.tmp)[ ,2]
  u.hat <- (1 + beta * y.sum)/(1 + beta * lam.sum)
  Vu.hat <- (beta*(1 + beta * y.sum)/(1 + beta * lam.sum)^2)

   theta <- fit.hir$par
   beta <- exp(-theta[3])
   alpha<-1/beta
   
   par(mfrow=c(1,2),bg="white")
   plot(log(u.hat),pois.norm1.subj$sdr$par.random,xlab="log(Pois-Gam random effects)",
        ylab="Pois-norm random effects",pch=19)
   abline(a=0,b=1)
   
   plot(u.hat,exp(pois.norm1.subj$sdr$par.random),xlab="Pois-Gam random effects",
        ylab="exp(Pois-norm random effects)",pch=19)
   abline(a=0,b=1)
   which(u.hat>2)
    dfcount[dfcount$subjId==levels(factor(dfcount$subjId))[44], ]
    

##################################################
## Example 6.21

## pois.norm1.subj
dfcount.summ <- cbind(aggregate(cbind(clo,nobs,time) ~ subjId, 
                                data = dfcount,sum),
                      sex = aggregate(sex ~ subjId,
                                      data = dfcount, 
                                      function(x){x[1]})[ ,2])


## Likelihood compound beta-binomial
ldCBb <- function(y,mu,phi,n){
    lchoose(n, y) + lbeta(y + mu * phi, n - y +(1 - mu) * phi) - 
        lbeta(mu * phi, (1 - mu) * phi)
}

CBinBetaReg <- function(pars,y,n,X){
  p <- dim(X)[2]
  mu <- 1 / (1 + exp( -(X %*% pars[1:p])))
  phi <- exp(pars[p+1])
  -sum(ldCBb(y, mu, phi, n))
}



fit.glm.binom <- glm(cbind(clo,nobs-clo) ~ sex, 
               data = dfcount.summ,
               family=binomial)
fit.glm.pois <- glm(clo ~ sex+offset(log(time)), 
                    data = dfcount.summ,
               family=poisson)

X <- cbind(1,dfcount.summ$sex=="male")
fit.hir.beta <- nlminb(c(-2,-1,0), CBinBetaReg,
                       y = dfcount.summ$clo, 
                       n = dfcount.summ$nobs, X=X)
fit.hir.pois <- nlminb(c(-2,-1,0), nll.dnbinom, 
                       y = dfcount.summ$clo, 
                       off.set = log(dfcount.summ$time), 
                       X=X)


par.tab <- cbind(
    c(fit.hir.beta$par,- fit.hir.beta$objective),
    c(c(coef(fit.glm.binom),NA),logLik(fit.glm.binom)),
    c(fit.hir.pois$par,- fit.hir.pois$objective),
    c(c(coef(fit.glm.pois),NA),logLik(fit.glm.pois)))

row.names(par.tab)[3:4] <- c("log-phi","log-likelihood")
colnames(par.tab) <- c("Beta-binom","Binom","Gamma-Pois","Pois")
par.tab


my.dnbinom <- function(y,mu,phi){
    p <- mu / (phi +mu)
    exp(lgamma(phi+y) - lgamma(phi) - lfactorial(y) + 
        phi * log(1 - p) + y * log(p))
}
offset <- log(dfcount.summ$time)
phi.pois <- exp(fit$par[3])
fit.pois <- glm(clo~sex+offset(log(time)),data=dfcount.summ,family=poisson)

p <- exp(cumsum(coef(fit.glm.binom)))/(1+exp(cumsum(coef(fit.glm.binom))))


mu <-  exp(cumsum(fit.hir.beta$par[1:2]))/(1+exp(cumsum(fit.hir.beta$par[1:2])))
phi <- exp(fit.hir.beta$par[3])


dfcount.summ2 <- cbind(aggregate(cbind(clo,nobs,time) ~ subjId + day, 
                                data = dfcount,sum),
                      sex = aggregate(sex ~ subjId+day,
                                      data = dfcount, 
                                      function(x){x[1]})[ ,3])


dfM <- dfcount.summ[dfcount.summ$sex=="male", ]
dfF <- dfcount.summ[dfcount.summ$sex=="female", ]

dfM <- dfM[dfM$nobs<17 & dfM$nobs>13, ]
dfF <- dfF[dfF$nobs<17 & dfF$nobs>13, ]


x <- 0:20

tabF <- matrix(0,ncol=5,nrow=21)
tabM <- matrix(0,ncol=5,nrow=21)
i<-23


for(i in 1:dim(dfF)[1]){
    tabF[ ,2] <- tabF[ ,2] +  dbinom(x,prob=p[1],size=dfF$nobs[i])       
    tabF[ ,3] <- tabF[ ,3] +  dpois(x,exp(coef(fit.pois)[1]+log(dfF$time[i])))      
    tabF[ ,4] <- tabF[ ,4] +  exp(ldCBb(x,mu[1],phi,dfF$nobs[i]))
    tabF[ ,5] <- tabF[ ,5] +  my.dnbinom(0:20,
                                         mu= exp(fit$par[1]+log(dfF$time[i])),phi=phi.pois)       

}

for(i in 1:dim(dfM)[1]){
    tabM[ ,2] <- tabM[ ,2] +  dbinom(x,prob=p[2],size=dfM$n[i])       
    tabM[ ,3] <- tabM[ ,3] +  dpois(x,exp(sum(coef(fit.pois)[1:2])+log(dfM$time[i])))      
    tabM[ ,4] <- tabM[ ,4] +  exp(ldCBb(x,mu[2],phi,dfM$n[i]))
    tabM[ ,5] <- tabM[ ,5] +  my.dnbinom(x,mu= exp(sum(fit$par[1:2])+log(dfM$time[i])),phi=phi.pois)
}

for(i in 0:20){
    tabF[i+1,1] <- sum(dfF$clo==i)
    tabM[i+1,1] <- sum(dfM$clo==i)
}

colnames(tabM) <- c("Obs","Binom","Pois","Beta-binom","Gamma-Pois")
colnames(tabF) <- c("Obs","Binom","Pois","Beta-binom","Gamma-Pois")

## tabM
tabM.R <- rbind(tabM[1, ],
    colSums(tabM[2:3, ]),
    colSums(tabM[4:21, ]))
rownames(tabM.R) <- c("0","1-2",">2")

## tabF
tabF.R <- rbind(
              colSums(tabF[1:2, ]),
              colSums(tabF[3:4, ]),
      colSums(tabF[5:21, ]))
rownames(tabF.R) <- c("0-1","2-3",">3")
 

tabF.R
tabM.R

##################################################
## Example 6.22

make.sig <- function(rho,ns,sig.s){
    Sigma <- matrix(rho, ncol = ns, nrow = ns)
    diag(Sigma) <- 1
    Sigma <- Sigma * sig.s
}

report.fun <- function(M, y, mu , Sigma, phi, ns, sex, subj, s, sig.s){
    ri <- as.numeric((y - mu) %*% solve(Sigma) %*% (y - mu))
    M[s, c("Egam", "Vgam", "sex", "subj","sig.s")] <- 
        c((ns / 2 + phi)/(ri / 2 + phi), 
        (ns / 2 + phi) / (ri / 2 + phi)^2, sex, subj, sig.s)
    M
}

nll.t <- function(theta, y, X, X.sig, subj, full = FALSE){
    ## Parameters
    n.par <- c(mean = dim(X)[2], var = dim(X.sig)[2])
    mu <- X %*% theta[1:n.par[1]]
    sig.s <- exp(X.sig %*% theta[1:n.par[2] + n.par[1]])
    rho.u <- 2 / (1 + exp(-theta[sum(n.par) + 1])) - 1
    phi <- exp(theta[sum(n.par) + 2])
    ny <- 2 * phi
    subj.lev <- unique(subj)
    n.subj <- length(subj.lev)
    ll <- vec <- numeric(n.subj)
    ## Initialize dataframe for reporting
    M <- data.frame(sex = vec, subj = subj.lev, 
                         Egam = vec, Vgam = vec, sig.s=vec)
    for(s in 1:n.subj){
        ## Calculate likelihood
        I <- subj == subj[s]
        ns <- sum(I)
        Sigma <- make.sig(rho.u,ns,sig.s[s])
        ll[s] <- dmvt(y[I] - mu[I], sigma = Sigma, df = ny, 
                      log = TRUE)
        ## For reporting posterior
        M <- report.fun(M, y[I], mu[I] , Sigma, phi, ns,
                        sex[I][1], subj[s], s, sig.s[s])
    }
    if(full){return(list(ll = ll, M = M, rho.u = rho.u,
                         sigma=sqrt(sigma), phi = phi))}
    -sum(ll)
}


 library(mvtnorm)
dat.agrr <- aggregate(cbind(clo, sex, tOutCen) ~ subjId + day,
                      mean, data = dfMEMEx1)
dat.agrr$subjId <- as.numeric(factor(dat.agrr$subjId))
sex <- as.numeric(dat.agrr$sex)-1
X <- cbind(1,sex,dat.agrr$tOut)
X.sig <- cbind(1,sex)
y <- dat.agrr$clo
theta <- rep(0,7)

opt.tdist <- nlminb(theta, nll.t, lower = -10, upper = 10,
                    y = y, X = X, X.sig =X.sig, 
                    subj = dat.agrr$subj)


H<-hessian(nll.t,opt.tdist$par,y=y,X=X,X.sig=X.sig,subj = dat.agrr$subj)
se <- sqrt(diag(solve(H)))
tab <- rbind(cbind(opt.tdist$par[1:3], 
                   opt.tdist$par[1:3] - 2*se[1:3],
                   opt.tdist$par[1:3]+2*se[1:3]),
             exp(cbind(opt.tdist$par[4:5], 
                       opt.tdist$par[4:5] - 2*se[4:5],
                       opt.tdist$par[4:5] + 2*se[4:5])),
             1/(1 + exp(-c(opt.tdist$par[6], 
                           opt.tdist$par[6] - 2*se[6],
                           opt.tdist$par[6]+2*se[6]))),
             exp(cbind(opt.tdist$par[7], 
                       opt.tdist$par[7]-2*se[7],
                       opt.tdist$par[7]+2*se[7])))
rownames(tab) <- c("beta0", "beta1", "beta2", "sigma",
                   "alpha","rho.u", "phi")
colnames(tab) <- c("Estimate", "lower", "upper")
tab[c("sigma", "alpha"), ] <- sqrt(tab[c("sigma", "alpha"), ])
tab

X <- cbind(1,sex,dat.agrr$tOut)
X.sig <- cbind(1,sex)
y <- dat.agrr$clo

res.tdist <- nll.t(opt.tdist$par, y = y, X = X, X.sig = X.sig, subj = dat.agrr$subj,full=TRUE)
par(mfrow=c(1,2),bg="white")
res.tdist$M$sig.i <- res.tdist$M$sig.s / 
    res.tdist$M$Egam
boxplot(sig.i~sex,pch=19,axes=FALSE,
        ylab=expression(sigma^2/gamma),xlab="Sex",
        data=res.tdist$M)

box()
axis(1,at=c(1,2),labels=c("Females","Males"))
axis(2)
boxplot(Egam~sex,pch=19,axes=FALSE,
        ylab=expression(gamma),xlab="Sex",
        data=res.tdist$M)
box()
axis(1,at=c(1,2),labels=c("Females","Males"))
axis(2)


##################################################
## Section 6.5
##################################################
