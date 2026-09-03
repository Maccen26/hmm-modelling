library(expm)
##################################################
## Example 9.1

win1 <- read.table("data/b1.csv",sep=";",header=TRUE)
win1 <- win1[!is.na(win1$WindowClosed), ]

win1 <- win1[win1$Room=="Bedroom", ]
win1 <- win1[1:5000, ]
minute <- (win1$Time*24*60)%%60
Rday <- win1$Time + win1$Day -  18 + (win1$Month - 3) *31
win.dat <- cbind(Rday, win1[ ,c("Time","Day", "Month","WindowClosed","Hour",#"WeekDay",
                                "OutdoorTemp","Wind","TempC","CO2C")])



y <- win1$WindowClosed

Dt <- 10/60/24
Rday <- seq(min(win.dat$Rday),max(win.dat$Rday),by=Dt)

approxfun.Time <- approxfun(win.dat$Rday,win.dat$Time)
approxfun.Day <- approxfun(win.dat$Rday,win.dat$Day)
approxfun.Month <- approxfun(win.dat$Rday,win.dat$Month)
approxfun.WindowClosed <- approxfun(win.dat$Rday,win.dat$WindowClosed)
approxfun.Hour <- approxfun(win.dat$Rday,win.dat$Hour)
approxfun.WeekDay <- approxfun(win.dat$Rday,win.dat$WeekDay)
approxfun.OutdoorTemp <- approxfun(win.dat$Rday,win.dat$OutdoorTemp)
approxfun.Wind <- approxfun(win.dat$Rday,win.dat$Wind)
approxfun.TempC <- approxfun(win.dat$Rday,win.dat$TempC)
approxfun.CO2C <- approxfun(win.dat$Rday,win.dat$CO2C)

win.dat2 <- data.frame(Rday=Rday,Time=approxfun.Time(Rday),
                       Day=round(approxfun.Day(Rday)),
                       Month=round(approxfun.Month(Rday)),
                       WindowClosed=round(approxfun.WindowClosed(Rday)),
                       Hour=round(approxfun.Hour(Rday)),
                       WeekDay=round(approxfun.WeekDay(Rday)),
                       OutdoorTemp=approxfun.OutdoorTemp(Rday),
                       Wind=approxfun.Wind(Rday),
                       TempC=approxfun.TempC(Rday),
                       CO2C=approxfun.CO2C(Rday))




(n <- length(win.dat2$WindowClosed))
table(win.dat2$WindowClosed) / n

par(mfrow=c(1,1),mar=c(4,4,1,4),bg="white")
plot(win.dat2$Rday,win.dat2$WindowClosed,type="l",axes=FALSE,ylab="Window position",
     xlab="Time [days]")
box();axis(1);axis(2,at=c(0,1))
axis(4,at=c(0,1),labels=c("Open","Closed"))

par(mfrow=c(1,2),bg="white")
acf(win.dat2$WindowClosed,lag.max=3*24*6,main="Auto-correlation",axes=FALSE,xlab="Time/lags [Hours]",ci.col=gray(0.5))
axis(1,at=seq(0,3*24*6,by=6*24),labels=c(0,24,48,3*24))
axis(2);box()
plot(aggregate(WindowClosed~Hour,data=win.dat2,mean),type="l",ylim=c(0,1),main="Average window position",xlab="Hour of day")

##################################################
## Section 9.1
##################################################

##################################################
## Example 9.2

##################################################
## Section 9.2
##################################################

##################################################
## Example 9.3

## End example
##################################################

trans.count <- function(y, m){
    ## y take values 1,...,m
    F <- matrix(0, ncol = m, nrow = m)
    n <- length(y)
    for(i in 1:m){
        for(j in 1:m){
            F[i, j] <- sum(y[-n] == i & y[-1] == j)
        }
    }
    F
}

##################################################
## Example 9.4

win.dat2$WinClo1 <- win.dat2$WindowClosed+1
(F <- trans.count(win.dat2$WinClo1, 2))

(Gamma1 <- F / rowSums(F))

ll1 <- sum(F * log(Gamma1))
## null model 
y <- win.dat2$WinClo1
delta <- 1 / length(y) * c(sum(y == 1), sum(y == 2))
Gamma0 <- rbind(delta, delta)
ll0 <- sum(F * log(Gamma0))
## Compare likelihoods
ll1 - ll0

I1 <- F[1,1] / Gamma1[1,1] ^ 2 + F[1,2] / Gamma1[1,2] ^ 2
I2 <- F[2,2] / Gamma1[2,2] ^ 2 + F[2,1] / Gamma1[2,1] ^ 2
(se <- c(sqrt(1 / I1), sqrt(1 / I2)))

rbind(gamma12 = c(1,1) * Gamma1[1,2] + c(-1, 1) * 2 * se[1],
      gamma21 = c(1,1) * Gamma1[2,1] + c(-1, 1) * 2 * se[2])











##################################################
## Section 9.3
##################################################

##################################################
## Example 9.5

## End example
##################################################

stat.dist <- function(Gamma){
    m <- dim(Gamma)[1]
    E <- matrix(1, ncol = m, nrow = m)
    rep(1, m) %*% solve(diag(m) - Gamma + E)
}    

##################################################
## Example 9.6

(delta <- stat.dist(Gamma1))

## End example
##################################################

acf.markov <- function(k,Gamma){
    m <- dim(Gamma)[1]
    v <- 1:m
    V <- diag(v)
    delta <- stat.dist(Gamma)
    gamma0 <- delta %*% V %*%  v - (delta %*% v)^2                     
    (delta %*% V %*% (Gamma %^% k) %*% v - 
     (delta %*% v)^2) / gamma0
}


##################################################
## Example 9.7

par(bg="white")
acf.obs <- acf(win.dat$WindowClosed,lag.max=48,plot=FALSE)$acf
acf1 <- sapply(1:48,acf.markov,Gamma=Gamma1)
plot(((1:48)-0.2)/6,acf.obs[-1],type="h",xlab="Time/lags [Hours]",ylab="ACF",lwd=2)
lines(((1:48)+0.2)/6,acf1,type="h",col=gray(0.5),lwd=2)
legend("topright",legend=c("Empirical","Markov model"),col=gray(c(0,0.5)),lty=1,lwd=2)
abline(a=0,b=0)

## End example
##################################################



sim.markov <- function(Gamma, n, y1 = NULL){
    m <- dim(Gamma)[1]; y <- numeric(n)
    if(is.null(y1)){
        delta <- stat.dist(Gamma)
        y1 <- sample(1:m, size = 1, prob = delta)
    }
    y[1] <- y1
    for(i in 2:n){
        y[i] <- sample(1:m, size = 1, prob = Gamma[y[i - 1], ])
    }
    y
}



##################################################
## Section 9.4
##################################################

Markov.link <- function(Gamma){
    m <- dim(Gamma)[1]
    diag(Gamma) <- 0
    beta <-t( log(Gamma/(1-rowSums(Gamma))))
    diag(beta) <- NA
    beta[!is.na(beta)]   
}

Markov.Inv.Link <- function(pars,m){
    exp.pars <- exp(pars)
    Gamma <- diag(m)
    Gamma[Gamma != 1] <- exp.pars
    t(Gamma) / (colSums(Gamma))
}



nll1.markov <- function(pars,y,m){
    F <- trans.count(y, m)
    Gamma <- Markov.Inv.Link(pars, m)
    delta <- stat.dist(Gamma)
    ll <- log(delta[y[1]]) + sum(F * log(Gamma))
    - ll
}

##################################################
## Example 9.8

opt1.markov <- nlminb(c(0, 0), nll1.markov, 
                      y = win.dat2$WinClo1, m = 2)
opt1.markov$par

Markov.Inv.Link(opt1.markov$par,2)

##################################################
## Example 9.9

##################################################
## Example 9.10

tau <- c(log(F[1, 2] / F[1, 1]), log(F[2, 1] / F[2, 2]))
I <- diag(c(prod(Gamma1[1, ]) * sum(F[1, ]), 
            prod(Gamma1[2, ]) * sum(F[2, ])))
V.tau <- solve(I)

set.seed(1454)
k <- 1e4
Tau <- rmvnorm(k, mean = tau, solve(I))

Theta <- matrix(ncol=2,nrow=k)
colnames(Theta) <- c("p12","p21")
Delta <- matrix(ncol=2,nrow=k)
colnames(Delta) <- c("delta1","delta2")
for(i in 1:k){
    Gam.tmp <- Markov.Inv.Link(Tau[i, ], 2)
    Theta[i, ] <- c(Gam.tmp[1, 2], Gam.tmp[2, 1])
    Delta[i, ] <- stat.dist(Gam.tmp)
}

apply(Theta, 2, quantile, prob = c(0.025, 0.975))
apply(Delta, 2, quantile, prob = c(0.025, 0.975))

k <- 100
set.seed(1245)
Tau2 <- matrix(ncol=2,nrow=k)
Theta2 <- matrix(ncol=2,nrow=k);
colnames(Theta2) <- c("p12","p21")
Delta2 <- matrix(ncol=2,nrow=k)
colnames(Delta2) <- c("delta1","delta2")
T <- n
for(i in 1:k){
    yi <- sim.markov(Gamma1, T)
    F.tmp <- trans.count(yi, 2)
    Tau2[i, ] <- c(log(F.tmp[1, 2] / F.tmp[1, 1]), 
                   log(F.tmp[2, 1] / F.tmp[2, 2]))
    Gam.tmp <- Markov.Inv.Link(Tau2[i, ], 2)
    Theta2[i, ] <- c(Gam.tmp[1, 2], Gam.tmp[2, 1])
    Delta2[i, ] <- stat.dist(Gam.tmp)
}

apply(Theta2, 2, quantile, prob = c(0.025, 0.975))
apply(Delta2, 2, quantile, prob = c(0.025, 0.975))

##################################################
## Section 9.5
##################################################

##################################################
## Example 9.11

##################################################
## Figure 9.1
source("functions/markovPlot.R")


## End figure
##################################################

 make.states <- function(m, l){
     state.mat <- matrix(1, ncol = l, nrow  =m ^ l)
     tpm <- matrix(0, ncol = m ^ l, nrow  =m ^ l)
     states <- 1 : m ^ l
     s <- 0
     for(i in 1:m^(l-1)){
         for(j in 1:m){
             state.mat[s+j, ] <- c(state.mat[i, -1], j)
         }
         s <- s + m
     }
     state.mat
 }
 

##################################################
## Example 9.12

(state.mat2 <- make.states(2, 2))

## End example
##################################################

  make.high.order.seq <- function(y, state.mat){
      y.new <- y * 0
      s <- 1:dim(state.mat)[1]
      l <- dim(state.mat)[2]
      for(i in l:length(y)){
          for(j in s){
              if(sum(y[(i-l+1):i] == state.mat[j, ]) == l){
                  y.new[i] <- j
              }
          }
      }
      y.new
  }

##################################################
## Example 9.13

win.dat2$WinClo2 <- make.high.order.seq(win.dat2$WinClo1, state.mat2)

(tpm.struc <- rbind(c(1, 2, 0, 0), c(0, 0, 2, 1), 
                    c(1, 2, 0, 0), c(0, 0, 2, 1)))


(F2 <- trans.count(win.dat2$WinClo2,4))

  Markov.Inv.Link.Q <- function(pars, tpm.struc){
      exp.pars <- exp(pars)
      m <- dim(tpm.struc)[1]
      Gamma <- matrix(0, ncol = m, nrow = m)
      s <- 1
      for(i in 1:m){
          r <- sum(tpm.struc[i, ] == 2)
          p <- exp.pars[s : (s + r - 1)] / 
              (1 + sum(exp.pars[s:(s + r - 1)]))
          Gamma[i, tpm.struc[i, ] == 2] <- p
          Gamma[i, tpm.struc[i, ] == 1] <- 1 - sum(p)
          s <- s + r 
      }
      Gamma
  }

 nll1.struc.markov <- function(pars, y, m, tpm.struc){
     F <- trans.count(y, m)
     Gamma <- Markov.Inv.Link.Q(pars, tpm.struc)
     delta <- as.numeric(stat.dist(Gamma))
     ll <- log(delta[y[1]]) + sum(F[tpm.struc != 0] * 
                                  log(Gamma[tpm.struc != 0]))
     - ll
 }

pars <- rep(0, 4)
opt2.markov <- nlminb(pars, nll1.struc.markov, y = win.dat2$WinClo2[-1],
                      m = 4, tpm.struc = tpm.struc)


 -c(opt2.markov$objective, opt1.markov$objective)



(Gamma2 <- Markov.Inv.Link.Q(opt2.markov$par,tpm.struc))

##################################################
## Example 9.14

1/delta

delta2 <- stat.dist(Gamma2)
1/delta2



## End example
##################################################




##################################################
## Section 9.6
##################################################

Markov.Inv.Link.Reg <- function(pars, m, X){
    p <- dim(X)[2]; n <- dim(X)[1]
    beta <- matrix(pars[1 : (p * m)], ncol = m)
    exp.eta <- exp(X %*% beta)
    exp.tau <- exp(pars[-(1: (p * m))])
    Tau.mat <- diag(m)
    Tau.mat[Tau.mat != 1] <- exp.tau
    Gamma <- array(dim=c(n, m, m))
    for(i in 1:n){
        diag(Tau.mat) <- exp.eta[i, ]
        Gamma[i, , ] <- t(Tau.mat) / colSums(Tau.mat)
    }
    Gamma
}

nll.markov.reg <- function(pars, y, X, mm){
    n <- length(y)
    Gamma <- Markov.Inv.Link.Reg(pars, mm, X)
    lGamma <- log(Gamma)
    ll <- 0
    for(i in 2:n){
        ll <- ll + lGamma[i - 1, y[i - 1], y[i]]
    }
    -ll
}

##################################################
## Example 9.15

X1 <- cbind(sin(2 * pi * win.dat2$Time), cos(2 * pi * win.dat2$Time),
            sin(4 * pi * win.dat2$Time), cos(4 * pi * win.dat2$Time))
m <- 2
pars <- c(rep(0, dim(X1)[2] * m), opt1.markov$par)
opt.markov.reg1 <- nlminb(pars, nll.markov.reg, y = win.dat2$WinClo1, 
                          X = X1, mm = m)

X2 <- cbind(X1, log(win.dat2$CO2C) - 6.5)
pars <- c(opt.markov.reg1$par[1 : 4], 0, 
          opt.markov.reg1$par[5 : 8], 0, 
          opt.markov.reg1$par[9 : 10])
opt.markov.reg2 <- nlminb(pars, nll.markov.reg, y = win.dat2$WinClo1, 
                          X = X2, mm = m)

Q <- -2 * (opt.markov.reg2$objective - opt.markov.reg1$objective)
1 - pchisq(Q, df = length(opt.markov.reg2$par) - 
                  length(opt.markov.reg1$par))

c(ll1, -opt.markov.reg2$objective)

library(numDeriv)
H <- hessian(nll.markov.reg,opt.markov.reg2$par,y=y,X=X2,mm=m)
tab <- cbind(opt.markov.reg2$par,sqrt(diag(solve(H))))
tab <- cbind(tab,tab[ ,1]/tab[ ,2])
rownames(tab) <- c("beta11.1","beta11.2","beta11.3","beta11.4","beta11.5",
                   "beta22.1","beta22.2","beta22.3","beta22.4","beta22.5","tau12","tau21")

colnames(tab) <- c("theta","se","z")
tab

X2Re <- X2[ , -(3 : 4)]
pars <- rep(0, 2 * dim(X2Re)[2] + 2)
opt.markov.reg2Re <- nlminb(pars, nll.markov.reg, 
                            y = win.dat2$WinClo1, X = X2Re, 
                            mm = m)
Q <- - 2 * (opt.markov.reg2$objective - 
            opt.markov.reg2Re$objective)
1 - pchisq(Q, df = length(opt.markov.reg2$par) - 
                  length(opt.markov.reg2Re$par))

H <- hessian(nll.markov.reg,opt.markov.reg2Re$par,y=y,X=X2Re,mm=m)
tab <- cbind(opt.markov.reg2Re$par,sqrt(diag(solve(H))))
tab <- cbind(tab,tab[ ,1]/tab[ ,2])
rownames(tab) <- c("beta11.1","beta11.2","beta11.3",
                   "beta22.1","beta22.2","beta22.3","tau12","tau21")

colnames(tab) <- c("theta","se","z")
tab

t <- seq(0,1,length=200)
X.plot <- cbind(sin(2*pi*t),cos(2*pi*t),0)
X.plot2 <- cbind(sin(2*pi*t),cos(2*pi*t),-0.5)
X.plot3 <- cbind(sin(2*pi*t),cos(2*pi*t),0.5)

par(mfrow=c(1,2),bg="white")
Gamma.plot <- Markov.Inv.Link.Reg(opt.markov.reg2Re$par,m,X.plot)
Gamma2.plot <- Markov.Inv.Link.Reg(opt.markov.reg2Re$par,m,X.plot2)
Gamma3.plot <- Markov.Inv.Link.Reg(opt.markov.reg2Re$par,m,X.plot3)
ylim <- c(0,1)
col.tmp <- gray(c(0,0.25,0.5))
matplot(t*24,cbind(Gamma.plot[ ,1,2],
                Gamma2.plot[ ,1,2],
                Gamma3.plot[ ,1,2]),type="l",ylim=ylim,
        ylab=expression(p[ij]),xlab="Time of day",
        main=expression(p[12]),lwd=2,lty=c(1,2,4),col=col.tmp)
matplot(t*24,cbind(Gamma.plot[ ,2,1],
                Gamma2.plot[ ,2,1],
                Gamma3.plot[ ,2,1]),type="l",ylim=ylim,
        axes=FALSE,ylab="",xlab="Time of day",
        main=expression(p[21]),lwd=2,lty=c(1,2,4),col=col.tmp)
box()
axis(1);axis(2)
legend("topright",legend=c(expression(paste(CO[2],"  670 ppm")),
                           expression(paste(CO[2],"  400 ppm")),
                           expression(paste(CO[2]," 1100 ppm"))),lwd=2,lty=c(1,2,4),
       col=col.tmp)

##################################################
## Example 9.16

sim.markovReg <- function(Gamma, y1){
    T <- dim(Gamma)[1]
    m <- dim(Gamma)[2]; y <- numeric(T)
    y[1] <- y1
    for(i in 2:T){
        y[i] <- sample(1:m, size = 1, prob = Gamma[i,y[i-1], ])
    }
    y
}
t <- seq(0,1,by=10/(60*24))
t <- rep(t,10000)
X.tmp <- cbind(sin(2*pi*t),cos(2*pi*t))
Gamma.sim1 <- Markov.Inv.Link.Reg(opt.markov.reg2Re$par,2,cbind(X.tmp,0))
Gamma.sim2 <- Markov.Inv.Link.Reg(opt.markov.reg2Re$par,2,cbind(X.tmp,-0.5))
Gamma.sim3 <- Markov.Inv.Link.Reg(opt.markov.reg2Re$par,2,cbind(X.tmp,0.5))
y.sim1 <- sim.markovReg(Gamma = Gamma.sim1, y1 = 2)
y.sim2 <- sim.markovReg(Gamma = Gamma.sim2, y1 = 2)
y.sim3 <- sim.markovReg(Gamma = Gamma.sim3, y1 = 2)
df.tmp1 <- data.frame(y.sim=y.sim1,t=t)
df.tmp2 <- data.frame(y.sim=y.sim2,t=t)
df.tmp3 <- data.frame(y.sim=y.sim3,t=t)
mean.tmp1 <- aggregate(y.sim~t,data=df.tmp1,mean)
mean.tmp2 <- aggregate(y.sim~t,data=df.tmp2,mean)
mean.tmp3 <- aggregate(y.sim~t,data=df.tmp3,mean)
col.tmp <- gray(c(0,0.25,0.5))
par(bg="white")
matplot(mean.tmp1$t*24,cbind(mean.tmp1$y.sim, 
                            mean.tmp2$y.sim,
                            mean.tmp3$y.sim)-1,type="l",xlab="Time of day",
     ylab="Probability of window closed",ylim=c(0,1),lty=c(1,2,4),lwd=2,
     col=col.tmp)
legend("bottomright",legend=
                         c(expression(paste(CO[2],"  670 ppm")),
                           expression(paste(CO[2],"  400 ppm")),
                           expression(paste(CO[2]," 1100 ppm"))),
       lwd=2,lty=c(1,2,4),col=col.tmp)







##################################################
## Section 9.7
##################################################

##################################################
## Example 9.17

Xar <- cbind(C = win.dat2$WindowClosed, X1)
colnames(Xar) <- c("C", "sin1", "cos1", "sin2", "cos2")
ar2x.reg <- arima(log(win.dat2$CO2C), order=c(2, 0, 0), xreg = Xar)

ar2x.reg

nll.markovSwich <- function(pars, y, C, Gamma, X, full = FALSE){
    ## parameters
    beta0 <- pars[1] + pars[2] * C
    beta <- pars[3 : (dim(X)[2] + 2)]
    phi <- pars[(dim(X)[2] + 3) : (dim(X)[2] + 4)]
    sigma <- exp(pars[dim(X)[2] + 5])
    ## initialization
    n <- length(y); mu <- rep(NA, n); sigma2 <- rep(NA, n)
    f <- X %*% beta
    ll <- 0
    ## main loop for likelihood
    for(i in 3:n){
        index <- (i-1):(i-2)
        tmp <- pars[1] + f[i] + sum(phi * (y[index] - beta0[index] - f[index]))
        ll <- ll + log(sum(Gamma[i, C[i - 1] + 1,1 : 2] * 
                           dnorm(y[i], mean =tmp + c(0, pars[2]), sd = sigma)))
        if(full){
            mu[i] <- tmp + sum(Gamma[i, C[i - 1] + 1, 2] * pars[2])
            sigma2[i] <- sigma^2 + prod(Gamma[i, C[i - 1] + 1, 1 : 2]) * 
                pars[2]^2
        }
    }
    if(full){return(list(mu = mu, sigma2 = sigma2))}
    -ll    
}

X <- Xar[ ,-1] ## remove win. pos. from X
## All transition probabilities
Gamma <- Markov.Inv.Link.Reg(opt.markov.reg1$par, 2, X)
pars <- c(coef(ar2x.reg)[3:4], coef(ar2x.reg)[5:8],
          coef(ar2x.reg)[1:2], -2)
opt2 <- nlminb(pars,nll.markovSwich, y = log(win.dat2$CO2C),
               C = win.dat2$WindowClosed, Gamma = Gamma, X = X)

logLik(ar2x.reg)

-opt2$objective

names(opt2$par) = c("Intercept","beta","sin1","cos1","sin2","cos2","ar1","ar2","lsig")
opt2$par[1:2] ## intercept and effect of window pos.
opt2$par[3:6] ## diurnal variation
opt2$par[7:8] ## AR-parameters
exp(opt2$par["lsig"] * 2) ## sigma^2



##################################################
## Section 9.8
##################################################

##################################################
## Example 9.18
library(msm)
win.dat2$WinClo1 <- win.dat2$WindowClosed+1
win.dat2$time.min <- (1:length(win.dat2$WindowClosed))*10
Q <- rbind(c(-0.5, 0.5), c(0.5,-0.5))
fit.msm <- msm(WinClo1 ~ time.min, data = win.dat2, 
               qmatrix = Q ) 

fit.msm

(Gamma1min <- expm(fit.msm$Qmatrices$baseline))

Gamma1min %^% 10

ll1 - logLik(fit.msm)

##################################################
## Section 9.9
##################################################
