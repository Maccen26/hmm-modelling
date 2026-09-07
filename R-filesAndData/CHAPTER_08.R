##################################################
## Section 8.1
##################################################

##################################################
## Example 8.1

win1 <- read.table("data/b1.csv",sep=";",header=TRUE)

win1 <- win1[!is.na(win1$WindowClosed), ]
win1 <- win1[win1$Room=="Bedroom", ]

# win1$NoPresenceRoom[is.na(win1$NoPresenceRoom)]<-0
# win1$NoPresenceDwelling[is.na(win1$NoPresenceDwelling)]<-0
win1 <- win1[1:5000, ]## Only a subset of the data is considered
## win1$day <- win1$Date-min(win1$Date)
win1$HalfHour <- round((win1$day+win1$Time)*24*2)

#win1.agr <- aggregate(cbind(CO2C,WindowClosed,NoPresenceRoom,NoPresenceDwelling,Time,Day,Month)~
#                          HalfHour, data=win1,mean)
win1.agr <- aggregate(cbind(CO2C,WindowClosed,Time,Day,Month)~
                          HalfHour, data=win1,mean)
win1.agr$HalfHour <-  (win1.agr$HalfHour)%%48

win1.agr$Hour <- win1.agr$HalfHour/2
win1.agr$Time <-  (1:length(win1.agr$Hour))/2/24

par(mfrow=c(1,1),mar=c(4,5,4,5),bg="white")
plot(win1.agr$Time,log(win1.agr$CO2C),type="b",ylab=expression(log(CO[2])),
     xlab="Time [days]",pch=19,cex=0.5)
lines(win1.agr$Time,win1.agr$WindowClosed+6,col=gray(0.6),lwd=3)
axis(4,at=c(6,7),lab=c(0,1))
mtext("Window Closed",side=4,line=3)
legend("topright",legend=c(expression(log(CO[2])),"Win. Pos."),lty=1,col=gray(c(0,0.6)),pch=c(19,NA),lwd=c(1,3))


par(mfrow=c(1,2),mar=c(5,5,1,0),oma=c(0,0,0,4),bg="white")
I <- 100:350
plot(win1.agr$Time[I],log(win1.agr$CO2C)[I],type="b",ylab=expression(log(CO[2])),
     xlab="Time [days]",pch=19,cex=0.5)
lines(win1.agr$Time[I],win1.agr$WindowClosed[I]+6,col=gray(0.6),lwd=3)
legend("topleft",legend=c(expression(log(CO[2])),"Win. Pos."),lty=1,col=gray(c(0,0.6)),pch=c(19,NA),lwd=c(1,3))
axis(4,at=c(6,7),lab=c(0,1))

axis(4,at=c(6,7),lab=c(0,1))
plot(win1.agr$Hour,log(win1.agr$CO2C),xlab="Time of day",pch=19,cex=0.5,ylab="",axes=FALSE)
axis(4);box()
axis(1)


fitCO2.1 <- lm(log(CO2C) ~ WindowClosed + sin(2 * pi * Hour / 24) +
                   cos(2 * pi * Hour / 24) + sin(4 * pi * Hour / 24) + 
                   cos(4 * pi * Hour / 24), data = win1.agr)

par(mfrow=c(1,3),bg="white")
panel.fun.tmp <- function(x, y, ...)
              panel.smooth(x, y, iter=3, col.smooth=gray(0.5),lwd=3,
                           lty=2,...)

plot(fitCO2.1,which=c(1,2),pch=19,cex=0.5, panel = panel.fun.tmp)
res <- residuals(fitCO2.1)
n <- length(res)
plot(res[-n],res[-1],pch=19,cex=0.5,xlab=expression(e[t-1]),ylab=expression(e[t]))

res <- residuals(fitCO2.1)
n <- length(res)
cor(res[-1], res[-n])

##################################################
## Example 8.2


nll.fun <- function(theta, y){
    phi <- theta[1]; sigma.sq <- theta[2]
    ll <- 0
    for(t in 2 : length(y)){
        ll <- ll + dnorm(y[t], mean = phi * y[t-1], 
                         sd = sqrt(sigma.sq), log = TRUE)}
    -ll 
}

opt.ar1 <- nlminb(c(0, 1), nll.fun, y=res,lower=c(-Inf,0))

phi <- opt.ar1$par[1]; sigma <- sqrt(opt.ar1$par[2])
c(phi = phi, sigma = sigma)

c(var(res), sigma^2 / (1 - phi^2))

##################################################
## Section 8.2
##################################################

##################################################
## Example 8.3

##################################################
## Example 8.4

##################################################
## Example 8.5

##################################################
## Example 8.6

par(mfrow=c(1,2),bg="white")
acf(res,ci.col=gray(0.5))
pacf(res,ci.col=gray(0.5))



n <- length(res)
e <- res[-1] - phi * res[-n]



par(mfrow=c(1,2),bg="white")
acf(e,ci.col=gray(0.5))
pacf(e,ci.col=gray(0.5))


##################################################
## Section 8.3
##################################################

##################################################
## Example 8.7

##################################################
## Example 8.8

##################################################
## Figure 8.1

buildA <- function(phi1,phi2){
    A <- matrix(0,ncol=3,nrow=3)
    A[1,1] <- 1
    A[1,2] <- -phi1
    A[1,3] <- -phi2
    A[2,1] <- -phi1
    A[2,2] <- 1-phi2
    A[3,1] <- -phi2
    A[3,2] <- -phi1
    A[3,3] <- 1
    b <- c(sigma,0,0)
    gamma <- solve(A,b)
    gamma
}

gammaAR2.fun <- function(phi1,phi2,sigma,lag.max){
    gamma <- numeric(lag.max+1)
    gamma[1:3] <- sigma^2*buildA(phi1,phi2)
    for(i in 4:(lag.max+1)){
        gamma[i] <- phi1 * gamma[i-1] + phi2 * gamma[i-2]
    }
    gamma
}

phi1 <- 0.7;phi2=0.1;sigma=1
ar.pars1 <- c(phi1,phi2,sigma)
gamma.ar1 <- gammaAR2.fun(ar.pars1[1],ar.pars1[2],ar.pars1[3],lag.max=15)
rho.ar1 <- gamma.ar1[2:16]/gamma.ar1[1]

phi1 <- 1.3;phi2<--0.6;sigma=1
ar.pars2 <- c(phi1,phi2,sigma)
gamma.ar2 <- gammaAR2.fun(ar.pars2[1],ar.pars2[2],ar.pars2[3],lag.max=15)
rho.ar2 <- gamma.ar2[2:16]/gamma.ar2[1]

YuleWalker.fun <- function(rho){
    n <- length(rho)
    P <- diag(n)    
    for(i in 1:(n-1)){
        P[(i+1):n,i] <- P[i,(i+1):n]  <- rho[1:(n-i)]
    }
    phi.kk <- numeric(n)
    for(i in 1:n){
        phi.kk[i] <- solve(P[1:i,1:i],rho[1:i])[i]
    }
    phi.kk
}

rho.part.ar1 <- YuleWalker.fun(rho.ar1)
rho.part.ar2 <- YuleWalker.fun(rho.ar2)

gammaARMA11.fun <- function(phi,theta,sigma,lag.max){
    gamma <- numeric(lag.max + 1)
    gamma[1] <- sigma^2 * (1+(phi+theta)^2/(1-phi^2))
    gamma[2] <- phi * gamma[1] + theta * sigma^2
    for(i in 3:(lag.max+1)){
        gamma[i] <- phi * gamma[i-1]
    }
    gamma
}



arma.pars1 <- c(0.8,-0.3,1)
arma.pars2 <- c(0.6,0.8,1)

gamma.arma1 <- gammaARMA11.fun(arma.pars1[1],arma.pars1[2],arma.pars1[3],lag.max=15)
gamma.arma2 <- gammaARMA11.fun(arma.pars2[1],arma.pars2[2],arma.pars2[3],lag.max=15)

rho.arma1 <- gamma.arma1[2:16]/gamma.arma1[1]
rho.arma2 <- gamma.arma2[2:16]/gamma.arma2[1]


rho.part.arma1 <- YuleWalker.fun(rho.arma1)
rho.part.arma2 <- YuleWalker.fun(rho.arma2)

ylim <- range(rho.ar1,rho.ar1,rho.part.ar2,1)
lwd=2
col.tmp <- gray(c(0,0.5))
par(mfrow=c(2,2),mar=c(1,1,1,1),oma=c(4,5,4,4))
plot(c(0:(length(rho.ar1))-0.1,0:(length(rho.ar1))+0.1),c(1,rho.ar1,1,rho.ar2) ,
     type="h",col=rep(col.tmp,each=length(rho.ar1)),ylim=ylim,axes=FALSE,
     ylab="",xlab="",lwd=lwd)
legend("topright",legend=as.expression(c(bquote(phi[1]*"="*.(-ar.pars1[1])*", "*phi[2]*"="*.(-ar.pars1[2])),
                                         bquote(phi[1]*"="*.(-ar.pars2[1])*", "*phi[2]*"="*.(-ar.pars2[2]))
                                         )),lty=1,col=col.tmp,lwd=2)
box();axis(2)
mtext("ACF",side=3,cex=1.5)
mtext("AR(2)",side=2,cex=1,line=3)
plot(c(1:(length(rho.ar1))-0.1,1:(length(rho.ar1))+0.1),
       c(rho.part.ar1,rho.part.ar2) ,type="h",col=rep(col.tmp,each=length(rho.ar1)),
     ylim=ylim,ylab="",xlab="",axes=FALSE,lwd=lwd)

box();mtext("PACF",side=3,cex=1.5)
plot(c(0:(length(rho.arma1))-0.1,0:(length(rho.arma1))+0.1),c(1,rho.arma1,1,rho.arma2), type="h",col=rep(col.tmp,each=length(rho.arma1)),ylim=ylim,ylab="",xlab="",axes=FALSE,lwd=lwd)

legend("topright",legend=as.expression(c(bquote(phi[1]*"="*.(-arma.pars1[1])*", "*theta[1]*"="*.(arma.pars1[2])),
                                         bquote(phi[1]*"="*.(-arma.pars2[1])*", "*theta[1]*"="*.(arma.pars2[2]))
                                         )),lty=1,col=col.tmp,lwd=2)

box();axis(2);axis(1)
mtext("ARMA(2)",side=2,cex=1,line=3)

plot(c(1:length(rho.arma1)-0.1,1:length(rho.arma1)+0.1),c(rho.part.arma1,rho.part.arma2) ,type="h",col=rep(col.tmp,each=length(rho.arma1)),ylim=ylim,ylab="",xlab="",axes=FALSE,lwd=lwd)
box();axis(1)
mtext("Lags",side=1,line=3,outer=TRUE)



##################################################
## Example 8.9

##################################################
## Section 8.4
##################################################

##################################################
## Example 8.10

n <- length(res)
Y <- res[-(1:2)]
X <- cbind(res[-c(1, n)], res[-c(n - 1, n)])
(phi <- coef(lm(Y ~ -1 + X)))

fit.ar2 <- arima(res,order=c(2, 0, 0),include.mean = FALSE)           
coef(fit.ar2) - phi

##################################################
## Example 8.11

css.arma11 <- function(theta,y){
    n <- length(y)
    e <- numeric(n)
    for(i in 2:n){
        e[i] <- (y[i] - theta[1] * y[i - 1] - theta[2] * e[i - 1])
    }
    sum(e^2)
}

n <- length(res)
opt.arma11 <- nlminb(c(0, 0), css.arma11, y = res)

arma11.fit <- arima(res,order=c(1, 0, 1), include.mean = FALSE)
coef(arma11.fit) - opt.arma11$par

##################################################
## Example 8.12

opt.arma11$objective / (n - 1 - 2)

arma11.fit$sigma2

##################################################
## Example 8.13

par(mfrow=c(1,2),bg="white")
acf(residuals(fit.ar2),ci.col=gray(0.5))
acf(residuals(fit.ar2),type="partial",ci.col=gray(0.5))


##################################################
## Example 8.14

fit.ar2

AIC(fit.ar2, arma11.fit)

fit.ar1 <- arima(res, order=c(1, 0, 0), include.mean = FALSE)
(Q <- 2 * (logLik(fit.ar2) - logLik(fit.ar1)))

##################################################
## Section 8.5
##################################################

##################################################
## Example 8.15



##################################################
## Example 8.16

par(mfrow=c(1,2),bg="white")
acf1 <- acf(residuals(fit.ar2),lag.max=3*48,ci.col=gray(0.5),col=gray(0.25))
lines(48*c(1,1),c(0,acf1$acf[49]),lty=1,lwd=3,col=1)
lines(2*48*c(1,1),c(0,acf1$acf[1+2*48]),lty=1,lwd=3,col=1)
lines(3*48*c(1,1),c(0,acf1$acf[1+3*48]),lty=1,lwd=3,col=1)
pacf1 <- pacf(residuals(fit.ar2),lag.max=3*48,ci.col=gray(0.5),col=gray(0.25))
lines(48*c(1,1),c(0,pacf1$acf[48]),lty=1,lwd=3,col=1)
lines(2*48*c(1,1),c(0,pacf1$acf[2*48]),lty=1,lwd=3,col=1)
lines(3*48*c(1,1),c(0,pacf1$acf[3*48]),lty=1,lwd=3,col=1)


ar2Sma1.fit <- arima(res, order = c(2, 0, 0),
                     seasonal = list(order = c(0 ,0, 1), period = 48),
                     include.mean=FALSE)
ar2Sar1.fit <- arima(res,order=c(2,0,0),seasonal=list(order=c(1,0,0),period=48),include.mean=FALSE)

par(mfrow=c(1,2),bg="white")
acf1 <- acf(residuals(ar2Sma1.fit),lag.max=3*48,ci.col=gray(0.5),col=gray(0.25))
lines(48*c(1,1),c(0,acf1$acf[49]),lty=1,lwd=3,col=1)
lines(2*48*c(1,1),c(0,acf1$acf[1+2*48]),lty=1,lwd=3,col=1)
lines(3*48*c(1,1),c(0,acf1$acf[1+3*48]),lty=1,lwd=3,col=1)
pacf1 <- pacf(residuals(ar2Sma1.fit),lag.max=3*48,ci.col=gray(0.5),col=gray(0.25))
lines(48*c(1,1),c(0,pacf1$acf[48]),lty=1,lwd=3,col=1)
lines(2*48*c(1,1),c(0,pacf1$acf[2*48]),lty=1,lwd=3,col=1)
lines(3*48*c(1,1),c(0,pacf1$acf[3*48]),lty=1,lwd=3,col=1)

##################################################
## Example 8.17

X <- model.matrix(fitCO2.1)
ar2Sma1x.reg <- arima(log(win1.agr$CO2), order = c(2, 0, 0),
                      seasonal = list(order = c(0, 0, 1), period = 48),
                      xreg = X,include.mean = FALSE)
ar2Sar1x.reg <- arima(log(win1.agr$CO2), order = c(2, 0, 0),
                      seasonal=list(order = c(1, 0, 0), period = 48),
                      xreg = X, include.mean = FALSE)
ar2x.reg <- arima(log(win1.agr$CO2),order = c(2, 0, 0),
                  xreg = X, include.mean = FALSE)

c(AIC(ar2Sma1x.reg), AIC(ar2Sar1x.reg), AIC(ar2x.reg))

Q <- 2 * (logLik(ar2Sar1x.reg)-logLik(ar2x.reg))
1 - pchisq(Q, df = 1)

##################################################
## Section 8.6
##################################################

## cpgram(x)

##################################################
## Example 8.18


e0 <- residuals(fitCO2.1)
e <- residuals(ar2Sar1x.reg)
e.sign0 <- sign(e0)
e.sign <- sign(e)
n <- length(e)

sig.shift0 <- sum(abs(e.sign0[-1] != e.sign0[-n]))
sig.shift <- sum(abs(e.sign[-1] != e.sign[-n]))
zc <- (c(sig.shift0, sig.shift)  - (n - 1) / 2) / 
    sqrt((n -1 ) / 4)
zc


e0 <- residuals(fitCO2.1)
e <- residuals(ar2Sar1x.reg)
e0.acf <- acf(e0, lag.max = 48, plot = FALSE)$acf
e.acf <- acf(e, lag.max = 48, plot = FALSE)$acf
Q <- c(n * sum(e0.acf^2), n * sum(e.acf^2))

p <- c(length(coef(fitCO2.1)),  length(coef(ar2x.reg)))
m <- length(e.acf)
(1 - pchisq(Q, df = m - p)) 

## 
## cpgram(e)


I.fun <- function(nu,e){
    n <- length(e)
    t <- 1:n
    (sum(e*cos(2*pi*nu*t))^2+sum(e*sin(2*pi*nu*t))^2)/n
}


nu <- (1:floor(length(e)/2))/length(e)

I.hat0 <- sapply(nu,I.fun,e0)
I.hat <- sapply(nu,I.fun,e)


par(mfrow=c(1,2),bg="white")
max.l <- 3*48
acf0 <- acf(e0,lag.max=max.l,plot=FALSE)
acf1 <- acf(e,lag.max=max.l,plot=FALSE)
plot((0:(max.l)-0.2)/2,c(acf0$acf),type="h",ylab="ACF",xlab="lag (Hours)",
     col=gray(0.4))
lines((0:(max.l)+0.2)/2,c(acf1$acf),type="h",col=1,lwd=2)
abline(a=2/sqrt(n),b=0,col=gray(0.5),lty=2)
abline(a=-2/sqrt(n),b=0,col=gray(0.5),lty=2)
abline(a=0,b=0,col=1,lty=1)
legend("topright",legend=c("GLM-residuals","ARMAX-residuals"),lty=1,
       col=gray(c(0.4,0)),
       lwd=2)

plot(nu,cumsum(I.hat0)/sum(I.hat0),col=gray(0.4),lwd=2,type="l",ylab=expression(C(nu[i])),
     xlab=expression(nu[i]))
lines(nu,cumsum(I.hat)/sum(I.hat),col=1,lwd=2)
abline(a=0,b=2,lty=2,col=gray(0.5))
abline(a=1.36/sqrt((n-1)/2),b=2,lty=2,col=gray(0.5))
abline(a=-1.36/sqrt((n-1)/2),b=2,lty=2,col=gray(0.5))



e0 <- residuals(fitCO2.1)
e <- residuals(ar2Sar1x.reg)
 library(forecast)
par(mfrow=c(1,2),bg="white")
qqnorm(e/sd(e),pch=19,col=1,cex=0.5)
abline(a=0,b=1)
qq0 <- qqnorm(e0/sd(e0),plot=FALSE)
points(qq0$x,qq0$y,pch=4,col=gray(0.5),cex=0.5)
legend("topleft",c("GLM-residuals","ARMAX-residuals"),col=gray(c(0.5,0)),pch=c(4,19))
plot(fitted(ar2Sma1x.reg),e/sd(e),pch=19,xlab=expression(hat(Y)[t]),ylab="Standardized ARMAX residuals",cex=0.5)


e0 <- residuals(fitCO2.1)
e <- residuals(ar2Sar1x.reg)
par(mfrow=c(1,2),bg="white")
plot(win1.agr$Time,e/sd(e),pch=19,type="l",xlab="Time [Days]")

plot(win1.agr$Time,log(win1.agr$CO2C),pch=19,type="l",xlab="Time [Days]",ylab=expression(log(CO[2])))


## library(forecast)

i1 <- which(residuals(ar2Sar1x.reg)< -0.45)
i2 <- which(residuals(ar2Sar1x.reg)> 0.55)

lty=c(1,2,1)
col.tmp <- gray(c(0,0.25))
par(mfrow=c(1,2),bg="white")
matplot(win1.agr$Time[i1+c(-48:48)],cbind(log(win1.agr$CO2C),
                                          win1.agr$WindowClosed+6)[i1+c(-48:48),
                                                                   ],ylim=c(6,8),
        col=col.tmp,xlab="Time [Days]",
        ylab=expression(log(CO[2])),lwd=2,lty=lty,type=c("l","l","l"))
lines(win1.agr$Time[i1]*c(1,1),c(0,10),lty=4,col=gray(0.5),lwd=2)
matplot(win1.agr$Time[i2+c(-48:48)],cbind(log(win1.agr$CO2C),
                                          win1.agr$WindowClosed+6)[i2+c(-48:48), ],type="l",ylim=c(6,8),
        col=col.tmp,xlab="Time [Days]",ylab=expression(log(CO[2])),lwd=2,lty=lty)
lines(win1.agr$Time[i2]*c(1,1),c(0,10),lty=4,col=gray(0.5),lwd=2)
legend("topright",legend=c(expression(CO[2]),"Win. Pos."),lty=1:2,col=col.tmp)


##################################################
## Section 8.7
##################################################

##################################################
## Example 8.19

##################################################
## Figure 8.2
phi <- 0.9;theta<-0.5;sigma2 <- 0.1
vs <- sigma2*(1+theta^2+2*phi*theta)/(1-phi^2)


n <- 20
y <- numeric(n)
v <- numeric(n+1)
y[1] <- 3*sqrt(vs)
e <- numeric(n)
e[1] <- 3*sqrt(sigma2)
y[1] <- y[1] + e[1]
ys <- y
es <- e
v[2] <- sigma2
set.seed(1452)
for(i in 2:n){
    y[i] <- phi * y[i-1] + theta *  e[i-1]
    v[i+1] <- phi^2 * v[i] + sigma2 * (1 + theta^2 + 2 * theta * phi)
    es[i] <- rnorm(1,sd=sqrt(sigma2))
    ys[i] <- phi * ys[i-1] + theta *  es[i-1] + es[i]
}

par(mfrow=c(1,2),bg="white")
col.tmp <- gray(c(0.25,0))
matplot(0:(n-1),cbind(ys,y),type="b",pch=c(4,19),col=col.tmp,lty=1,cex=0.5,ylab="",xlab="Time",
        ylim=c(-1,5))
polygon(c(0:(n-1),rev(0:(n-1))),c(y+2*sqrt(v[-(n+1)]),rev(y-2*sqrt(v[-(n+1)]))),col=gray(0.75),
                                  border=FALSE)
matlines(0:(n-1),cbind(ys,y),type="b",pch=c(4,19),col=col.tmp,lty=1,cex=0.5,lwd=2)
legend("topright",legend=c(expression(hat(Y)["t|0"]),
                           expression(Y[t])
                           ),lty=1,pch=c(4,19),col=col.tmp,lwd=2)

abline(a=0,b=0,col=gray(0.5),lwd=2,lty=4)
plot(v,type="b",pch=19,ylim=c(0,vs),cex=0.5,ylab="Variance",xlab="Time",lwd=2)
abline(a=vs,b=0,col=gray(0.5),lty=4,lwd=2)
legend("bottomright",legend=c(expression(paste(V,"[",Y["t|0"],"]")),
                              expression(paste(V,"[",Y["t"],"]"))
                           ),lty=c(1,4),pch=c(19,NA),col=gray(0,0.5),lwd=2)



##################################################
## Example 8.20

X <- model.matrix(fitCO2.1)
n <- dim(X)[1]
Xnew <- X[(n-48+1):n, ]
pred <- predict(ar2Sar1x.reg, newxreg = Xnew, n.ahead = 48, se.fit = TRUE)


summary(fitCO2.1)$sigma

##################################################
## Figure 8.3


fit.val <- fitted(ar2Sar1x.reg)[(n-48+1):n]
hour <- win1.agr$Hour[(n-48+1):n]

par(mfrow=c(1,2),mar=c(5,5,1,2))
plot((-47:48)/2,c(fit.val*NA,pred$pred),type="b",ylim=c(5.7,7.7),pch=19,xlim=c(-7,7),cex=0.5,
     ylab=expression(log(CO[2])),xlab="Time [Hours]",axes=FALSE,col=gray(0.25))
axis(1,at = -7:7,labels=c("","","","","","","","","","","","","","",""))
axis(1,at = c(-6,-2,2,6),labels=c("n-6","n-2","n+2","n+6"),cex=.5)
polygon(c(1:48,48:1)/2,c(pred$pred+2*pred$se,rev(pred$pred-2*pred$se)),border=FALSE,col=gray(0.75))
axis(2);box()
axis(4,at=5.7+c(0,1),labels=c(0,1))

lines((-47:0)/2,log(win1.agr$CO2C)[(n-48+1):n],col=1,type="b",pch=4,cex=0.5)
lines((-47:48)/2,c(Xnew[ ,2],Xnew[ ,2])+5.7,col=gray(0.5),type="l",lwd=3,lty=4)

statval <-as.vector( Xnew %*% coef(ar2Sma1x.reg)[-(1:3)])
lines((-47:48)/2,c(statval,statval),col=gray(0.5),type="l",lwd=5)
lines(c(0,0),c(0,19),col=gray(0.75),lty=3,lwd=2)
matlines((-47:48)/2,c(fit.val*NA,pred$pred),type="b",pch=19,cex=0.5,col=gray(0.5))
legend("topleft",legend=c(expression(Y["t"]),expression(hat(Y)["t|n"]),
                          expression(hat(Y)["t|0"]),"Win. Pos."),lty=c(rep(1,3),4),
       col=gray(c(0,0.5,0.5,0.5)),
       lwd=c(2,2,5,2),pch=c(4,19,NA,NA),cex=0.65)


plot((1:48)/2,pred$se,type="b",xlim=c(0,7),ylim=c(0,0.23),pch=19,cex=0.5,xlab="Time [Hours]",ylab="Standard deviation")
abline(a=summary(fitCO2.1)$sigma,b=0,col=gray(0.5),lwd=2,lty=4)


##################################################
## Section 8.8
##################################################
