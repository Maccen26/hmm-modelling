library(KFAS)
library(glmmTMB)
library(numDeriv)
##################################################
## Section 10.1
##################################################

##################################################
## Figure 10.1
par(mar=c(0,0,0,0),oma=c(0,0,0,0),bg="white")
r <- 0.25
text.cex <- 2
ar.len <- 0.1
row <- c(2,1)
col<-c(1,2,3)
plot(1,1,xlim=c(0,4),ylim=c(0.5,2.5),col="white",axes=FALSE,ylab="",xlab="")
polygon(col[1]+r*c(-1,1,1,-1),row[1]+ r*c(1,1,-1,-1))
text(col[1],row[1],expression(X["t-1"]),cex=text.cex)
arrows(x0=0,y0=row[1],x1=col[1]-r,y1=row[1],length=ar.len)

arrows(x0=col[1],y0=row[1]-r,x1=col[1],y1=row[2]+r,length=ar.len)

text(col[1]-2*r,mean(row),expression(e["t-1"]),cex=text.cex)
arrows(x0=col[1]-1.5*r,y0=mean(row),x1=col[1],y1=mean(row),length=ar.len)


arrows(x0=col[1]+r,y0=row[1],x1=col[2]-r,y1=row[1],length=ar.len)
arrows(x0=mean(col[1:2]),y0=row[1]+r,x1=mean(col[1:2]),y1=row[1],length=ar.len)

text(mean(col[1:2]),row[1]+1.5*r,expression(paste(epsilon["t"],", ",u["t"])),cex=text.cex)



##################################################

polygon(col[2]+r*c(-1,1,1,-1),row[1]+ r*c(1,1,-1,-1))
text(col[2],row[1],expression(X["t"]),cex=text.cex)
arrows(x0=col[2]+r,y0=row[1],x1=col[3]-r,y1=row[1],length=ar.len)

arrows(x0=col[2],y0=row[1]-r,x1=col[2],y1=row[2]+r,length=ar.len)

text(col[1]-2*r,mean(row),expression(e["t-1"]),cex=text.cex)

##################################################


polygon(col[3]+r*c(-1,1,1,-1),row[1]+ r*c(1,1,-1,-1))
text(col[3],row[1],expression(X["t+1"]),cex=text.cex)
arrows(x0=col[3]+r,y0=row[1],x1=4,y1=row[1],length=ar.len)

arrows(x0=col[3],y0=row[1]-r,x1=col[3],y1=row[2]+r,length=ar.len)

##################################################

polygon(col[1]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
text(col[1],row[2],expression(Y["t-1"]),cex=text.cex)


polygon(col[2]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
text(col[2],row[2],expression(Y["t"]),cex=text.cex)


polygon(col[3]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
text(col[3],row[2],expression(Y["t+1"]),cex=text.cex)


##################################################
## Example 10.1

win1 <- read.table("data/b1.csv",sep=";",header=TRUE)
win1 <- win1[!is.na(win1$WindowClosed), ]

win1B <- win1[win1$Room=="Bedroom", ]
win1L <- win1[win1$Room=="Livingroom", ]
win1B <- win1B[1:5000, ]
win1L <- win1L[1:5000, ]
RdayB <- win1B$Time + win1B$Day -  18 + (win1B$Month - 3) *31
RdayL <- win1L$Time + win1L$Day -  18 + (win1L$Month - 3) *31
win1B.dat <- cbind(RdayB, win1B[ ,c("Time","Day", "Month","WindowClosed","Hour",
                                "OutdoorTemp","Wind","TempC","CO2C")])
win1L.dat <- cbind(RdayL, win1L[ ,c("Time","Day", "Month","WindowClosed","Hour",
                                "OutdoorTemp","Wind","TempC","CO2C")])

Dt <- 10/60/24
Rday <- seq(min(win1B.dat$Rday),max(win1B.dat$Rday),by=Dt)

approxfunB.Time <- approxfun(win1B.dat$Rday,win1B.dat$Time)
approxfunB.Day <- approxfun(win1B.dat$Rday,win1B.dat$Day)
approxfunB.Month <- approxfun(win1B.dat$Rday,win1B.dat$Month)
approxfunB.WindowClosed <- approxfun(win1B.dat$Rday,win1B.dat$WindowClosed)
approxfunB.Hour <- approxfun(win1B.dat$Rday,win1B.dat$Hour)
approxfunB.WeekDay <- approxfun(win1B.dat$Rday,win1B.dat$WeekDay)
approxfunB.OutdoorTemp <- approxfun(win1B.dat$Rday,win1B.dat$OutdoorTemp)
approxfunB.Wind <- approxfun(win1B.dat$Rday,win1B.dat$Wind)
approxfunB.TempC <- approxfun(win1B.dat$Rday,win1B.dat$TempC)
approxfunB.CO2C <- approxfun(win1B.dat$Rday,win1B.dat$CO2C)


approxfunL.Time <- approxfun(win1L.dat$Rday,win1L.dat$Time)
approxfunL.Day <- approxfun(win1L.dat$Rday,win1L.dat$Day)
approxfunL.Month <- approxfun(win1L.dat$Rday,win1L.dat$Month)
approxfunL.WindowClosed <- approxfun(win1L.dat$Rday,win1L.dat$WindowClosed)
approxfunL.Hour <- approxfun(win1L.dat$Rday,win1L.dat$Hour)
approxfunL.WeekDay <- approxfun(win1L.dat$Rday,win1L.dat$WeekDay)
approxfunL.OutdoorTemp <- approxfun(win1L.dat$Rday,win1L.dat$OutdoorTemp)
approxfunL.Wind <- approxfun(win1L.dat$Rday,win1L.dat$Wind)
approxfunL.TempC <- approxfun(win1L.dat$Rday,win1L.dat$TempC)
approxfunL.CO2C <- approxfun(win1L.dat$Rday,win1L.dat$CO2C)


win.datSS <- data.frame(Rday=Rday,Time=approxfunB.Time(Rday),
                       Day=round(approxfunB.Day(Rday)),
                       Month=round(approxfunB.Month(Rday)),
                       WindowClosedB=round(approxfunB.WindowClosed(Rday)),
                       WindowClosedL=round(approxfunL.WindowClosed(Rday)),
                       Hour=round(approxfunB.Hour(Rday)),
                       WeekDay=round(approxfunB.WeekDay(Rday)),
                       CO2CB=approxfunB.CO2C(Rday),
                       CO2CL=approxfunL.CO2C(Rday))

par(mfrow=c(2,1),oma=c(0,1,0,1),mar=c(2,2,2,2),bg="white")
matplot(win.datSS$Rday,cbind(win.datSS$CO2CB),type="l",lty=1,lwd=2,ylim=c(400,2700),axes=FALSE,
        xlab="",ylab=expression(paste(CO[2]," [ppm]")),main="Bedroom")
lines(win.datSS$Rday,win.datSS$WindowClosedB*800+1700,col=gray(0.5),lwd=2)
axis(2)
axis(4,at=c(1700,2500),labels=c(0,1))
box()
matplot(win.datSS$Rday,cbind(win.datSS$CO2CL),type="l",lty=1,lwd=2,ylim=c(400,2700),
        xlab="Time [days]",ylab=expression(paste(CO[2]," [ppm]")),main="Livingroom")
lines(win.datSS$Rday,win.datSS$WindowClosedL*800+1700,col=gray(0.5),lwd=2)
axis(4,at=c(1700,2500),labels=c(0,1))
box()


##################################################
## Section 10.2
##################################################

##################################################
## Example 10.2

##################################################
## Example 10.3

##################################################
## Section 10.3
##################################################

##################################################
## Example 10.4

##################################################
## Example 10.5

##################################################
## Section 10.4
##################################################

##################################################
## Section 10.5
##################################################

##################################################
## Example 10.6

##################################################
## Example 10.7

phi1 <- 0.9; phi2 <- -0.3; theta <- 0.5; sigma <- 1

y <- c(1.73, 2.82,  4.12, 2.69, -1.24, -2.29, -0.31)

A <- matrix(0,ncol=2,nrow=2)
A[1,2] <- 1
A[ ,1] <- c(phi1, phi2)
G <- c(1, theta)
C <- c(1,0)

X.hat <- c(0,1)
Sig.x <- matrix(solve(diag(4) - kronecker(A, A)) %*% 
                as.vector(G %*% t(G)), ncol = 2)

(Sig.y <- C %*% Sig.x %*% C)

X.hat <- X.hat + Sig.x %*% C %*% solve(Sig.y) %*% 
    (y[1] - C %*% X.hat)
Sig.x <- Sig.x - Sig.x %*% C %*% solve(Sig.y) %*% C %*% Sig.x

## prediction
X.hat <- A %*% X.hat
Sig.x <- A %*% Sig.x %*% t(A)+ sigma * G %*% t(G)

(Sig.y <- C %*% Sig.x %*% C)



##################################################
## Section 10.6
##################################################

##################################################
## Example 10.8

##################################################
## Example 10.9

n <- dim(win.datSS)[1]
Zt <- array(0, dim=c(1,3,n))
Zt[1,2, ] <- 1

Qt <- diag(2); Qt[1,1] <- 0;Qt[2,2] <- 1
Tt <- matrix(c(1, 0, 0, 0, 0.9, -0.1, 0, 1, 0), 3, 3)
Rt <- matrix(c(0,0,0,0,1, 0), 3, 2)
Ht <- matrix(0, 1, 1)

a1 <- matrix( 1, 3 , 1)
P1 <- diag(3) * 0.05
P1[1,1] <-0

model_gaussianReg <- SSModel(log(win.datSS$CO2CB) ~ -1 +
                                 SSMcustom(Z = Zt, T = Tt, 
                                           R = Rt, Q = Qt, 
                                           a1 = a1, P1 = P1), H = Ht)

KFASnll1 <- function(pars, model, X, full=FALSE) {
  model$T[2, 2, ] <- pars[1]
  model$T[3, 2, ] <- pars[2]
  model$Q[2, 2, 1] <- exp(pars[3])
  model$a1[-1, 1] <- pars[4:5]
  model$Z[1, 1, ] <- X %*% pars[6:7]
  if(full){model}
  -logLik(model)
}

X <- cbind(1,win.datSS$WindowClosedB)
optKFAS <- nlminb(c(0.9, -0.4, 0, 0, 0, 0, 0), KFASnll1,
                  model = model_gaussianReg, X = X)

fitArx <- arima(log(win.datSS$CO2CB), order = c(2, 0, 0),
                include.mean = TRUE, 
                xreg = win.datSS$WindowClosedB)
c(-optKFAS$objective,logLik(fitArx))

##################################################
## Example 10.10

n <- dim(win.datSS)[1]
Zt <- array(0, dim=c(1,3,n))
Zt[1,2, ] <- 1

Qt <- array(0, dim=c(2,2,n)); Qt[2,2, ] <- 1
Tt <- array(0, dim=c(3,3,n))
Tt[1,1, ] <- 1
Tt[2,3, ] <- 1
Rt <- array(0, dim=c(3,2,n)) 
Rt[2,2, ] <- 1
Ht <- matrix(0, 1, 1)

a1 <- matrix( 1, 3 , 1)
P1 <- diag(3) * 0.05
P1[1,1] <-0

model_gaussianRegT <- SSModel(log(win.datSS$CO2CB) ~ -1 +
                                 SSMcustom(Z = Zt, T = Tt, 
                                           R = Rt, Q = Qt, 
                                           a1 = a1, P1 = P1), H = Ht)

KFASnll21T <- function(pars, model, X, full=FALSE) {    
    model$T[2,2, ] <- X %*% pars[1:2]
    model$T[3,2, ] <- X %*% pars[3:4]
    eta <- X %*% pars[5:6]
    model$R[3,2, ] <- 2 * exp(eta) / (1 + exp(eta)) - 1
    model$Q[2,2, ] <- exp(X %*% pars[7:8])
    model$a1[-1,1] <- pars[9:10]
    model$Z[1,1, ] <- X %*% pars[11:12]
    if(full){return(model)}
    -logLik(model)
}

pars <- c(0.9,0, -0.2, 0 , 0, 0, 0, 0, 0, 0, 0, 0)
optKFAS21Time <- nlminb(pars, KFASnll21T,
                  model = model_gaussianRegT, X = X)

H <- hessian(KFASnll21T, optKFAS21Time$par, 
             model = model_gaussianRegT, X = X)
VarTheta <- solve(H)

index <- c(2,4,6,8,12)
optKFAS21Time$par[index] / sqrt(diag(VarTheta)[index])

mod21T <- KFASnll21T(optKFAS21Time$par, model_gaussianRegT, X, 
                     full=TRUE)

c(mod21T$Z[1,1,X[ ,2]==0][1], mod21T$Z[1,1,X[ ,2]==1][1])

exp(c(mod21T$Z[1,1,X[ ,2]==0][1], mod21T$Z[1,1,X[ ,2]==1][1]))

mod21T$T[2:3,2,X[ ,2]==0][ ,1]

mod21T$T[2:3,2,X[ ,2]==1][ ,1]

c(mod21T$R[2:3, 2, X[ , 2] == 0][2, 1], 
  mod21T$R[2:3, 2, X[ ,2] == 1][2,1])

c(mod21T$Q[ , , X[ ,2] == 0][2 ,2 ,1],
  mod21T$Q[ , , X[ ,2] == 1][2 ,2 ,1])

A <- mod21T$T[2:3,2:3,X[ ,2]==0][ , ,1]
G <- mod21T$R[2:3, 2, X[ , 2] == 0][ , 1]
sigma.sq <- mod21T$Q[ , , X[ ,2] == 0][2 ,2 ,1]
stat.var.open <- matrix(
    sigma.sq * solve(diag(4) - kronecker(A, A)) %*% 
    as.vector(G %*% t(G)), 2, 2)

A <- mod21T$T[2:3,2:3,X[ ,2]==1][ , ,1]
G <- mod21T$R[2:3, 2, X[ , 2] == 1][ , 1]
sigma.sq <- mod21T$Q[ , , X[ ,2] == 1][2 ,2 ,1]
stat.var.closed <- matrix(
    sigma.sq * solve(diag(4) - kronecker(A, A)) %*% 
    as.vector(G%*%t(G)), 2,2)

sqrt(c(stat.var.open[1,1],stat.var.closed[1,1]))

##################################################
## Example 10.11

dat <- na.omit(win.datSS)
n <- dim(dat)[1]
resp <- as.matrix(dat[ ,c("CO2CB","CO2CL")])
X <- as.matrix(cbind(1,dat[ ,c("WindowClosedB","WindowClosedL")]))
Zt <- array(0, dim=c(2, 4, n))
Zt[1, 3, ] <- Zt[2, 4, ] <- 1
Qt <- matrix(0, 4, 4)
Tt <- matrix(0, 4, 4)
Tt[1, 1] <- Tt[2, 2] <- 1
Rt <- diag(4); Rt[1, 1] <- 0; Rt[2, 2] <- 0
a1 <- matrix(1, 4, 1)
P1 <- diag(4) * 0.4
P1[1, 1] <- P1[2, 2] <-0
Ht <- diag(2) * 0


model_multiVar <- SSModel(log(resp) ~ -1 +
                              SSMcustom(Z = Zt, T = Tt, R = Rt, Q = Qt, 
                                        a1 = a1, P1 = P1), H = Ht)


KFASnllMultivar0 <- function(pars, model, X, full=FALSE) {
    model$Z[1,1, ] <- X[ ,c(1:2)] %*% pars[1:2]
    model$Z[2,2, ] <- X[ ,c(1,3)] %*% pars[3:4]
    model$T[3,3, ] <- pars[5]
    model$T[4,4, ] <- pars[6]
    model$Q[3,3, ] <- exp(pars[7])
    model$Q[4,4, ] <- exp(pars[8])
    model$a1[-(1:2),1] <- pars[9:10]
    if(full){return(model)}
    -logLik(model)
}


pars <- c(6, 0, 6, 0, 0.9, 0.9, -3, -3, 0, 0)
optMultiKFAS0 <- nlminb(pars, KFASnllMultivar0, 
                        model = model_multiVar, X = X)

KFASnllMultivar1 <- function(pars, model, X, full=FALSE) {
    model$Z[1, 1, ] <- X[ ,c(1:2)] %*% pars[1:2]
    model$Z[2, 2, ] <- X[ ,c(1,3)] %*% pars[3:4]
    model$T[3, 3, ] <- pars[5]
    model$T[4, 4, ] <- pars[6]
    model$T[3, 4, ] <- pars[7]
    model$T[4, 3, ] <- pars[8]
    sigma1 <- exp(pars[9] / 2)
    sigma2 <- exp(pars[10] / 2)
    rho <- 2 * exp( pars[11]) / (1 + exp(pars[11])) - 1
    model$Q[3, 3, ] <- sigma1^2
    model$Q[4, 4, ] <- sigma2^2
    model$Q[4, 3, ] <- model$Q[3, 4, ] <- sigma1 * sigma2 * rho
    model$a1[-(1:2), 1] <- pars[12:13]
    if(full){return(model)}
    -logLik(model)
}




pars <- c(6, 0, 6, 0, 0.9, 0.9, 0, 0, -3, -3, 0, 0, 0)
optMultiKFAS1 <- nlminb(pars, KFASnllMultivar1, 
                        model = model_multiVar, X = X)


-optMultiKFAS1$objective + optMultiKFAS0$objective

modMult1 <- KFASnllMultivar1(optMultiKFAS1$par, 
                        model = model_multiVar, X, full = TRUE)
(T <- modMult1$T[3:4 , 3:4, 1])
eigen(T)$values

cov2cor(modMult1$Q[3:4 ,3:4 ,1])[1,2]

##################################################
## Section 10.7
##################################################

##################################################
## Example 10.12

n <- dim(win.datSS)[1]
Zt <- matrix(0, 1, 2)
Zt[1, 1] <- 1
Qt <- diag(2); Qt[1, 1] <- 0; 
Tt <- array(0, dim=c(2, 2, n))
Tt[2, 2, ] <- 1
Tt[1, 2, ] <- 1 / 6
Rt <- diag(2); Rt[1, 1] <- 0 
Ht <- matrix(25^2, 1, 1)

a1 <- matrix(1, 2, 1)
P1 <- diag(2) * 2200

model_gaussianRegPhys <- SSModel((win.datSS$CO2CB-400) ~ -1 +
                                 SSMcustom(Z = Zt, T = Tt, 
                                           R = Rt, Q = Qt, 
                                           a1 = a1, P1 = P1), H = Ht)


KFASnllPh <- function(pars, model, X, full=FALSE) {
    model$T[1,1, ] <- 1 - X %*% pars[1:2] / 6
    model$Q[2,2,1] <- exp(pars[3])
    model$a1[,1] <- pars[4:5]
    if(full){return(model)}
    -logLik(model)
}

X <- cbind(1, win.datSS$WindowClosedB)
optKFASPh <- nlminb(c(0, 0, 0, 0, 0), KFASnllPh,
                    model = model_gaussianRegPhys, X = X)
H <- hessian(KFASnllPh, optKFASPh$par,
             model = model_gaussianRegPhys, X = X)

tab <- cbind(theta = optKFASPh$par, se = sqrt(diag(solve(H))))
rownames(tab) <- c("beta0","beta1","q22","C0","E0")
tab


modPh <- KFASnllPh(optKFASPh$par, model_gaussianRegPhys, X, full=TRUE)
States <- KFS(modPh)


modPh$Q
optKFASPh$par
logLik(modPh)
modPh$R
modPh$P1
names(KFS(modPh))

I <- 1:4995
par(mfrow=c(1,2),bg="white")

dim(KFS(modPh)$alphahat)
matplot(win.datSS$Rday[I],cbind(KFS(modPh)$alphahat[-1,2],
              KFS(modPh)$alphahat[-1 ,2]+2*sqrt(KFS(modPh)$V[2,2,-1 ]),
              KFS(modPh)$alphahat[-1 ,2]-2*sqrt(KFS(modPh)$V[2,2,-1 ]))[I, ],type="l",col="white",
        ylab=expression(paste(E["r"], " [ppm/h]")),xlab="Time [Days]")

polygon(c(win.datSS$Rday[I],rev(win.datSS$Rday[I])),
        c(KFS(modPh)$alphahat[-1 ,2][I]+ 2*sqrt(KFS(modPh)$V[2,2,-1 ][I]),
          rev(KFS(modPh)$alphahat[-1 ,2][I]-2*sqrt(KFS(modPh)$V[2,2,-1 ][I]))),
        col=gray(0.5),
        border=FALSE)

lines(win.datSS$Rday[I],cbind(KFS(modPh)$alphahat[-1,2][I]))

abline(a=0,b=0,lwd=2,lty=2,col=gray(0.75))

I <- 1:(6*6*24)
matplot(win.datSS$Rday[I],cbind(KFS(modPh)$alphahat[-1,2],
              KFS(modPh)$alphahat[-1 ,2] + 2*sqrt(KFS(modPh)$V[2,2,-1 ]),
              KFS(modPh)$alphahat[-1 ,2] - 2*sqrt(KFS(modPh)$V[2,2,-1 ]))[I, ],type="l",col="white",ylab="",xlab="Time [Days]")

polygon(c(win.datSS$Rday[I],rev(win.datSS$Rday[I])),c(KFS(modPh)$alphahat[-1 ,2][I]+2*sqrt(KFS(modPh)$V[2,2,-1 ][I]),
                                                     rev(KFS(modPh)$alphahat[-1 ,2][I]-2*sqrt(KFS(modPh)$V[2,2,-1 ][I]))),
        col=gray(0.5),
        border=FALSE)

lines(win.datSS$Rday[I],cbind(KFS(modPh)$alphahat[-1,2][I]))

abline(a=0,b=0,lwd=2,lty=2,col=gray(0.75))
      
      
dim(win.datSS)
dim(KFS(modPh)$a)


##################################################
## Section 10.8
##################################################

##################################################
## Example 10.13

res <- rstandard(KFS(modPh))

par(mfrow=c(2,2),mar=c(4,4,1,4),bg="white")
plot(win.datSS$Rday,res,type="h",xlab="Time [days]")
qqnorm(res,main="")
qqline(res)
acf(res,lag.max=6*24,main="",ci.col=gray(0.5))
pacf(res,lag.max=6*24,main="",ci.col=gray(0.5))

##################################################
## Section 10.9
##################################################

##################################################
## Example 10.14

n <- dim(win.datSS)[1]
Zt <- matrix(1, 1,2)
Qt <- diag(2); Qt[1,1] <- 0; 
Tt <- matrix(0,2,2)
Tt[1,1] <- 1
Tt[2,2] <- 0.9
Rt <- diag(2); Rt[1,1] <- 0
a1 <- matrix(1, 2,1)
P1 <- matrix(0, 2,2); P1[2,2]<-1000

model_window <- SSModel(win.datSS$WindowClosedB ~ -1 +
                                 SSMcustom(Z = Zt, T = Tt, R = Rt,
                                           Q = Qt, a1 = a1, P1 = P1), 
                                 distribution = "binomial")

KFASnllBinom <- function(pars, model, full=FALSE) {
    model$Z[1,1, ] <- pars[1]
    model$T[2,2, ] <- pars[2]
    model$Q[2,2, ] <- exp(pars[3])
    model$a1[-1,1] <- pars[4]
    if(full){return(model)}
    -logLik(model)
}

optWindowKFAS1 <- nlminb(rep(0, 4), KFASnllBinom, 
                         model = model_window,
                         upper = c(100, 0.999, 100, 100),
                         lower = c(-100, -0.999, -100, -100))
mod <- KFASnllBinom(optWindowKFAS1$par, model_window, full = TRUE)



win.datSS$obs.no <- 1:length(win.datSS$WindowClosedB)
win.datSS$dwel <- 1
ar1.windowGlmmTMB <- glmmTMB(WindowClosedB ~ 1 + 
                             ar1(as.factor(obs.no) - 1 | dwel), 
                             data = win.datSS, family = binomial)

c(-optWindowKFAS1$objective, logLik(ar1.windowGlmmTMB))

summary(ar1.windowGlmmTMB)

(pars = c(beta0=optWindowKFAS1$par[1],phi=optWindowKFAS1$par[2],logsigma2=optWindowKFAS1$par[3],x0=optWindowKFAS1$par[4]))

exp(pars["logsigma2"]) / (1 - pars["phi"] ^ 2)

mod <- KFASnllBinom(optWindowKFAS1$par, model_window, full = TRUE)
filtered <- KFS(mod)

par(mfrow=c(2,1))
par(mar=c(4,5,1,4),oma=c(0,0,0,0),bg="white")
matplot(win.datSS$Rday,cbind(KFS(mod)$alphahat[,2]),type="l",
        xlab="Time [days]",ylab=expression(hat(x)["t|n"]))
matplot(win.datSS$Rday,win.datSS$WindowClosedB,type="l",
        xlab="Time [days]",ylab="Window Position")



##################################################
## Example 10.15

n <- dim(win.datSS)[1]
Zt <- matrix(c(0,1,0), 1,3)
Qt <- diag(2); Qt[1,1] <- 0; 
Qt <- diag(2); Qt[1,1] <- 0;Qt[2,2] <- 1
Tt <- matrix(c(1, 0, 0, 0, 0.9, -0.1, 0, 1, 0), 3, 3)
Rt <- matrix(c(0,0,0,0,1, 0), 3, 2)
Ht <- matrix(0, 1, 1)
a1 <- matrix( 1, 3 , 1)
P1 <- diag(3) * 100
P1[1,1] <-0
model_window21 <- SSModel(win.datSS$WindowClosedB ~ -1 +
                                 SSMcustom(Z = Zt, T = Tt, 
                                           R = Rt, Q = Qt, 
                                           a1 = a1, P1 = P1), 
                                 distribution="binomial")

KFASnllBinom2 <- function(pars, model, full=FALSE) {
    model$Z[1,1, ] <- pars[1]
    model$T[2,2, ] <- pars[2]
    model$T[3,2, ] <- pars[3]
    model$Q[2,2, ] <- exp(pars[4])
    model$a1[-1,1] <- pars[5:6]
    if(full){return(model)}
    -logLik(model)
}
pars <- rep(0,6)
pars[1:2] <- optWindowKFAS1$par[1:2]
pars[4] <- optWindowKFAS1$par[3]
pars[5] <- optWindowKFAS1$par[4]

optWindowKFAS2 <- nlminb(pars,KFASnllBinom2, model = model_window21,
                         upper = c(100, 2, 1, 10, 100, 100),
                         lower = c(-100, -1, -1, -10, -100, -100))

-c(optWindowKFAS2$objective, optWindowKFAS1$objective)

sprintf(optWindowKFAS2$par[2:3], fmt = '%#.3f')



##################################################
## Section 10.10
##################################################
