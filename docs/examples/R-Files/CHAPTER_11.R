library(numDeriv)
##################################################
## Data (see Chapter 8)
win1 <- read.table("/Users/madshaakonsson/Desktop/5. Semester/special/hmm-code/data/b1.csv",sep=";",header=TRUE)

win1 <- win1[!is.na(win1$WindowClosed), ]
win1 <- win1[win1$Room=="Bedroom", ]


win1 <- win1[1:5000, ]## Only a subset of the data is considered

win1$HalfHour <- round((win1$day+win1$Time)*24*2)

win1.agr <- aggregate(cbind(CO2C,WindowClosed,Time,Day,Month)~
                          HalfHour, data=win1,mean)
win1.agr$HalfHour <-  (win1.agr$HalfHour)%%48

win1.agr$Hour <- win1.agr$HalfHour/2
win1.agr$Time <-  (1:length(win1.agr$Hour))/2/24

##################################################
## Chap 09 functions
source("/Users/madshaakonsson/Desktop/5. Semester/special/hmm-code/examples/R-Files/functions/chap09funs.R")

##################################################
## Example 11.1

##################################################
## Section 11.1
##################################################

##################################################
## Example 11.2

## End example 
##################################################

HMM.filter <- function(Gamma, g){
    ## Initialize matrices and vectors
    m <- dim(Gamma)[1]
    n <- dim(g)[1]
    Ut <- matrix(ncol = m, nrow = n)
    Utt <- matrix(ncol = m, nrow = n)
    ft <-numeric(n)
    
    ## Inital state and distribution
    Ut[1, ] <- stat.dist(Gamma)
    ft[1] <- sum(Ut[1, ] %*% diag(g[1, ]))
    Utt[1, ] <- Ut[1, ] %*% diag(g[1, ]) / ft[1]

    ## Recursive filtering
    for(i in 2:n){
        Ut[i, ] <- Utt[i-1, ] %*% Gamma
        v <- Ut[i, ] %*% diag(g[i, ])
        ft[i] <- sum(v)
        Utt[i, ] <- v / ft[i]
    }
    ## Return state predictions and reconstruction
    return(list(Ut = Ut, Utt = Utt, ft = ft))
}

##################################################
## Example 11.3
win1 <- read.table("/Users/madshaakonsson/Desktop/5. Semester/special/hmm-code/data/b1.csv",sep=";",header=TRUE)
win1 <- win1[!is.na(win1$WindowClosed), ]

par(bg="white")
y <- win1.agr$CO2C
n <- length(y)
plot(win1.agr$Time,y,type="l",xlab="Time [days]",ylab ="CO2 [ppm]")

Gamma <- matrix(0, 3, 3)
Gamma[1, ] <- c(0.95, 0.05, 0)
Gamma[2, ] <- c(0.05, 0.9, 0.05)
Gamma[3, ] <- c(0, 0.05, 0.95)

normal.obs <- function(mu, sigma, y){
    n <-  length(y)
    m <- length(mu)
    g <- matrix(NA, ncol = m, nrow = n)
    for(i in 1:m){
        g[ ,i] <- dnorm(y, mean = mu[i], sd = sigma[i])
    }
    g
}

mu <- quantile(y, probs = c(0.25, 0.5, 0.75))
sigma <- rep(150, 3)

g <- normal.obs(mu,sigma,y)
filt <- HMM.filter(Gamma, g)

par(mfrow=c(1,1),mar=c(4,5,4,2),bg="white")
col.tmp<- gray(c(0,0.5,0.9))
matplot(win1.agr$Time+win1.agr$Hour[1]/24,filt$Utt,type="l",col="white",xlab="Time [days]",ylab=expression(hat(u)["t|t"]))
polygon(c(win1.agr$Time+win1.agr$Hour[1]/24,rev(win1.agr$Time+win1.agr$Hour[1]/24)),
          c(filt$Utt[ ,1],filt$Utt[ ,1]*0),col=col.tmp[1],border=FALSE)
polygon(c(win1.agr$Time+win1.agr$Hour[1]/24,rev(win1.agr$Time+win1.agr$Hour[1]/24)),
          c(filt$Utt[ ,1],rev(filt$Utt[ ,1]+filt$Utt[ ,2])),col=col.tmp[2],border=FALSE)
polygon(c(win1.agr$Time+win1.agr$Hour[1]/24,rev(win1.agr$Time+win1.agr$Hour[1]/24)),
          c(rep(1,n),rev(filt$Utt[ ,1]+filt$Utt[ ,2])),col=col.tmp[3],border=FALSE)


## End example 
##################################################

##################################################
## Section 11.2
##################################################

HMM.ll1 <- function(pars, y, ms, g.fun, full = FALSE){
    Gamma <- Markov.Inv.Link(pars[1 : (ms * (ms - 1))], ms)
    g.list <- g.fun(pars[- (1 : (ms * (ms - 1)))], y, ms)
    filt <- HMM.filter(Gamma, g.list$g)
    if(full){
        return(list(Gamma = Gamma, g.list = g.list, filt = filt))
    }
    - sum(log(filt$ft))
}

##################################################
## Example 11.4

g.fun1 <- function(pars, y, ms){
    ## Initialization of state depedent density and distribution
    n <- length(y)
    g <- matrix(NA, ncol = ms, nrow = n)
    G <- matrix(NA, ncol = ms, nrow = n)
    
    ## Parameters for state depedent distributions
    mu <- c(400, pars[1 : (ms - 1)])
    sigma2 <- exp(pars[ -c(1 : (ms - 1))])
    sigma <- sqrt(sigma2)

    ## Calculation of state depedent density and distribution
    for(i in 1 : ms){
        g[ ,i] <- dnorm(y, mean = mu[i] , sd = sigma[i])
        G[ ,i] <- pnorm(y, mean = mu[i] , sd = sigma[i])
    }
    ## Return results
    return(list(g = g, G = G, mu = mu, sigma = sigma))
 }

gamma.pars <- rep(-3,6)

Markov.Inv.Link(gamma.pars, 3)

pars <- c(gamma.pars, 800, 1000, rep(10, 3))
opt3s <- nlminb(pars, HMM.ll1, y = y, ms = 3, g.fun = g.fun1)

res3s <- HMM.ll1(opt3s$par, y, ms = 3, g.fun1, full = TRUE)

## Mean values
  res3s$g.list$mu
## standard deviation
  res3s$g.list$sigma
## Coefficient of variation
  res3s$g.list$sigma / res3s$g.list$mu

res3s$Gamma
stat.dist(res3s$Gamma)

gamma.pars <- c(-2,-10,-10,
                -2,-2,-10,
                -10,-2,-2,
                -10,-10,-2)
                
Markov.Inv.Link(gamma.pars, 4)

pars <- c(gamma.pars,550,800,1200,rep(10,4))
opt4s <- nlminb(pars,HMM.ll1,y=y,ms=4,g.fun=g.fun1)

- c(opt3s$objective, opt4s$objective)

res4s <- HMM.ll1(opt4s$par,y,m=4,g.fun1,full=TRUE)

tab <- cbind(mu=res4s$g.list$mu, sd= res4s$g.list$sigma, cv= res4s$g.list$sigma/res4s$g.list$mu)
rownames(tab) = c("State 1","State 2","State 3","State 4")
tab

res4s$Gamma
stat.dist(res4s$Gamma)

##################################################
## Section 11.3
##################################################

##################################################
## Section 11.4
##################################################

##################################################
## Example 11.5

##################################################
## Example 11.6

##################################################
## Example 11.7



g.funGamma <- function(pars, y, ms){
    n <- length(y)
    g <- matrix(NA,ncol = ms, nrow = n)      
    G <- matrix(NA,ncol = ms, nrow = n)
    
    mu <- c(400, pars[1 : (ms - 1)])
    cv <- exp(pars[(ms : (2 * ms - 1))])
    k <- 1 / cv^2
    theta <- mu  * cv^2
    
    for(i in 1:ms){
        g[ ,i] <- dgamma(y, shape = k[i], scale = theta[i])
        G[ ,i] <- pgamma(y, shape = k[i], scale = theta[i])
    }
    return(list(g=g,G=G,mu=mu,cv=cv))
}

pars <- c(opt4s$par)
pars[16:19] <- log(0.2)
opt4sGam <- nlminb(pars, HMM.ll1, y = y, ms = 4, g.fun = g.funGamma)
res4sGam <- HMM.ll1(opt4sGam$par,y,ms=4,g.funGamma,full=TRUE)

c(Gamma = -opt4sGam$objective, Gaus = -opt4s$objective)




IGam <- hessian(HMM.ll1,opt4sGam$par, y = y, ms = 4, g.fun = g.funGamma)
seGam <- sqrt(diag(solve(IGam)))

tab <- data.frame(log.cv=opt4sGam$par[16:19],
             se.log.cv=seGam[16:19], 
             cv=exp(opt4sGam$par[16:19]))
tab$cv.lower <- exp(tab$log.cv-2*tab$se.log.cv)
tab$cv.upper <- exp(tab$log.cv+2*tab$se.log.cv)
tab

par(bg="white")
xlim <- c(300,2000)
x <- seq(xlim[1],xlim[2],by=2)

k <- 1/res4sGam$g.list$cv^2 
theta <- res4sGam$g.list$cv^2 * res4sGam$g.list$mu
lty.tmp <- c(1,2,3,4)
plot(x, dgamma(x, shape = k[1], scale = theta[1]), type="l",lwd=2,ylab="Probability density function",xlab=expression(CO[2]))
i <- 2
lines(x, dgamma(x, shape = k[i], scale = theta[i]), type="l",col=1,lwd=2,lty=lty.tmp[i])
i <- 3
lines(x, dgamma(x, shape = k[i], scale = theta[i]), type="l",col=1,lwd=2,lty=lty.tmp[i])
i <- 4
lines(x, dgamma(x, shape = k[i], scale = theta[i]), type="l",col=1,lwd=2,lty=lty.tmp[i])

mu <- res4s$g.list$mu
sig <- res4s$g.list$sigma
i <- 1
lines(x, dnorm(x, mean = mu[i], sd = sig[i]), type="l",col=gray(0.5),lty=lty.tmp[i],lwd=3)
i <- 2
lines(x, dnorm(x, mean = mu[i], sd = sig[i]), type="l",col=gray(0.5),lty=lty.tmp[i],lwd=3)
i <- 3
lines(x, dnorm(x, mean = mu[i], sd = sig[i]), type="l",col=gray(0.5),lty=lty.tmp[i],lwd=3)
i <- 4
lines(x, dnorm(x, mean = mu[i], sd = sig[i]), type="l",col=gray(0.5),lty=lty.tmp[i],lwd=3)

i <- 1
lines(x, dgamma(x, shape = k[i], scale = theta[i]), type="l",col=1,lwd=2,lty=lty.tmp[i])
i <- 2
lines(x, dgamma(x, shape = k[i], scale = theta[i]), type="l",col=1,lwd=2,lty=lty.tmp[i])
i <- 3
lines(x, dgamma(x, shape = k[i], scale = theta[i]), type="l",col=1,lwd=2,lty=lty.tmp[i])
i <- 4
lines(x, dgamma(x, shape = k[i], scale = theta[i]), type="l",col=1,lwd=2,lty=lty.tmp[i])


legend("topright",lty=c(lty.tmp,1,1),col=gray(c(rep(0,5),0.5)),legend=c("state 1","state 2","state 3","state 4","Gamma","Gaus"),lwd=c(rep(2,5),3))


##################################################
## Figure 11.1
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


arrows(x0=col[1]+r,y0=row[1],x1=col[2]-r,y1=row[1],length=ar.len)



##################################################

polygon(col[2]+r*c(-1,1,1,-1),row[1]+ r*c(1,1,-1,-1))
text(col[2],row[1],expression(X["t"]),cex=text.cex)
arrows(x0=col[2]+r,y0=row[1],x1=col[3]-r,y1=row[1],length=ar.len)

arrows(x0=col[2],y0=row[1]-r,x1=col[2],y1=row[2]+r,length=ar.len)

##################################################


polygon(col[3]+r*c(-1,1,1,-1),row[1]+ r*c(1,1,-1,-1))
text(col[3],row[1],expression(X["t+1"]),cex=text.cex)
arrows(x0=col[3]+r,y0=row[1],x1=4,y1=row[1],length=ar.len)


##################################################

polygon(col[1]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
text(col[1],row[2],expression(Y["t-1"]),cex=text.cex)
arrows(x0=0,y0=row[2],x1=col[1]-r,y1=row[2],length=ar.len)


polygon(col[2]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
text(col[2],row[2],expression(Y["t"]),cex=text.cex)
arrows(x0=col[1]+r,y0=row[2],x1=col[2]-r,y1=row[2],length=ar.len)



polygon(col[3]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
text(col[3],row[2],expression(Y["t+1"]),cex=text.cex)

arrows(x0=col[2]+r,y0=row[2],x1=col[3]-r,y1=row[2],length=ar.len)
arrows(x0=col[3]+r,y0=row[2],x1=4,y1=row[2],length=ar.len)

arrows(x0=col[3],y0=row[1]-r,x1=col[3],y1=row[2]+r,length=ar.len)


##################################################
## Example 11.8



g.funAR1 <- function(pars, y, ms){
    n <- length(y)
    g <- matrix(NA,ncol = ms, nrow = n - 1)      
    G <- matrix(NA,ncol = ms, nrow = n - 1)
    
    mu <- c(400, pars[1 : (ms - 1)])
    sigma2 <- exp(pars[ (ms : (2 * ms - 1))])
    sigma <- sqrt(sigma2)
    phi <- pars[-c(1 : (2 * ms - 1))]
    for(i in 1 : ms){
        mu.t <- mu[i] + phi[i] * (y[-n] - mu[i]) #Error? 
        g[ ,i] <- dnorm(y[-1], mean = mu.t, sd = sigma[i])
        G[ ,i] <- pnorm(y[-1], mean = mu.t, sd = sigma[i])
    }
    return(list(g = g, G = G, mu = mu, sigma = sigma, phi = phi))
 }

pars <- c(opt4s$par, rep(0, 4))
opt4sAR1 <- nlminb(pars, HMM.ll1, y = y, ms = 4, g.fun = g.funAR1)
print(opt4sAR1$par)
######### PROFILE LIKELIHOOD 

profile <- function(phi0, opt){
    nll_p <- function(pars, y, ms, g.fun){
        pars[length(pars) - 3] <- phi0
        HMM.ll1(pars, y, ms, g.fun)
    } 
    pars <- opt$par
    opt <- nlminb(pars, nll_p, y = y, ms = 4, g.fun = g.funAR1)
    return(-opt$objective)
}

phi0 = seq(-0.5, 0.5, by = 0.01) 
log_likes <- sapply(phi0, profile, opt = opt4sAR1) 
plot(phi0, exp(log_likes), type = "l", xlab = expression(phi), ylab = "Log-likelihood")
 
#
#######
print("Hello")


c("HMM-Gaus"= -opt4s$objective, "HMM-Gam"=-opt4sGam$objective, "HMM-AR1"=-opt4sAR1$objective, "AR2"=  logLik(arima(y,order=c(2,0,0))))

ar = arima(y,order=c(2,0,0)) 
print(ar)
print(y)
res4sAR1 <- HMM.ll1(opt4sAR1$par,y,m=4,g.funAR1,full=TRUE)
partab <- cbind(mu=res4sAR1$g.list$mu, sigma=res4sAR1$g.list$sigma, phi=res4sAR1$g.list$phi)
partab


sqrt(res4sAR1$g.list$sigma^2 / (1 - res4sAR1$g.list$phi^2))

res4sAR1$Gamma
stat.dist(res4sAR1$Gamma)



##################################################
## Example 11.9

time <- win1.agr$HalfHour
X <- cbind(cos(2 * pi * time / 48), sin(2 * pi * time / 48))


HMM.filterGamma.t <- function(Gamma, g){
    m <- dim(Gamma)[2]
    n <- dim(g)[1]
    Ut <- matrix(ncol=m,nrow=n)
    Utt <- matrix(ncol=m,nrow=n)
    ft <-numeric(n)
    
    Ut[1, ] <- stat.dist(Gamma[1, , ])
    ft[1] <- sum(Ut[1, ] %*% diag(g[1, ]))
    Utt[1, ] <- Ut[1, ] %*% diag(g[1, ]) / ft[1]

    for(i in 2:n){
        Ut[i, ] <- Utt[i-1, ] %*% Gamma[i, , ]
        v <- Ut[i, ] %*% diag(g[i, ])
        ft[i] <- sum(v)
        Utt[i, ] <- v / ft[i]
    }
    return(list(Ut = Ut, Utt = Utt, ft = ft))
}

HMM.llGamReg <- function(pars, y, ms, g.fun,  X.reg, 
                         full = FALSE){
    p <- dim(X.reg)[2]
    n.parGam <- (ms * (ms - 1) + 4 * p)
    Gamma <- Markov.Inv.Link.Reg(pars[1 : n.parGam], ms, X.reg)
    g.list <- g.fun(pars[- (1 : n.parGam)], y, ms)
    filt <- HMM.filterGamma.t(Gamma, g = g.list$g)
    if(full){
        return(list(Gamma=Gamma, g.list = g.list, filt = filt))
    }
    - sum(log(filt$ft))
}


pars <- c(rep(0, 8), opt4sAR1$par)

opt4sAR1.Gx <- nlminb(pars, HMM.llGamReg, y = y, ms = 4, 
                      X.reg = X, g.fun = g.funAR1)

tab <- data.frame(df=c(length(opt4sAR1.Gx$par),length(opt4sAR1$par)),
    ll=c(opt4sAR1.Gx$objective,opt4sAR1$objective))
tab$Dll <- c(diff(tab$ll),NA)
tab$p.v <- pchisq(2 * tab$Dll, df = -diff(tab$df), lower.tail=FALSE)
tab                  

res4sAR1Gx <- HMM.llGamReg(pars=opt4sAR1.Gx$par,y,ms=4,g.fun=g.funAR1,X.reg=X,full=TRUE)
partab <- cbind(mu=res4sAR1Gx$g.list$mu, sigma=res4sAR1Gx$g.list$sigma, phi=res4sAR1Gx$g.list$phi)
 halfhour <- 0:48
X.sim <- cbind(cos(2 * pi * halfhour/48), sin(2 * pi * halfhour/48))
p <- 2
m <- 4
Gamma.t <- Markov.Inv.Link.Reg(opt4sAR1.Gx$par[1 : (m * (m - 1) + 4 * p)], m, X.sim)


 par(mfrow=c(1,1),bg="white")
col.tmp <- gray(seq(0,0.5,length=4))
 matplot(halfhour/2,Gamma.t[ ,1, ],type="l",ylim=c(0,1),lty=1:4,lwd=2,
         xlab="Hour of day",ylab =expression(p["1i"]),col=col.tmp)
legend("topleft",legend=c(expression(p["11"]),expression(p["12"]),
                          expression(p["13"]),expression(p["14"])),lty=1:4,
       col=col.tmp,lwd=2)
 

t <- rep(0:48,4000)
X.sim <- cbind(cos(2 * pi * t/48), sin(2 * pi *t/48))
 p <- dim(X.sim)[2]
 Gamma.sim <- Markov.Inv.Link.Reg(opt4sAR1.Gx$par[1 : (m * (m - 1) + 4 * p)], m, X.sim)
 y.sim <- sim.markovReg(Gamma.sim, 1)
 df.tmp <- data.frame(y=y.sim,t=t)
 I1 <- y.sim==1
 I2 <- y.sim==2
 I3 <- y.sim==3
 I4 <- y.sim==4
 df.tmp <- data.frame(I1=I1,I2=I2,I3=I3,I4=I4,t=t)
 agg <- aggregate(cbind(I1,I2,I3,I4)~t,data=df.tmp,mean)

agg2 <- aggregate(1-WindowClosed~HalfHour,data=win1.agr,mean)
par(mfrow=c(1,1),bg="white")

df.tmp <- data.frame(Utt=res4sAR1Gx$filt$Utt,HalfHour=win1.agr$HalfHour[-1])
agg.tmp <- aggregate(cbind(Utt.1,Utt.2,Utt.3,Utt.4)~HalfHour,data=df.tmp,mean)
matplot(agg.tmp[ ,1]/2,agg.tmp[ ,-1],type="l",lty=1:4,lwd=3,xlab="Hour of day",
        ylab="Observed average probability",col=gray(c(0,0.15,0.3,0.45)))
matlines(agg2[ ,1]/2,agg2[ ,-1],type="b",lty=2,col=gray(0.5),lwd=3,pch=19,cex=0.5) 

legend("topright",legend=c("P(X=1)","P(X=2)","P(X=3)","P(X=4)","Window open"),
       lty=c(1:4,2),col=gray(c(0,0.15,0.3,0.45,0.5)),cex=0.75,ncol=2,
       pch=c(rep(NA,4),19))



##################################################
## Example 11.10

H <- hessian(HMM.llGamReg, opt4sAR1.Gx$par,  y = y, 
             ms = 4, X.reg = X, g.fun = g.funAR1)

se <- sqrt(diag(solve(H)))
tab.mu <- cbind(c(400,opt4sAR1.Gx$par[21:23]),c(NA,se[21:23]))
row.names(tab.mu) <- c("mu1","mu2","mu3","mu4")
colnames(tab.mu) <- c("MLE","se")
par.gam <- opt4sAR1.Gx$par[1:20]
se.gam <- se[1:20]
tab.gam <- cbind(par.gam,se.gam)


par.beta <- par.gam[1:8]
se.beta <- se.gam[1:8]
tab.beta <- cbind(par.beta,se.beta)
row.names(tab.beta) <- c("beta11.cos","beta11.sin","beta22.cos","beta22.sin","beta33.cos","beta33.sin","beta44.cos","beta44.sin")
## colnames(tab.tau) <- c("MLE","se")
tab.beta

par.tau <- par.gam[-(1:8)]
se.tau <- se.gam[-(1:8)]
tab.tau <- cbind(par.tau,se.tau)
row.names(tab.tau) <- c("tau12","tau13","tau14","tau21","tau23","tau24","tau31","tau32","tau34","tau41","tau42","tau43")
colnames(tab.tau) <- c("MLE","se")
tab.tau

par.sig2 <- opt4sAR1.Gx$par[24:27]
se.sig2 <- se[24:27]
tab.sig2 <- cbind(par.sig2,se.sig2)
row.names(tab.sig2) <- c("lsig.sq1","lsig.sq2","lsig.sq3","lsig.sq4")
colnames(tab.sig2) <- c("MLE","se")
tab.sig2



par.phi <- opt4sAR1.Gx$par[28:31]
se.phi <- se[28:31]
tab.phi <- cbind(par.phi,se.phi)
row.names(tab.phi) <- c("phi1","phi2","phi3","phi4")
colnames(tab.phi) <- c("MLE","se")
tab.phi


sum(abs(sort(cov2cor(solve(H))))>0.9)

C <- abs((cov2cor(solve(H))))
C[is.na(C)]<-0
which(C>0.9 & C<1,arr.ind=TRUE)
pars <- opt4sAR1.Gx$par
names(pars) <- c("beta11.1","beta11.2","beta22.1","beta22.2","beta33.1","beta33.2","beta44.1","beta44.2",
                 "tau12","tau13","tau14","tau21","tau23","tau24","tau31","tau32","tau34",
                 "tau41","tau42","tau43","mu2","mu3","mu3","sig1","sig2","sig3","sig4",
                 "phi1","phi2","phi3","phi4")
pars[c(8,20)]
pars[c(15,16)]
pars[c(15,17)]
pars[c(16,17)]

C <- cov2cor(solve(H))
colnames(C)<- names(pars)
rownames(C)<- names(pars)
C[c(5,6,15:17),c(5,6,15:17)]

C[c(1,2,9:11),c(5,6,9:11)]
C[c(3,4,12:14),c(3,4,12:14)]
C[c(7,8,18:20),c(7,8,18:20)]


C[c(8,20),c(8,20)]

C[21:31,21:31]


tab.mu

tab.sig2

tab.phi

tab.beta

tab.tau

C[c(5,6,15:17),c(5,6,15:17)]

##################################################
## Section 11.5
##################################################

##################################################
## Figure 11.2


ps.res4 <- qnorm(rowSums(res4s$g.list$G * res4s$filt$Ut))

ps.res4Gam <- qnorm(rowSums(res4sGam$g.list$G * res4sGam$filt$Ut))

ps.res4AR1 <- qnorm(rowSums(res4sAR1$g.list$G * res4sAR1$filt$Ut))

ps.res4AR1Gx <- qnorm(rowSums(res4sAR1Gx$g.list$G * res4sAR1Gx$filt$Ut))


library(TSA)
cex <- 0.5
par(mfrow=c(5,3),mar=c(1,4,1,1),oma=c(5,3,0,0))
plot(win1.agr$Time,ps.res4,type="h",xlab="",axes=FALSE,ylab="Pseudo residuals")
box();axis(2)
mtext("HMM-Normal",side=2,line=5)

qqnorm(ps.res4,axes=FALSE,main="",ylab="Sample quantiles",xlab="",pch=19,cex=cex)
qqline(ps.res4)
box();axis(2)
acf(ps.res4,lag.max=2*48,axes=FALSE,xlab="",main="",drop.lag.0=FALSE,ci.col=gray(0.5))
box();axis(2)

plot(win1.agr$Time,ps.res4Gam,type="h",xlab="",axes=FALSE,ylab="Pseudo residuals")
mtext("HMM-Gamma",side=2,line=5)
box();axis(2)
qqnorm(ps.res4Gam,axes=FALSE,main="",ylab="Sample quantiles",xlab="",pch=19,cex=cex)
qqline(ps.res4Gam)
box();axis(2)
acf(ps.res4Gam,lag.max=2*48,axes=FALSE,xlab="",main="",drop.lag.0=FALSE)
box();axis(2)


plot(win1.agr$Time[-1],ps.res4AR1,type="h",xlab="",axes=FALSE,ylab="Pseudo residuals")
box();axis(2)
mtext("HMM-AR(1)",side=2,line=5)
qqnorm(ps.res4AR1,axes=FALSE,main="",ylab="Sample quantiles",xlab="",pch=19,cex=cex)
qqline(ps.res4AR1)
box();axis(2)
acf(ps.res4AR1,lag.max=2*48,axes=FALSE,xlab="",main="",drop.lag.0=FALSE,ci.col=gray(0.5))
box();axis(2)

plot(win1.agr$Time[-1],ps.res4AR1Gx,type="h",xlab="",axes=FALSE,ylab="Pseudo residuals")
mtext("Inhom. HMM-AR(1)",side=2,line=5)
box();axis(2)
qqnorm(ps.res4AR1Gx,axes=FALSE,main="",ylab="Sample quantiles",xlab="",pch=19,cex=cex)
qqline(ps.res4AR1Gx)
box();axis(2)
acf(ps.res4AR1Gx,lag.max=2*48,axes=FALSE,xlab="",main="",drop.lag.0=FALSE,ci.col=gray(0.5))
box();axis(2)


fit.arma <- arima(y,order=c(2,0,0))
arma.res <- residuals(fit.arma)/sqrt(fit.arma$sigma2)
plot(win1.agr$Time,arma.res,type="h",xlab="Time [days]",axes=FALSE,ylab="Standardized residuals")
mtext("AR(2)",side=2,line=5)
mtext("Time [Days]",side=1,line=4)
box();axis(2);axis(1)
qqnorm(arma.res,axes=FALSE,main="",ylab="Sample quantiles",pch=19,cex=cex)
qqline(arma.res)
box();axis(2);axis(1)
mtext("Theoretical quantiles",side=1,line=4)
acf(arma.res,lag.max=2*48,axes=FALSE,xlab="Lag [Hours]",main="",drop.lag.0=FALSE,
    ci.col=gray(0.5))
box();axis(2);axis(1,at=seq(0,96,by=12),labels=seq(0,96,by=12)/2)
mtext("Lag [Hours]",side=1,line=4)




##################################################
## Section 11.6
##################################################

##################################################
## Example 11.11

(y0 <- rev(y)[1])

n <- dim(res4sAR1Gx$filt$Utt)[1]
(u0 <- res4sAR1Gx$filt$Utt[n, ])

t0 <- 9
t <- t0 + 1:49
X <- cbind(cos(2 * pi * t / 48), sin(2 * pi * t / 48))
Gamma <- Markov.Inv.Link.Reg(opt4sAR1.Gx$par[1 : 20], 4, X)

x0 <- sample(1:4,prob=u0,size=1)
x <- sim.markovReg(Gamma,x0)

T <- length(t)
y.pred <- numeric(T + 1)
y.pred[1] <- y0
e <- rnorm(T,0,sd = res4sAR1Gx$g.list$sigma[x])
mu <- res4sAR1Gx$g.list$mu
phi <- res4sAR1Gx$g.list$phi
for(i in 2:(T+1)){
    y.pred[i] <- mu[x[i - 1]] + phi[x[i - 1]] * 
        (y.pred[i - 1] - mu[x[i - 1]]) + e[i - 1]
}

n <- dim(res4sAR1Gx$filt$Utt)[1]
u0 <- res4sAR1Gx$filt$Utt[n, ]
t0 <- win1.agr$HalfHour[n+1]
t <- t0 + 1:49
X <- cbind(cos(2*pi*t/48),sin(2*pi*t/48))
Gamma <- Markov.Inv.Link.Reg(opt4sAR1.Gx$par[1 : 20], 4, X)
set.seed(1455)
T <- length(t)
y.tmp <- numeric(T+1)
y.tmp[1] <- win1.agr$CO2C[n+1]
K <- 100000
Y <- matrix(ncol=K,nrow=T)
for(j in 1:K){
    x0 <- sample(1:4,prob=u0,size=1)
    x <- sim.markovReg(Gamma,x0)
    e <- rnorm(T,0,sd=res4sAR1Gx$g.list$sigma[x])
    mu <- res4sAR1Gx$g.list$mu
    for(i in 2:(T+1)){
        y.tmp[i] <- mu[x[i-1]]+res4sAR1Gx$g.list$phi[x[i-1]]*(y.tmp[i-1]-mu[x[i-1]]) + e[i-1]
    }
    Y[ ,j] <- y.tmp[-1]
}

n.bins <- 70
par(mfrow=c(2,2),bg="white")
hist(Y[1, ],xlim=range(Y),freq=FALSE,ylim=c(0,0.005),
     main=paste("Time of day= ",(t[1]/2)%%24,", Horizon 0.5h"),
     xlab=expression(paste(CO[2]," concentration [ppm]")),nclass=n.bins)
rug(mu,col=gray(0.5),lwd=3)
hist(Y[17, ],xlim=range(Y),freq=FALSE,ylim=c(0,0.005),main=paste("Time of day= ",(t[17]/2)%%24,", Horizon 8.5h"),
     xlab=expression(paste(CO[2]," concentration [ppm]")),nclass=n.bins)
rug(mu,col=gray(0.5),lwd=3)
hist(Y[33, ],xlim=range(Y),freq=FALSE,ylim=c(0,0.005),
     main=paste("Time of day= ",(t[33]/2)%%24,", Horizon 16.5h"),
     xlab=expression(paste(CO[2]," concentration [ppm]")),nclass=n.bins)
rug(mu,col=gray(0.5),lwd=3)
hist(Y[49, ],xlim=range(Y),freq=FALSE,ylim=c(0,0.005),
     main=paste("Time of day= ",(t[49]/2)%%24,", Horizon 24.5h"),
     xlab=expression(paste(CO[2]," concentration [ppm]")),nclass=n.bins)
rug(mu,col=gray(0.5),lwd=3)

##################################################
## Section 11.7
##################################################
