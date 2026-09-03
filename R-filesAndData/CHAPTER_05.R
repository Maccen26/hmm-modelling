library(lme4)
library(ggplot2)
library(glmmTMB)

## data from chapter 3
dfLMEx1 <- read.csv(file = "data/clothing.csv")
dfLMEx1$sex <- factor(dfLMEx1$sex)

##################################################
## Example 5.1

dfMEMEx1 <- read.csv(file = "data/clothing.csv")
 dfMEMEx1$subjId <- factor(dfMEMEx1$subjId)
 dfMEMEx1$tOut <- dfMEMEx1$tOut-mean(dfMEMEx1$tOut)
 fit.aov <- lm(clo ~ subjId, data = dfMEMEx1)

 fit0.lme <- lmer(clo  ~ 1 + (1| subjId), data = dfMEMEx1)

 summary(fit0.lme)

##################################################
## Section 5.1
##################################################

##################################################
## Example 5.2

  ## Data set including time of obs
  dfMEMEx1 <- dfMEMEx1[ ,-c(1,7)] 
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
  
  ## Aggregated data set
   dat.aggr <-  aggregate(.~day:subjId,dfMEMEx1,FUN="mean")[ ,c("day", "subjId","clo", "tOut","tInOp", "sex", "tOutCen", "tInOpCen")]
  dat.aggr$sex[dat.aggr$sex==1] <- "female"
  dat.aggr$sex[dat.aggr$sex==2] <- "male"
  dat.aggr$sex <- as.factor(dat.aggr$sex)
  
  ## Aggregated dataset for females
   dat.female <- dfMEMEx1[dfMEMEx1$sex==1, ]
   dat.aggrFemale <-  aggregate(.~day:subjId,dat.female,FUN="mean")
   I <- aggregate(.~day:subjId,dat.female,FUN="length")[ ,3] == 6
   dat.aggrFemale <- dat.aggrFemale[I, ]
    
   dat.aggrFemale <- dat.aggrFemale[dat.aggrFemale$obs.no==3.5, ]
   
   dat.agr2 <- aggregate(.~subjId,dat.aggrFemale,FUN="length")[ ,1:2]
   subjI2 <- dat.agr2$subjId[dat.agr2[ ,2]==3 ]
     
   i <- 1
     while(i <length(dat.aggrFemale$subjId)){
         if(sum(dat.aggrFemale$subjId[i]==subjI2)==0){
             dat.aggrFemale <- dat.aggrFemale[!(dat.aggrFemale$subjId[i] == dat.aggrFemale$subjId), ]
         }
         i<-i+1
     }
   dat.aggrFemale$subjId <- factor(dat.aggrFemale$subjId)
   dat.aggrFemale <- dat.aggrFemale[ ,c("day", "subjId","clo", "tOut", "tInOp", "sex",
                                                                "tOutCen", "tInOpCen")]
   dat.aggrFemale$sex[dat.aggrFemale$sex==1] <- "female"
   dat.aggrFemale$sex[dat.aggrFemale$sex==2] <- "male"
   dat.aggrFemale$sex <- as.factor(dat.aggrFemale$sex)
   
   levs <- levels(dat.aggrFemale$subjId)
   col <- numeric(length(dat.aggrFemale$subjId))
   for(i in 1:length(levs)){
       col[dat.aggrFemale$subjId==levs[i]] <- i
   }
  dfMEMEx1$subjId <- as.factor(dfMEMEx1$subjId)
  dfMEMEx1$day <- as.factor(dfMEMEx1$day)
  dfMEMEx1$obs.no <- as.factor(dfMEMEx1$obs.no)    
  dfMEMEx1$sex <- as.factor(dfMEMEx1$sex)
dat.aggr$subjId <- as.factor(dat.aggr$subjId)

summary(dfMEMEx1[ ,c(1:3)])

 summary(dat.aggrFemale)
  table(as.vector(by(as.numeric(dat.aggrFemale$sex),dat.aggrFemale$subjId,mean)))
  table(as.vector(by(as.numeric(dat.aggrFemale$day),dat.aggrFemale$subjId,max)))

 summary(dat.aggr)

##################################################
## Figure 5.1
par(mfrow=c(1,3))
x <- table(as.vector(by(as.numeric(dfMEMEx1$sex),dfMEMEx1$subjId,mean)))
names(x) <- c("Female","Male")
barplot(x,xlab="Sex")
barplot(table(as.vector(by(as.numeric(dfMEMEx1$day),dfMEMEx1$subjId,max))),
        xlab = "Number of days pr subject")
barplot(table(as.vector(by(as.numeric(dfMEMEx1$obs.no),dfMEMEx1$subjId:dfMEMEx1$day,max))),
        xlab="Number of observations per day")

##################################################
## Section 5.2
##################################################

##################################################
## Example 5.3

 table(dat.aggr$day)
 table(dat.aggrFemale$day)


anova(lm(clo ~ subjId, data = dat.aggrFemale))

##################################################
## Example 5.4

##################################################
## Figure 5.2

corrPlot <- function(C,main,by=1){
    n <- dim(C)[1]
    plot(0:n,0:n,axes=FALSE,col="white",ylab="",xlab="")
    mtext(main,side=3,line=-1,cex=1.5)
    mtext(seq(1,n,by=by),at=seq(0.5,n-0.5,by=by),side=1,line=-1)
    mtext(seq(1,n,by=by),at=seq(n-0.5,0.5,by=-by),side=2,line=-1)
    mtext("Obs number",at=n/2,side=1,line=1)
    for(i in 1:n){
        for(j in 1:n){
            if(C[i,j]!=0){ 
                border=1
                polygon(c(i-1,i,i,i-1),c(n-j+1,n-j+1,n-j,n-j),col=gray(1-C[i,j]),border=border)
            }
        }
    }
    lines(c(0,n),c(0,0))
    lines(c(n,n),c(0,n))
    lines(c(0,n),c(n,n))
    lines(c(0,0),c(n,0))
}


corrPlot2 <- function(C,main,ni=3,by){
    n <- dim(C)[1]
    plot(0:n,0:n,axes=FALSE,col="white",ylab="",xlab="")
    mtext(main,side=3,line=-1,cex=1.5)
    mtext(seq(1,n/ni,by=1),at=seq(ni/2,n-ni/2,by=by),side=1,line=-1)
    mtext(seq(1,n/ni,by=1),at=seq(n-ni/2,ni/2,by=-by),side=2,line=-1)
    mtext("Subject number",at=n/2,side=1,line=1)
    for(i in 1:n){
        for(j in 1:n){
            if(C[i,j]!=0){ 
                border=1
                polygon(c(i-1,i,i,i-1),c(n-j+1,n-j+1,n-j,n-j),col=gray(1-C[i,j]),border=border)
            }
        }
    }
    lines(c(0,n),c(0,0))
    lines(c(n,n),c(0,n))
    lines(c(0,n),c(n,n))
    lines(c(0,0),c(n,0))
}


fit0.lmer <- lmer(clo  ~ 1 + (1|subjId), data = dat.aggrFemale)
sigma <-  0.1078^2
sigma.u <- 0.1548^2
n <- 3
k <- 21
I <- diag(n)
J <- matrix(1,ncol=n,nrow=n)
V <- I*sigma^2 + J * sigma.u^2
C <- cov2cor(V)

par(mfrow=c(1,2),mar=c(2,0,2,0))
corrPlot(C^3,main=expression(V[i]),by=1)

V <- kronecker(diag(k),V)
C <- cov2cor(V)
corrPlot2(C^3,main=expression(V),ni=3,by=3)


##################################################
## Example 5.5

## no. of groups, obs. per group, and observations
n <- 3; 
k <- length(levels(dat.aggrFemale$subjId))
N <- n * k

## estimates
mu.hat <- mean(dat.aggrFemale$clo)
SSE <- sum((n - 1) * as.vector(by(dat.aggrFemale$clo, 
                                  dat.aggrFemale$subjId, var)))
SSB <- n * sum( (as.vector(by(dat.aggrFemale$clo, 
                              dat.aggrFemale$subjId, mean)) - 
                 mu.hat)^2)
sigma2.hat <- SSE / (k * (n-1))
sigmau2.hat <- SSB / N - SSE / (N * (n-1))

## report estimates
c(mu.hat, sqrt(sigma2.hat), sqrt(sigmau2.hat))

lmer(clo ~ 1 + (1 | subjId), data = dat.aggrFemale, 
     REML = FALSE)

##################################################
## Example 5.6

sigma2.u.m <- (SSB / (k - 1) - sigma2.hat) / n
round(sqrt(sigma2.u.m), digits = 4)

fit <- lmer(clo ~ 1 + (1 | subjId), data = dat.aggrFemale, 
            REML = TRUE)
myLmeSumm <- function(fit){
    print("Random effects:",quote = FALSE)
    print(summary(fit)[["varcor"]])
    print("Fixed effects:",quote = FALSE)
    summary(fit)[["coefficients"]]
}
myLmeSumm(fit)

##################################################
## Example 5.7

##################################################
## Example 5.8

fit.aggr <- lmer(clo  ~ 1 + (1|subjId), data = dat.aggr)
fixef(fit.aggr) - mean(dat.aggr$clo)

##################################################
## Example 5.9

sum(dfLMEx1$sex=="male")

##################################################
## Example 5.10

fit0 <- lmer(clo  ~ sex * tInOpCen + (tInOpCen| subjId), 
             data = dat.aggr)
fit0R <- lmer(clo  ~ sex * tInOpCen + (1| subjId), 
              data = dat.aggr)
aov <- anova(fit0, fit0R)
round(cbind(aov[ ,c(1,3,4,6:8)]),digits=2)

drop1(fit0R,test="Chisq")

summary(fit0R)

##################################################
## Section 5.3
##################################################

##################################################
## Example 5.11

##################################################
## Example 5.12

# define and fit model
modMEMEx12 <- lmer(clo ~ tOutCen + (1 + tOutCen | subjId), 
                   data = dfMEMEx1)
summary(modMEMEx12)


betamodMEMEx12 <- coef(modMEMEx12)$subjId

modLMEx11 <- lm(formula = clo ~ tOutCen * tInOpCen, data = dfMEMEx1)
cols <- gray(c(0.2,0.45,0.75))

base <- seq(round(min(dfMEMEx1$tOutCen),0), 
            round(max(dfMEMEx1$tOutCen),0), length.out=100)

dfPred <- data.frame(base, pred = betamodMEMEx12[1,1] + 
            betamodMEMEx12[1,2] * base, 
            subjId = unique(dfMEMEx1$subjId)[1])

for(i in 2:length(unique(dfMEMEx1$subjId))){
 dfPred <- rbind(dfPred, 
            data.frame(base, pred = betamodMEMEx12[i,1] + 
            betamodMEMEx12[i,2] * base, 
            subjId = unique(dfMEMEx1$subjId)[i]))
}
dfPred$tOut <- dfPred$base + mean(dfMEMEx1$tOut)


p <- ggplot(dfMEMEx1, aes(x=tOut, y=clo)) +
	    ylab("Clothing insulation level [CLO]") +
	    xlab(expression(paste("Outdoor air temp. ["^o,"C]")))+
	    scale_y_continuous(breaks = seq(0,1,.1), 
	          limits = c(0,1)) +  # Set ticks on y-axis
	    scale_x_continuous(breaks=seq(-20,20,2)) + # Set ticks on x-axis
	    theme_bw() +  
	    theme(legend.justification=c(0,0),
			      legend.position=c(.05,.05),
			      axis.text.x=element_text(size=9), 
			      axis.text.y=element_text(size=9),
			      axis.title=element_text(size=12),
			      axis.title.y=element_text(vjust=1)
	    )


dfPred <- dfPred[dfPred$pred>0 & dfPred$pred<1, ]

p1 <- p + geom_point() +
            geom_line( data = dfPred, 
	          aes(x=tOut, y=pred, 
	            group=subjId), 
	            colour="gray", 
	          size=.2
                  )

dfPredLM0 <- data.frame(tOut = base +  mean(dfMEMEx1$tOut), 
                      fit = modLMEx11$coeff[1] + base * modLMEx11$coeff[2],
                      Model="Linear")

dfPredLMM0 <- data.frame(tOut = base +  mean(dfMEMEx1$tOut), 
                        fit = fixef(modMEMEx12)[1] + (base) *
                            fixef(modMEMEx12)[2],
                        Model="Mixed effect")

dfPred0 <- rbind(dfPredLM0, dfPredLMM0)
dfPred0$Model <- factor(dfPred0$Model)

p1 + geom_line(data=dfPred0,aes(x=tOut,y=fit,linetype=Model,color=Model),size=2) +	 scale_fill_grey(start=0.3,end=0.6) + scale_color_grey(start=0.3,end=0.6)



##################################################
## Section 5.4
##################################################

##################################################
## Example 5.13

## detach("package:ordinal") ## to make ranef work for lme4
modMEMEx15 <- lmer(clo ~ tOutCen * sex + 
                       (1 + tOutCen | subjId)+
                       (1 + tOutCen| subjId: day), 
                   data = dfMEMEx1, REML = TRUE)

modMEMEx16 <- lmer(clo ~ tOutCen * sex + 
                       (1 + tOutCen | subjId)+
                       (1 | subjId: day), data = dfMEMEx1,
                   REML = TRUE)

modMEMEx17 <- lmer(clo ~ tOutCen * sex + (1 | subjId) + 
                       (1 | subjId : day), 
                   data = dfMEMEx1, REML = TRUE)

anova(modMEMEx15, modMEMEx16, modMEMEx17)[ ,c(1,4,6:8)]

summary(modMEMEx16)

##################################################
## Example 5.14

modMEMEx16tmb <- glmmTMB(clo ~ tOutCen * sex + 
                             (1 + tOutCen|subjId) +
                             (1|subjId:day), data = dfMEMEx1)

modMEMEx16AR1 <- glmmTMB(clo ~ tOutCen *as.numeric(sex) + 
                             (1 + tOutCen|subjId) +
                             ar1(as.factor(obs.no) - 1 | 
                                 subjId:day), data = dfMEMEx1)

anova(modMEMEx16tmb, modMEMEx16AR1)[ ,c(1,4,6:8)]

modMEMEx17AR1 <- glmmTMB(clo ~ tOutCen * sex + (1 | subjId) + 
                             ar1(as.factor(obs.no) - 1 |
                                 subjId : day), data=dfMEMEx1,
                         REML = FALSE)

modMEMEx18AR1 <- glmmTMB(clo ~ tOutCen + sex + (1 | subjId) + 
                             ar1(as.factor(obs.no) - 1 | 
                                 subjId:day), data = dfMEMEx1,
                         REML = FALSE)


anova(modMEMEx16AR1, modMEMEx17AR1, 
      modMEMEx18AR1)[ ,c(1,4,6:8)]

modMEMEx18AR1reml <- 
    glmmTMB(clo ~ tOutCen + sex + (1 | subjId) + 
                ar1(as.factor(obs.no) - 1 | subjId:day), 
            data = dfMEMEx1, REML = TRUE)

summary(modMEMEx18AR1reml)

sig.u.fun <- function(sigma.u,nu){
    sigma.u^2 * matrix(1, ncol = nu, nrow = nu)
}

sig.v.fun <- function(sigma.v,nv){
    sigma.v^2 * matrix(1, ncol = nv, nrow = nv)
}

sig.e.fun <- function(sigma,rho,n){
    exponent <- abs(matrix(1:n - 1, nrow = n, ncol = n, byrow = TRUE) - 
                    (1:n - 1))
    sigma^2 * rho^exponent
}

Sigma.U.fun <- function(sigma.u,dat){
    n.u <- table(dat$subjId)
    Sig.U <- matrix(0,ncol=dim(dat)[1],nrow=dim(dat)[1])
    j <- 1
    for(i in 1:length(n.u)){
       Sig.U[j:(j+n.u[i]-1),j:(j+n.u[i]-1)] <-  
           sig.u.fun(sigma.u,n.u[i])
           j <- j+n.u[i]
    }
    Sig.U
}


Sigma.V.fun <- function(sigma.v,dat){
    n.v <- table(dat$subjId:dat$day)
    Sig.V <- matrix(0,ncol=dim(dat)[1],nrow=dim(dat)[1])
    j <- 1
    for(i in 1:length(n.v)){
        if(n.v[i]>0){
            Sig.V[j:(j+n.v[i]-1),j:(j+n.v[i]-1)] <-  
                sig.u.fun(sigma.v,n.v[i])
            }
        j <- j+n.v[i]
    }
    Sig.V
}


Sigma.e.fun <- function(sigma,rho,dat){
    n.v <- table(dat$subjId:dat$day)
    Sig.V <- matrix(0,ncol=dim(dat)[1],nrow=dim(dat)[1])
    j <- 1
    for(i in 1:length(n.v)){
        if(n.v[i]>0){
            Sig.V[j:(j+n.v[i]-1),j:(j+n.v[i]-1)] <-  
                sig.e.fun(sigma,rho,n.v[i])
            }
        j <- j+n.v[i]
    }
    Sig.V
}


nll <- function(theta,X,dat){
    mu <- X %*% theta[1:3]
    sigmas <- exp(theta[4:6])
    rho <- 2 * exp(theta[7])/(1+exp(theta[7])) - 1
    V <- Sigma.U.fun(sigmas[1],dat)+Sigma.V.fun(sigmas[2],dat)+Sigma.e.fun(sigmas[3],rho,dat)
    0.5 * determinant(V, logarithm = TRUE)$modulus + 
                                             0.5 * t(dat$clo - mu) %*% solve(V) %*% (dat$clo - mu) + 
                                             0.5 * dim(dat)[1] * log(2*pi)
}

X <- model.matrix(modMEMEx18AR1)
theta <- modMEMEx18AR1$fit$par
optMEME19AR1 <- nlminb(theta,nll,X=X,dat=dfMEMEx1)

Z.fun <- function(dat,sigma.u,sigma.v){
    lev.sub <- levels(dat$subjId)
    Z.u <- c()
    Z.v <- c()
    for(i in 1:length(lev.sub)){
        Z.u <- cbind(Z.u,as.numeric(dat$subjId==lev.sub[i]))
        lev.day <- levels(as.factor(as.numeric(dat$day[dat$subjId==lev.sub[i]])))
        for(j in 1:length(lev.day)){
            Z.v <- cbind(Z.v,as.numeric(dat$subjId==lev.sub[i] & dat$day==lev.day[j]))
        }
    }
    psi.u <- diag(dim(Z.u)[2]) * sigma.u^2
    psi.v <- diag(dim(Z.v)[2]) * sigma.v^2
    psi <- rbind(cbind(psi.u,matrix(0,nrow=dim(psi.u)[2],ncol=dim(psi.v)[2])),
                 cbind(matrix(0,nrow=dim(psi.v)[1],ncol=dim(psi.u)[2]),psi.v))
    Z <- cbind(Z.u,Z.v)
    list(Z=Z, psi =psi)
}

##################################################
## Figure 5.3

sigma <- exp(0.5*modMEMEx18AR1$sdr$par.fixed[4])
sigma.u <- exp(modMEMEx18AR1$sdr$par.fixed[5])
sigma.v <- exp(modMEMEx18AR1$sdr$par.fixed[6])
rho <- exp(modMEMEx18AR1$sdr$par.fixed[7])/(1+exp(modMEMEx18AR1$sdr$par.fixed[7]))
Sigma.u.fun <- function(n,sigma.u){
    sigma.u^2*matrix(1,ncol=n,nrow=n)
}
Sigma.v.fun <- function(n,sigma.v,rho){
    R <- diag(n)
    for(i in 1:(n-1)){
        for(j in (i+1):n){
            R[i,j] <- R[j,i] <- rho^(abs(i-j))
        }
    }
    sigma.v^2*R
}

V <- matrix(0,ncol=dim(dfMEMEx1)[1],nrow=dim(dfMEMEx1)[1])
lev.subj <- levels(dfMEMEx1$subjId)
for(i in 1:length(lev.subj)){
    I <- dfMEMEx1$subjId==lev.subj[i]
    V[I,I] <- Sigma.u.fun(sum(I),sigma.v)
    lev.day <-  levels(as.factor(as.numeric(dfMEMEx1$day[I])))
    for(j in 1:length(lev.day)){
        I.d <- dfMEMEx1$subjId==lev.subj[i] & dfMEMEx1$day == lev.day[j]
        V[I.d,I.d] <- V[I.d,I.d] + Sigma.v.fun(sum(I.d),sigma.v,rho)
    }
}
diag(V) <- diag(V)+sigma^2
X <- model.matrix(modMEMEx18AR1)
mu <- X%*%modMEMEx18AR1$sdr$par.fixed[1:3]
par(mfrow=c(1,1))
corrPlot(cov2cor(V)[1:18,1:18],main=expression(V[i]))


##################################################
## Example 5.15

beta <- optMEME19AR1$par[1:3]
names(beta) <- 
sigma.u <- exp(optMEME19AR1$par[4])
sigma.v <- exp(optMEME19AR1$par[5])
sigma <- exp(optMEME19AR1$par[6])
phi <- 2 * exp(optMEME19AR1$par[7]) / (1 + exp(optMEME19AR1$par[7])) - 1
theta <- c(beta,sigma.u=sigma.u,sigma.v=sigma.v,sigma,phi=phi)
names(theta) <- c("beta.0","beta.1","beta.m","sigma.u","sigma.v","sigma","phi") 
logLik19 <- - optMEME19AR1$objective
round(theta,digits=4)
logLik19

##################################################
## Figure 5.4
rho <-  2 * exp(optMEME19AR1$par[7])/(1+exp(optMEME19AR1$par[7]))-1
Z <- Z.fun(dfMEMEx1,sigma.u=exp(optMEME19AR1$par[4]),sigma.v=exp(optMEME19AR1$par[5]))
Sigma <- Sigma.e.fun(exp(optMEME19AR1$par[6]),rho,dfMEMEx1)
u.hat <- solve((t(Z$Z) %*% solve(Sigma) %*% Z$Z + solve(Z$psi))) %*% t(Z$Z) %*% solve(Sigma) %*% (dfMEMEx1$clo - X %*% optMEME19AR1$par[1:3])
Vu <- solve((t(Z$Z) %*% solve(Sigma) %*% Z$Z + solve(Z$psi)))
par(mfrow=c(1,2))
u.subj <- u.hat[1:length(levels(dfMEMEx1$subjId))]
sdu.subj <- sqrt(diag(Vu)[1:length(levels(dfMEMEx1$subjId))])
I.subj <- sort(u.subj,index.return=TRUE)$ix
plot(1:length(u.subj),u.subj[I.subj],ylim=range(c(u.subj+2*sdu.subj,u.subj-2*sdu.subj)),col="white",xlab="Subject",ylab="u")
points(1:length(u.subj),sort(u.subj),pch=19)
for(i in 1:length(u.subj)){
    lines(c(i,i),c(u.subj[I.subj[i]]+2*sdu.subj[I.subj[i]],
                   u.subj[I.subj[i]]-2*sdu.subj[I.subj[i]]),col=grey(0.5))
}


Sigma.u <- Sigma.U.fun(exp(optMEME19AR1$par[4]),dfMEMEx1)
u.day <- u.hat[-(1:length(levels(dfMEMEx1$subjId)))]


sdu.day <- sqrt(diag(Vu)[-(1:length(levels(dfMEMEx1$subjId)))])
u.day <- u.hat[-(1:length(levels(dfMEMEx1$subjId)))]
I.day <- sort(u.day,index.return=TRUE)$ix
plot(1:length(u.day),u.day[I.day],ylim=range(c(u.day[I.day]+2*sdu.day,u.day[I.day]-2*sdu.day)),col="white",xlab="Subject:day",
ylab="v")
points(1:length(u.day),sort(u.day),pch=19)
for(i in 1:length(u.day)){
    lines(c(i,i),c(u.day[I.day[i]]+2*sdu.day[I.day[i]],
                   u.day[I.day[i]]-2*sdu.day[I.day[i]]),col=grey(0.5))
}
points(1:length(u.day),sort(u.day),pch=19)


##################################################
## Section 5.5
##################################################
