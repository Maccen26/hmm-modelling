library(ggplot2)
library(grid)
source("functions/multiplot.R")
##################################################
## Section 3.1
##################################################

##################################################
## Section 3.2
##################################################

##################################################
## Example 3.1

##################################################
## Section 3.3
##################################################

##################################################
## Example 3.2

## Read data
dfLMEx1 <- read.csv(file = "data/clothing.csv")
dfLMEx1$sex <- factor(dfLMEx1$sex)

# show histogram
ggplot(data=dfLMEx1, aes(clo)) + 
  geom_histogram(breaks=seq(0, 1.2, by = .05), 
                 col="black", 
                 fill="white") + 
  labs(x="Clothing insulation level [CLO]", y="Count") +
  theme_bw() +	 
	theme(axis.text.x=element_text(size=9), 
              axis.text.y=element_text(size=9),
              axis.title=element_text(size=12),
              axis.title.y=element_text(vjust=1)
	)


# figure showing relationship between clo and tOut
p1 <- ggplot(dfLMEx1, aes( tOut,clo)) +
  geom_point() +
  labs(y="Clo. insul. lev. [CLO]", 
       x=expression(paste("Outdoor air temp. ["^o,"C]"))) +
  theme_bw() +	 
	theme(axis.text.x=element_text(size=9), 
			      axis.text.y=element_text(size=9),
			      axis.title=element_text(size=12),
			      axis.title.y=element_text(vjust=1)
              )

# figure showing relationship between clo and tInOp
p3 <- ggplot(dfLMEx1, aes(tInOp,tOut)) +
  geom_point() +
  labs(y=expression(paste("Outdoor air temp. ["^o,"C]")),
       x=expression(paste("Indoor opr. temp. ["^o,"C]")))+
  theme_bw() +	 
	theme( axis.text.x=element_text(size=9), 
			      axis.text.y=element_text(size=9),
			      axis.title=element_text(size=12),
			      axis.title.y=element_text(vjust=1)
	)


# figure showing relationship between clo and tInOp
p2 <- ggplot(dfLMEx1, aes(tInOp,clo)) +
  geom_point() +
  labs(y="Clo. insul. lev. [CLO]", 
       x=expression(paste("Indoor opr. temp. ["^o,"C]")))+
  theme_bw() +	 
	theme( axis.text.x=element_text(size=9), 
			      axis.text.y=element_text(size=9),
			      axis.title=element_text(size=12),
			      axis.title.y=element_text(vjust=1)
	)


multiplot(p1, p2,p3, cols=2)

##################################################
## Section 3.4
##################################################

##################################################
## Example 3.3

## Define and fit linear model
modLMEx11 <- lm(clo ~ tOut * tInOp, data = dfLMEx1)
## Summary of linear model fit
summary(modLMEx11)


# Define and fit linear model
modLMEx12 <- lm(clo ~ tOut + tInOp, data = dfLMEx1)
summary(modLMEx12)

# define and fit linear model with one continuous and one 
# binary variable and interaction term
modLMEx13 <- lm(clo ~ tOut * sex + tInOp * sex, 
                data = dfLMEx1)
summary(modLMEx13)

# define and fit linear model with one continuous and one 
# binary variable and interaction term
modLMEx14 <- lm(clo ~ tOut * sex + tInOp, data = dfLMEx1)
summary(modLMEx14)


tout <- seq(min(dfLMEx1$tOut),max(dfLMEx1$tOut),
            length=200)
confF <- predict(modLMEx14,
                 newdata=
                     data.frame(tOut=tout,
                                tInOp =
                                 mean(dfLMEx1$tInOp),
                                sex="female"),
                 interval="confidence")
confM <- predict(modLMEx14,
                 newdata=
                     data.frame(tOut=tout,
                                tInOp =
                                 mean(dfLMEx1$tInOp),
                                sex="male"),
                 interval="confidence")
confF <- data.frame(tOut=tout, confF,
                    sex="female")
confM <- data.frame(tOut=tout, confM, 
                         sex="male")

conf <- rbind(confM,confF)


plot <- ggplot(data=conf, aes(x=tOut, y=fit))

plot + geom_line(aes(x=tOut, y=fit,linetype=sex, colour=sex))+
    geom_ribbon(aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .2) + 
    geom_point(data=dfLMEx1, aes(x=tOut, y=clo,
                                  colour=sex))+ 
    xlab(expression(paste("Outdoor air temp. ["^o,"C]")))+
    ylab("Clothing insulation level [CLO]")+
          theme_bw() +	 scale_fill_grey(end=0.5) + scale_color_grey(end=0.5) +
    theme(legend.justification=c(1,0),		
          legend.position=c(.95,.62),
          axis.text.x=element_text(size=9), 
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12),
          axis.title.y=element_text(vjust=1),
          legend.text=element_text(size=18),
          legend.title=element_text(size=18))


##################################################
## Example 3.4

## diagnostic plot reference
par(mfrow=c(2,2),bg="white")
col.tmp <- (dfLMEx1$sex=="female")*1+1
pt.type <- col.tmp 
pt.type[pt.type==1] <- 19
pt.type[pt.type==2] <- 4

col.tmp[col.tmp==2] <- 0.5
col.tmp[col.tmp==1] <- 0
panel.fun.tmp <- function(x, y, ...)
              panel.smooth(x, y, iter=3, col.smooth=1,lty=2,...)

 plot(modLMEx14,cex=0.5,col=gray(col.tmp),pch=pt.type,which=1,
      panel = panel.fun.tmp)
 plot(modLMEx14,cex=0.5,col=gray(col.tmp),pch=pt.type,which=2)
legend("topleft",legend=c("males","females"),pch=c(19,4),pt.cex=0.5,
       col=c(1,gray(0.5)))
 plot(modLMEx14,cex=0.5,col=gray(col.tmp),pch=pt.type,which=3,
      panel = panel.fun.tmp)
 plot(modLMEx14,cex=0.5,col=gray(col.tmp),pch=pt.type,which=5,
      panel = panel.fun.tmp)


par(mfrow=c(1,2),bg="white")
col.tmp <- (dfLMEx1$sex=="female")*1+1
pt.type <- col.tmp 
pt.type[pt.type==1] <- 19
pt.type[pt.type==2] <- 4

col.tmp[col.tmp==2] <- 0.5
col.tmp[col.tmp==1] <- 0

plot(rstudent(modLMEx14)~as.factor(dfLMEx1$sex),cex=0.2,xlab="Sex",ylab="Stundentized residuals")
plot(rstudent(modLMEx14)~dfLMEx1$tOut,
     col=gray(col.tmp),pch=pt.type,cex=0.4,xlab="tOut",ylab="Stundentized residuals")


##################################################
## Figure 3.1
set.seed(5335)
sig <- 0.5
n <- 100
x <- runif(n)
y1 <- -2 + x + rnorm(n,sd=sig)
y2 <- -2 + x + rnorm(n,sd=x*sig)
y4 <- -2 + x - 16 * x^2 + rnorm(n,sd=sig)
fit1 <- lm(y1 ~ x)
fit2 <- lm(y2 ~ x)
fit3 <- lm(y1 ~ -1 + x)
fit4 <- lm(y4 ~ x)
par(mfrow=c(2,2),mar=c(1,3,1,1),oma=c(1,1,1,1))

res <- residuals(fit1)
plot(x,res,pch=19,axes=FALSE,ylab="",xlab="")
mtext("Residuals",side=2,line=1)
at <- seq(min(res),max(res),length=5)
axis(2,at =at, labels=rep("",5))
lines(c(-1,max(x)),at[3]*c(1,1))
legend("topleft",legend="1)")

res <- residuals(fit2)
plot(x,res,pch=19,axes=FALSE,ylab="",xlab="")
mtext("Residuals",side=2,line=1)
at <- seq(min(res),max(res),length=5)
axis(2,at =at, labels=rep("",5))
lines(c(-1,max(x)),at[3]*c(1,1))
legend("topleft",legend="2)")

res <- residuals(fit3)
plot(x,res,pch=19,axes=FALSE,ylab="",xlab="")
mtext("Residuals",side=2,line=1)
at <- seq(min(res),max(res),length=5)
axis(2,at =at, labels=rep("",5))
lines(c(-1,max(x)),at[3]*c(1,1))
legend("topleft",legend="3)")

res <- residuals(fit4)
plot(x,res,pch=19,axes=FALSE,ylab="",xlab="")
mtext("Residuals",side=2,line=1)
at <- seq(min(res),max(res),length=5)
axis(2,at =at, labels=rep("",5))
lines(c(-1,max(x)),at[3]*c(1,1))
legend("topleft",legend="4)")



##################################################
## Section 3.5
##################################################

##################################################
## Example 3.5

modLMExNull <- lm(clo ~ tOut, data = dfLMEx1)
anova(modLMExNull,modLMEx11)

##################################################
## Section 3.6
##################################################

##################################################
## Example 3.6

m01 <- lm(formula = clo ~ tOut * sex + tInOp * sex + 
              tOut * tInOp, data = dfLMEx1)
m02 <- lm(formula = clo ~ tOut * sex + tOut * tInOp + 
              tInOp * sex, data = dfLMEx1)

anova(m01)

anova(m02)

drop1(m01, test = "F")

##################################################
## Section 3.7
##################################################

##################################################
## Example 3.7

set.seed(248)
dfLMEx1$tOut2 <- dfLMEx1$tOut + rnorm(length(dfLMEx1$tOut))
cor(dfLMEx1$tOut, dfLMEx1$tOut2)

summary(lm(clo ~ tOut + tOut2, data = dfLMEx1))

##################################################
## Example 3.8

X <- cbind(1, dfLMEx1$tOut, dfLMEx1$tOut2) ## Design matrix
kappa(solve(t(X) %*% X)) ## Condition number

## End Example
##################################################

1/(1-cor(dfLMEx1$tOut,dfLMEx1$tOut2)^2)

library(car)
vif(lm(clo~tOut + tOut2, data = dfLMEx1))

##################################################
## Section 3.8
##################################################

##################################################
## Example 3.9

## DEsign matrix
X <- model.matrix(modLMEx13)[ ,c(4,6)]
## Estimated variance
sigma.sq <- summary(modLMEx13)$sigma^2
## Parameter estimates
beta.hat <- coef(modLMEx13)[c(4,6)]
## Covariance matrix for parameters
Cov <- sigma.sq * summary(modLMEx13)$cov.unscaled
## Test statistics
Q <- (beta.hat %*% solve(Cov[c(4, 6), c(4, 6)]) %*% 
      beta.hat) / 2
## p-values
1 - pf(Q, df1 = 2, df2 = 797)

##################################################
## Section 3.9
##################################################

##################################################
## Example 3.10


tout <- seq(min(dfLMEx1$tOut),max(dfLMEx1$tOut),
            length=200)
confF <- predict(modLMEx14,
                 newdata=
                     data.frame(tOut=tout,
                                tInOp =
                                 mean(dfLMEx1$tInOp),
                                sex="female"),
                 interval="confidence")
confM <- predict(modLMEx14,
                 newdata=
                     data.frame(tOut=tout,
                                tInOp =
                                 mean(dfLMEx1$tInOp),
                                sex="male"),
                 interval="confidence")

predF <- predict(modLMEx14,
                 newdata=
                     data.frame(tOut=tout,
                                tInOp =
                                 mean(dfLMEx1$tInOp),
                                sex="female"),
                 interval="prediction")
predM <- predict(modLMEx14,
                 newdata=
                     data.frame(tOut=tout,
                                tInOp =
                                 mean(dfLMEx1$tInOp),
                                sex="male"),
                 interval="prediction")

confF <- data.frame(tOut=tout, confF,
                    sex="female")
confM <- data.frame(tOut=tout, confM, 
                         sex="male")
predF <- data.frame(tOut=tout, predF,
                    sex="female")
predM <- data.frame(tOut=tout, predM, 
                         sex="male")

conf <- rbind(confM,confF)
pred <- rbind(predM,predF)


plot <- ggplot(data=conf, aes(x=tOut, y=fit))

plot + geom_line(aes(x=tOut, y=fit, linetype=sex, colour=sex))+
    geom_ribbon(aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .6) +
    geom_ribbon(data=pred,aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .2) + 
    geom_point(data=dfLMEx1, aes(x=tOut, y=clo,
                                  colour=sex))+ 
    xlab(expression(paste("Outdoor air temp. ["^o,"C]")))+
    ylab("Clothing insulation level [CLO]")+
          theme_bw() +	  scale_fill_grey(end=0.5) + scale_color_grey(end=0.5) +
    theme(legend.justification=c(1,0),		
          legend.position=c(.95,.62),
          axis.text.x=element_text(size=9), 
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12),
          axis.title.y=element_text(vjust=1),
          legend.text=element_text(size=18),
          legend.title=element_text(size=18))


##################################################
## Section 3.10
##################################################

##################################################
## Example 3.11

    var(residuals(modLMEx14)[dfLMEx1$sex == "female"]) / 
        var(residuals(modLMEx14)[dfLMEx1$sex == "male"])

X <- model.matrix(modLMEx14)

  nll.weight <- function(gamma,X,y, full){
      n <- length(y) ## no. of obs
      p <- dim(X)[2] ## no. of parameters
      ## Setup Sigma (step 1)
      W <- rep(1, n)
      W[X[ ,"sexmale"] == 1] <- gamma
      Sigma <- diag(1 / W)
      ISigma <- diag(W)
      ## Step 2
      beta <- solve((t(X) %*% ISigma %*%X)) %*% t(X) %*% ISigma %*% y
      ## Step 3
      r <- y - X %*% beta
      ## Moment/marginal likelihood estimate
      sig.sq <- sum(r^2 * W) / (n - p) 
      ## Step 4
      ll <- sum(dnorm(y, mean = X %*% beta, sd = sqrt(sig.sq / W), 
                      log = TRUE))
      ## If full TRUE, the parameters and the likelihood is 
      ## returned otherwise only likelihood is returned
      if(full){
          return(list(gamma=gamma, beta = as.numeric(beta), 
                      sig.sq = sig.sq, logLik = ll))
      }
      - ll ## negative log-likelihood returned
  }

-nll.weight(1,X=X, y=dfLMEx1$clo,full=FALSE)
logLik(modLMEx14)

(w1 <-optimise(nll.weight, lower = 0.01, upper = 10, X = X, 
              y = dfLMEx1$clo, full = FALSE))

cbind(coef(modLMEx14), 
      nll.weight(w1$minimum, X, y=dfLMEx1$clo, full=TRUE)$beta)

##################################################
## Section 3.11
##################################################

##################################################
## Example 3.12

s <- 141#57#43#19

col.tmp <- gray(c(0,1/4,1/2))
par(mfrow=c(1,2),bg="white")
In <- dfLMEx1$subjId==s & dfLMEx1$day==1
plot(cumsum(dfLMEx1$time[In]),dfLMEx1$clo[In],type="b",ylim=c(0,1),
     xlim=c(0,8),xlab="Time [h]",ylab = "Clo",pch=19,col=col.tmp[1])
for(i in 2:4){
    In <- dfLMEx1$subjId==s & dfLMEx1$day==i
    lines(cumsum(dfLMEx1$time[In]),dfLMEx1$clo[In],type="b",pch=i,
          col=col.tmp[i],lty=i)
}
legend("bottomleft",legend=c("Day 1","Day 2","Day 3"),
       pch=c(19,2,3),col=col.tmp,lty=1:3,ncol=2)

In <- dfLMEx1$subjId==s & dfLMEx1$day==1
dfLMEx1$res <- residuals(modLMEx14)
plot(cumsum(dfLMEx1$time[In]),dfLMEx1$res[In],type="b",
     xlim=c(0,8),ylim=c(-0.3,0.3),xlab="Time [h]",
     ylab = "residual",pch=19)
for(i in 2:4){
    In <- dfLMEx1$subjId==s & dfLMEx1$day==i
    lines(cumsum(dfLMEx1$time[In]),dfLMEx1$res[In],type="b",pch=i,
          col=col.tmp[i],lty=i)
}


dat.tmp <- dfLMEx1
dat.tmp$res <- residuals(modLMEx14)
## Calculate within day residual autocorrelation
cor <- aggregate(res ~ subjId + day, data = dat.tmp,
                 function(x){acf(x,plot=FALSE)$acf[2]})[ ,3]
summary(cor)
length(cor)

## Test
t.test(cor)

##################################################
## Example 3.13

par(bg="white")
dfLMEx1$sex <- as.factor(dfLMEx1$sex)
agrdat <- aggregate(. ~ subjId + day, data = dfLMEx1[ ,c(2:5,6,8)], 
                    mean)
agrdat$sex[agrdat$sex==1] <- "female"
agrdat$sex[agrdat$sex==2] <- "male"
agrdat$sex <- as.factor(agrdat$sex)
col.tmp <- (agrdat$sex=="female")*1+1
pt.type <- col.tmp 
pt.type[pt.type==1] <- 19
pt.type[pt.type==2] <- 4

col.tmp[col.tmp==2] <- 0.5
col.tmp[col.tmp==1] <- 0

plot(agrdat[ ,c("clo","tOut","tInOp")],
      col=gray(col.tmp),,pch=pt.type)

summary(lm(formula = clo ~ tOut * sex + tInOp, data = agrdat))

##################################################
## Section 3.12
##################################################

##################################################
## Section 3.13
##################################################

##################################################
## Example 3.14

modLMEx15 <- lm(clo ~ tOut * sex + I(log(tOut)) * sex  + 
                    tInOp, data = dfLMEx1)
drop1(modLMEx15, test="F")[ ,-(2:3)]



modLMEx16 <- lm(formula = clo ~ tOut  + 
                    I(log(tOut)) * sex  + tInOp, 
                data = dfLMEx1)
drop1(modLMEx16, test="F")[ ,-(2:3)]
anova(modLMEx16,modLMEx15)


modLMEx16 <- lm(clo ~ tOut  + I(log(tOut)) * sex  + 
                    tInOp, data = dfLMEx1)

c(AIC(modLMEx14),AIC(modLMEx16))
c(BIC(modLMEx14),BIC(modLMEx16))



 dfLMEx1$clo.sqrt <- sqrt(dfLMEx1$clo)
 modLMEx17a <- lm(formula = clo.sqrt ~ tOut * sex+I(log(tOut)) * sex  + tInOp, data = dfLMEx1)
 drop1(modLMEx17a,test="F")
 modLMEx17b <- lm(formula = clo.sqrt ~ tOut + I(log(tOut)) * sex  + tInOp, data = dfLMEx1)
 drop1(modLMEx17b,test="F")
 modLMEx17c <- lm(formula = clo.sqrt ~ tOut + I(log(tOut)) * sex , data = dfLMEx1)
 drop1(modLMEx17c,test="F")



 modLMEx17 <- lm(clo.sqrt ~ tOut + I(log(tOut)) * sex ,  
                 data = dfLMEx1)


AIC17 <- AIC(modLMEx17) - 2 * sum(log(dfLMEx1$clo)) - 
    0.5 * length(dfLMEx1$clo)
BIC17 <- BIC(modLMEx17) - 2 * sum(log(dfLMEx1$clo)) - 
    0.5 * length(dfLMEx1$clo)
c(AIC(modLMEx14),AIC(modLMEx16),AIC17)
c(BIC(modLMEx14),BIC(modLMEx16),BIC17)



dfLMEx1$clo.logit <- log(dfLMEx1$clo / (1 - dfLMEx1$clo))
modLMEx18a <- lm(clo.logit ~ tOut * sex + I(log(tOut)) * sex  + tInOp, data = dfLMEx1)
drop1(modLMEx18a, test = "F")

modLMEx18b <- lm(clo.logit ~ tOut + I(log(tOut)) * sex  + tInOp, data =  dfLMEx1)
drop1(modLMEx18b, test = "F")
dfLMEx1$clo.logit <- log(dfLMEx1$clo / (1 - dfLMEx1$clo))
modLMEx18 <- lm(clo.logit ~ tOut + I(log(tOut)) * sex  + tInOp, 
                data = dfLMEx1)

AIC18 <-  AIC(modLMEx18) + 2 * sum(log(dfLMEx1$clo * (1 - dfLMEx1$clo)))
BIC18 <-  BIC(modLMEx18) + 2 * sum(log(dfLMEx1$clo * (1 - dfLMEx1$clo)))
tab <- rbind(c(AIC(modLMEx14), AIC(modLMEx16), AIC17, AIC18),
             c(BIC(modLMEx14), BIC(modLMEx16), BIC17, BIC18))
rownames(tab) <- c("AIC","BIC")
colnames(tab) <- c("LMEx14","LMEx16","LMEx17","LMEx18")
tab

## Profile likelihood of transformation
lP.lambda <- function(lambda){
    dat <- dfLMEx1
    M <- lm(I(trans(clo,lambda)$y) ~ tOut * sex + I(log(tOut)) * sex + 
                tInOp, data = dat)
    n <- dim(dat)[1]
    -log(summary(M)$sigma^2) * n / 2 + 
        sum(abs(log(trans(dat$clo,lambda)$dy)))
}

## Transformation
trans<-function(x,g){
    y <- log(x^g / (1 - x^g)) / g
    dy <- 1 / (x * (1 - x^g))
     return(list(y = y, dy = dy))
}
(opt <- optimize(lP.lambda, c(0, 40), maximum = TRUE))

modLMEx19a <- lm(I(trans(clo,6.6)$y) ~ tOut * sex + I(log(tOut)) * sex + 
                     tInOp, data = dfLMEx1)
drop1(modLMEx19a,test="F")

modLMEx19b <- lm(I(trans(clo,6.6)$y) ~ tOut  + I(log(tOut)) * sex + tInOp,
                 data = dfLMEx1)
drop1(modLMEx19b,test="F")

modLMEx19c <- lm(I(trans(clo,6.6)$y) ~ tOut + I(log(tOut)) * sex,
                 data = dfLMEx1)
drop1(modLMEx19c,test="F")
modLMEx19 <- lm(I(trans(clo, 6.6)$y) ~ tOut + I(log(tOut)) * sex,
                data = dfLMEx1)
                                        
AIC19 <- AIC(modLMEx19) - 2 * sum(log(trans(dfLMEx1$clo, 6.6)$dy))
BIC19 <- BIC(modLMEx19) - 2 * sum(log(trans(dfLMEx1$clo, 6.6)$dy))

tab <- cbind(tab,LMEx19=c(AIC19,BIC19))
tab


transInv <- function(tclo,lambda){
    (exp(tclo*lambda)/(1+exp(tclo*lambda)))^(1/lambda)}

tout <- seq(min(dfLMEx1$tOut),max(dfLMEx1$tOut),
            length=200)
X.m <- data.frame(tOut=tout, tInOp = mean(dfLMEx1$tInOp), sex="male")
X.f <- data.frame(tOut=tout, tInOp = mean(dfLMEx1$tInOp), sex="female")

pred14.f <- predict(modLMEx14,newdata=X.f,interval="prediction")
pred14.m <- predict(modLMEx14,newdata=X.m,interval="prediction")
conf14.f <- predict(modLMEx14,newdata=X.f,interval="confidence")
conf14.m <- predict(modLMEx14,newdata=X.m,interval="confidence")
conf14.f <- data.frame(tOut=tout, conf14.f,sex="female")
conf14.m <- data.frame(tOut=tout, conf14.m,sex="male")
pred14.f <- data.frame(tOut=tout, pred14.f,sex="female")
pred14.m <- data.frame(tOut=tout, pred14.m,sex="male")
conf14 <- rbind(conf14.f,conf14.m)
pred14 <- rbind(pred14.f,pred14.m)

pred16.f <- predict(modLMEx16,newdata=X.f,interval="prediction")
pred16.m <- predict(modLMEx16,newdata=X.m,interval="prediction")
conf16.f <- predict(modLMEx16,newdata=X.f,interval="confidence")
conf16.m <- predict(modLMEx16,newdata=X.m,interval="confidence")
conf16.f <- data.frame(tOut=tout, conf16.f,sex="female")
conf16.m <- data.frame(tOut=tout, conf16.m,sex="male")
pred16.f <- data.frame(tOut=tout, pred16.f,sex="female")
pred16.m <- data.frame(tOut=tout, pred16.m,sex="male")
conf16 <- rbind(conf16.f,conf16.m)
pred16 <- rbind(pred16.f,pred16.m)


pred17.f <- predict(modLMEx17,newdata=X.f,interval="prediction")
pred17.m <- predict(modLMEx17,newdata=X.m,interval="prediction")
conf17.f <- predict(modLMEx17,newdata=X.f,interval="confidence")
conf17.m <- predict(modLMEx17,newdata=X.m,interval="confidence")
conf17.f <- data.frame(tOut=tout, conf17.f,sex="female")
conf17.m <- data.frame(tOut=tout, conf17.m,sex="male")
pred17.f <- data.frame(tOut=tout, pred17.f,sex="female")
pred17.m <- data.frame(tOut=tout, pred17.m,sex="male")
conf17 <- rbind(conf17.f,conf17.m)
pred17 <- rbind(pred17.f,pred17.m)

pred19.f <- predict(modLMEx19,newdata=X.f,interval="prediction")
pred19.m <- predict(modLMEx19,newdata=X.m,interval="prediction")
conf19.f <- predict(modLMEx19,newdata=X.f,interval="confidence")
conf19.m <- predict(modLMEx19,newdata=X.m,interval="confidence")
conf19.f <- data.frame(tOut=tout, conf19.f,sex="female")
conf19.m <- data.frame(tOut=tout, conf19.m,sex="male")
pred19.f <- data.frame(tOut=tout, pred19.f,sex="female")
pred19.m <- data.frame(tOut=tout, pred19.m,sex="male")
conf19 <- rbind(conf19.f,conf19.m)
pred19 <- rbind(pred19.f,pred19.m)

conf<- conf14
pred<-pred14
legend.size <- 12
plot14 <- ggplot(data=conf, aes(x=tOut, y=fit)) + 
    ylim(0,1.2)+
    geom_line(aes(x=tOut, y=fit, linetype=sex,
                     colour=sex))+
    geom_point(data=dfLMEx1, aes(x=tOut, y=clo,
                                  colour=sex),size=0.5)+ 
    geom_ribbon(aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .6) +
    geom_ribbon(data=pred,aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .2) + 
    xlab(expression(paste("Outdoor air temp. ["^o,"C]")))+
    ylab("Clothing insulation level [CLO]")+
          theme_bw() +	  scale_fill_grey(end=0.5) + scale_color_grey(end=0.5) +
    theme(legend.justification=c(1,0),		
          legend.position=c(.95,.65),
          axis.text.x=element_text(size=9), 
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12),
          axis.title.y=element_text(vjust=1),
          legend.text=element_text(size=legend.size),
          legend.title=element_text(size=legend.size))

conf<- conf16
pred<-pred16
plot16 <- ggplot(data=conf, aes(x=tOut, y=fit)) + 
        ylim(0,1.2)+
    geom_line(aes(x=tOut, y=fit, linetype=sex,
                     colour=sex))+
        geom_point(data=dfLMEx1, aes(x=tOut, y=clo,
                                  colour=sex),size=0.5)+ 
    geom_ribbon(aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .6) +
    geom_ribbon(data=pred,aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .2) + 
    xlab(expression(paste("Outdoor air temp. ["^o,"C]")))+
    ylab("Clothing insulation level [CLO]")+
          theme_bw() +	  scale_fill_grey(end=0.5) + scale_color_grey(end=0.5) +
    theme(legend.justification=c(1,0),		
          legend.position=c(.95,.65),
          axis.text.x=element_text(size=9), 
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12),
          axis.title.y=element_text(vjust=1),
          legend.text=element_text(size=legend.size),
          legend.title=element_text(size=legend.size))

conf <- conf17
pred<-pred17
conf[ ,c(2:4)]<- conf17[ ,c(2:4)]^2
pred[ ,c(2:4)]<-pred17[ ,c(2:4)]^2
plot17 <- ggplot(data=conf, aes(x=tOut, y=fit)) + 
        ylim(0,1.2)+
    geom_line(aes(x=tOut, y=fit, linetype=sex,
                     colour=sex))+
        geom_point(data=dfLMEx1, aes(x=tOut, y=clo,
                                  colour=sex),size=0.5)+ 
    geom_ribbon(aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .6) +
    geom_ribbon(data=pred,aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .2) + 
    xlab(expression(paste("Outdoor air temp. ["^o,"C]")))+
    ylab("Clothing insulation level [CLO]")+
          theme_bw() +	  scale_fill_grey(end=0.5) + scale_color_grey(end=0.5) +
    theme(legend.justification=c(1,0),		
          legend.position=c(.95,.65),
          axis.text.x=element_text(size=9), 
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12),
          axis.title.y=element_text(vjust=1),
          legend.text=element_text(size=legend.size),
          legend.title=element_text(size=legend.size))


conf <- conf19
pred<-pred19
conf[ ,c(2:4)]<- transInv(conf19[ ,c(2:4)],6.6)
pred[ ,c(2:4)]<- transInv(pred19[ ,c(2:4)],6.6)
plot19 <- ggplot(data=conf, aes(x=tOut, y=fit)) + 
        ylim(0,1.2)+
    geom_line(aes(x=tOut, y=fit, linetype=sex,
                     colour=sex))+
        geom_point(data=dfLMEx1, aes(x=tOut, y=clo,
                                  colour=sex),size=0.5)+
    geom_ribbon(aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .6) +
    geom_ribbon(data=pred,aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .2) + 
    xlab(expression(paste("Outdoor air temp. ["^o,"C]")))+
    ylab("Clothing insulation level [CLO]")+
          theme_bw() +	  scale_fill_grey(end=0.5) + scale_color_grey(end=0.5) +
    theme(legend.justification=c(1,0),		
          legend.position=c(.95,.65),
          axis.text.x=element_text(size=9), 
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12),
          axis.title.y=element_text(vjust=1),
          legend.text=element_text(size=legend.size),
          legend.title=element_text(size=legend.size))


library(gridExtra)
grid.arrange(plot14,plot16,plot17,plot19,ncol=2)


##################################################
## Section 3.14
##################################################
