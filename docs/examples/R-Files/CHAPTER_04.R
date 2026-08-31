
library(ggplot2)
library(ordinal)
##################################################
## Section 4.1
##################################################

##################################################
## Example 4.1

##################################################
## Figure 4.1

win1 <- read.table("data/b1.csv",sep=";",header=TRUE)
hourInterval <- c(17,21)
I <- win1$Room=="Livingroom" & win1$Hour>hourInterval[1] &
    win1$Hour<hourInterval[2] &
    !is.na(win1$WindowClosed)
agrdat1 <- cbind(aggregate(WindowClosed ~ Day + Month,
                           data = win1[I, ],sum),
                 OutdoorTemp=aggregate(OutdoorTemp ~ Day + Month,
                                       data = win1[I, ],
                                       mean)[ ,3])

agrdat1$WindowOpen<- (diff(hourInterval) - 1) * 6 -
    agrdat1$WindowClosed


agrdat1$WindowOpen[agrdat1$WindowOpen<=0] <- 0
agrdat1$WindowOpen[agrdat1$WindowOpen>0] <- 1

limits <- seq(-2,22,by=2)
n <- numeric(length(limits)-1)
y <- n

for(i in 1:(length(limits)-1)){
    I <- agrdat1$OutdoorTemp>limits[i] &
        agrdat1$OutdoorTemp<=limits[i+1]
    n[i] <- sum(I)
    y[i] <- sum(agrdat1$WindowOpen[I])
}

temp <- (limits[-1]+limits[-length(limits)])/2
temp<-temp[n>0]
y<-y[n>0]
n<-n[n>0]
WindowDat1 <- data.frame(temp=temp, y=y, n=n)
WindowDat1.print <- t(WindowDat1[ ,2:3])
colnames(WindowDat1.print)<- WindowDat1[ ,1]

WindowDat1.print

 plot(y/n~temp, data = WindowDat1, xlim=c(-4,24),pch=19)
 linmod<-glm(y/n~temp, data = WindowDat1, family=gaussian())
 abline(a = linmod$coefficients[1], b= linmod$coefficients[2])

##################################################
## Example 4.2

# plot the data
par(mfrow=c(1,1),oma=c(0,0,0,0),mar=c(4,5,1,1),bg="white")
plot(y/n~temp, data = WindowDat1,ylim=c(0,1),xlab="outdoor temperature",
     ylab="Relative frequency of \nopen window",pch=19)
# define and fit a logistic regression model
glmod<-glm(cbind(y,n-y)~temp, data = WindowDat1,family=binomial())
temp.plot <- seq(min(WindowDat1$temp),max(WindowDat1$temp),length=100)
lines(temp.plot,predict(glmod,newdata=data.frame(temp=temp.plot),type="response"))


##################################################
## Section 4.2
##################################################

##################################################
## Example 4.3

##################################################
## Example 4.4

##################################################
## Example 4.5

##################################################
## Example 4.6

##################################################
## Example 4.7

##################################################
## Example 4.8

##################################################
## Example 4.9

dfLMEx1 <- read.csv(file = "data/clothing.csv") 

modGLMEx11.gam <- glm(clo ~ tOut + I(log(tOut)) * sex + tInOp, 
                     data = dfLMEx1, family = "Gamma")
summary(modGLMEx11.gam)

tout <- seq(min(dfLMEx1$tOut),max(dfLMEx1$tOut),
            length=200)
X.m <- data.frame(tOut=tout, tInOp = mean(dfLMEx1$tInOp), sex="male")
X.f <- data.frame(tOut=tout, tInOp = mean(dfLMEx1$tInOp), sex="female")

conf11.f <- predict(modGLMEx11.gam,newdata=X.f,type="response",se.fit=TRUE)
conf11.m <- predict(modGLMEx11.gam,newdata=X.m,type="response",se.fit=TRUE)
conf11.f <- data.frame(tOut = tout, fit = conf11.f$fit, 
                       lwr = conf11.f$fit - 2 * conf11.f$se.fit, 
                       upr = conf11.f$fit + 2 * conf11.f$se.fit, sex="female")
conf11.m <- data.frame(tOut=tout, fit = conf11.m$fit, 
                       lwr = conf11.m$fit - 2 * conf11.m$se.fit, 
                       upr = conf11.m$fit + 2 * conf11.m$se.fit, sex="male")
conf11 <- rbind(conf11.f,conf11.m)

conf<- conf11

plot11 <- ggplot(data=conf, aes(x=tOut, y=fit)) + 
    ylim(0,1.2)+
    geom_line(aes(x=tOut, y=fit, linetype=sex,
                     colour=sex))+
    geom_point(data=dfLMEx1, aes(x=tOut, y=clo,
                                  colour=sex),size=0.5)+ 
    geom_ribbon(aes(tOut, ymin = lwr, ymax = upr, 
                    fill = sex), alpha = .6) +
    xlab(expression(paste("Outdoor air temp. ["^o,"C]")))+
    ylab("Clothing insulation level [CLO]")+
          theme_bw() +	  scale_fill_grey(end=0.5) + scale_color_grey(end=0.5) +
    theme(legend.justification=c(1,0),		
          legend.position=c(.95,.65),
          axis.text.x=element_text(size=9), 
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12),
          axis.title.y=element_text(vjust=1),
          legend.text=element_text(size=18),
          legend.title=element_text(size=18))
plot11

##################################################
## Section 4.3
##################################################

##################################################
## Example 4.10

WindowDat1

mod.binom <- glm(cbind(y, n - y) ~ temp, data = WindowDat1,
                 family = binomial())
(sum.binom <- summary(mod.binom))

## Profile likelihood CI
confint(mod.binom)
## Wald CI
coef(mod.binom)+1.96 * cbind(-sum.binom$coefficients[ ,2],
          sum.binom$coefficients[ ,2])


##################################################
## Section 4.4
##################################################

##################################################
## Example 4.11

set.seed(345)
## Step 1:
tout <- 20
X.f <- data.frame(tOut=tout, tInOp = mean(dfLMEx1$tInOp), 
                  sex="female")
conf11.f <- predict(modGLMEx11.gam,newdata=X.f,type="response",
                    se.fit=TRUE)
## Confidence interval
(CI <- conf11.f$fit + 1.96 * c(-1,1) * conf11.f$se.fit)


## Step 2: Prediction interval
k <-  100000
a <- summary(modGLMEx11.gam)$dispersion
r <- rgamma(k, scale = rnorm(k, mean = conf11.f$fit, 
                                 sd = conf11.f$se.fit) * a, 
                shape=1/a)
## Step 3: Expected value, median, and 95% Prediction interval
c(mean(r), quantile(r,probs = c(0.5,0.025,0.975)))

tout <- seq(min(dfLMEx1$tOut),max(dfLMEx1$tOut),
            length=200)
X.m <- data.frame(tOut=tout, tInOp = mean(dfLMEx1$tInOp), sex="male")
X.f <- data.frame(tOut=tout, tInOp = mean(dfLMEx1$tInOp), sex="female")

conf11.f <- predict(modGLMEx11.gam,newdata=X.f,type="response",se.fit=TRUE)
conf11.m <- predict(modGLMEx11.gam,newdata=X.m,type="response",se.fit=TRUE)

pred11.f <- matrix(ncol=5,nrow=length(tout))
pred11.m <- matrix(ncol=5,nrow=length(tout))
k <-  100000
a <- summary(modGLMEx11.gam)$dispersion
for(i in 1:length(tout)){
    pred11.f[i,2:4] <- quantile(rgamma(k, scale = rnorm(k, mean = conf11.f$fit[i], 
                                                   sd = conf11.f$se.fit[i]) * a, 
                                  shape=1/a),probs = c(0.5,0.025,0.975))
    pred11.m[i,2:4] <- quantile(rgamma(k, scale = rnorm(k, mean = conf11.m$fit[i], 
                                                   sd = conf11.m$se.fit[i]) * a, 
                                  shape=1/a),probs = c(0.5,0.025,0.975))
}
colnames(pred11.f) <- c("tOut","fit","lwr","upr","sex")
pred11.f <- as.data.frame(pred11.f)
pred11.f$tOut<-tout
pred11.f$sex <- "female"
colnames(pred11.m) <- c("tOut","fit","lwr","upr","sex")
pred11.m <- as.data.frame(pred11.m)
pred11.m$tOut<-tout
pred11.m$sex <- "male"

pred11 <- rbind(pred11.f,pred11.m)

pred<- pred11


conf11.f <- data.frame(tOut = tout, fit = conf11.f$fit, 
                       lwr = conf11.f$fit - 2 * conf11.f$se.fit, 
                       upr = conf11.f$fit + 2 * conf11.f$se.fit, sex="female")
conf11.m <- data.frame(tOut=tout, fit = conf11.m$fit, 
                       lwr = conf11.m$fit - 2 * conf11.m$se.fit, 
                       upr = conf11.m$fit + 2 * conf11.m$se.fit, sex="male")
conf11 <- rbind(conf11.f,conf11.m)

conf<- conf11



plot11 <- ggplot(data=conf, aes(x=tOut, y=fit)) + 
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
          legend.text=element_text(size=18),
          legend.title=element_text(size=18))

plot11

##################################################
## Example 4.12

conf.binom <- predict(mod.binom,newdata=data.frame(temp=temp.plot),type="response",se.fit=TRUE)
nrow <- length(temp.plot)
pred.binom <- matrix(ncol=4,nrow=nrow)
k <-  100000
n <- 20
for(i in 1:nrow){
    prob = rnorm(k, mean = conf.binom$fit[i], 
                 sd = conf.binom$se.fit[i])
    prob[prob<0]<-0;prob[prob>1]<-1
    pred.binom[i,2:4] <- quantile(rbinom(k, prob = prob,
                                  size=n),probs = c(0.5,0.025,0.975))/n
}
colnames(pred.binom) <- c("temp","fit","lwr","upr")
pred.binom  <- as.data.frame(pred.binom)
pred.binom$temp<-temp.plot

pred<- pred.binom

conf.binom <- data.frame(temp = temp.plot, fit = conf.binom$fit, 
                       lwr = conf.binom$fit - 2 * conf.binom$se.fit, 
                       upr = conf.binom$fit + 2 * conf.binom$se.fit)
conf<- conf.binom



plot11 <- ggplot(data=conf, aes(x=temp, y=fit)) + 
    ylim(-0.03,1.03)+
    geom_line(aes(x=temp, y=fit))+
    geom_point(data=WindowDat1, aes(x=temp, y=y/n),size=0.5)+ 
    geom_ribbon( aes(temp, ymin = lwr, ymax = upr), alpha = .6) +
    geom_ribbon(data=pred,aes(temp, ymin = lwr, ymax = upr), alpha = .2) +
    xlab("Temperature") +
    ylab("prob")+
          theme_bw() +	 
    theme(legend.justification=c(1,0),		
          legend.position=c(.95,.65),
          axis.text.x=element_text(size=9), 
          axis.text.y=element_text(size=9),
          axis.title=element_text(size=12),
          axis.title.y=element_text(vjust=1),
          legend.text=element_text(size=18),
          legend.title=element_text(size=18))
plot11


##################################################
## Section 4.5
##################################################

##################################################
## Example 4.13

par(mfrow=c(2,2),bg="white")
col.tmp <- (dfLMEx1$sex=="female")*1+1
pt.type <- col.tmp 
pt.type[pt.type==1] <- 19
pt.type[pt.type==2] <- 4

col.tmp[col.tmp==2] <- 0.5
col.tmp[col.tmp==1] <- 0
panel.fun.tmp <- function(x, y, ...)
              panel.smooth(x, y, iter=0, col.smooth=1,lty=2,...)


plot(modGLMEx11.gam,,cex=0.5,col=gray(col.tmp),pch=pt.type,which=1,
      panel = panel.fun.tmp)

plot(modGLMEx11.gam,,cex=0.5,col=gray(col.tmp),pch=pt.type,which=2,
      panel = panel.fun.tmp)
legend("topleft",legend=c("males","females"),pch=c(19,4),pt.cex=0.5,
       col=c(1,gray(0.5)))
plot(modGLMEx11.gam,,cex=0.5,col=gray(col.tmp),pch=pt.type,which=3,
      panel = panel.fun.tmp)

plot(modGLMEx11.gam,,cex=0.5,col=gray(col.tmp),pch=pt.type,which=5,
      panel = panel.fun.tmp)


##################################################
## Example 4.14

par(mfrow=c(1,3),bg="white")
panel.fun.tmp <- function(x, y, ...)
              panel.smooth(x, y, iter=0, col.smooth=1,lty=2,...)

plot(mod.binom,pch=19,cex=0.2,which=1:3, panel = panel.fun.tmp)

##################################################
## Example 4.15

par(bg="white")
plot(WindowOpen ~ OutdoorTemp,data=agrdat1,pch=19,cex=1)
fit.binom.bin <- glm(WindowOpen ~ OutdoorTemp, data = agrdat1,family=binomial)
lines(temp.plot,predict(glmod,newdata=data.frame(temp=temp.plot),type="response"),
      col=gray(0.5),lwd=4,lty=2)
lines(temp.plot,predict(fit.binom.bin,
                        newdata=data.frame(OutdoorTemp=temp.plot),
                        type="response"),col=1,lwd=2)

legend(x=-1.5,y=0.9,legend=c("Aggregated","Binary"),lty=2:1,lwd=2,col=c(gray(0.5),1))


par(mfrow=c(1,3),bg="white")
panel.fun.tmp <- function(x, y, ...)
              panel.smooth(x, y, iter=0, col.smooth=gray(0.5),lty=2,lwd=2,...)
plot(fit.binom.bin,pch=19,,cex=0.2,which=1:3, panel = panel.fun.tmp)

##################################################
## Example 4.16

dat.tmp <- dfLMEx1
dat.tmp$res <- residuals(modGLMEx11.gam, type = "pearson")
cor <- aggregate(res ~ subjId + day, data = dat.tmp,
                 function(x){acf(x,plot=FALSE)$acf[2]})[ ,3]

## Test
t.test(cor)


##################################################
## Section 4.6
##################################################

##################################################
## Example 4.17

modGLMEx11.gam.log <- glm(formula = clo ~ tOut + 
                              I(log(tOut)) * sex + tInOp,
                          family = Gamma(link=log), 
                          data = dfLMEx1)
modGLMEx11.gam.ident <- glm(formula = clo ~ tOut + 
                                I(log(tOut)) * sex + tInOp,
                            family = Gamma(link=identity), 
                            data = dfLMEx1)
AIC(modGLMEx11.gam,modGLMEx11.gam.log,modGLMEx11.gam.ident)

##################################################
## Example 4.18

mod.binom <- glm(cbind(y, n - y) ~ temp, data = WindowDat1,
                 family = binomial())
mod.binom.probit <- glm(formula = cbind(y, n - y) ~ temp, 
                        family = binomial(link=probit), 
                        data = WindowDat1)
mod.binom.cauchit <- glm(formula = cbind(y, n - y) ~ temp, 
                        family = binomial(link=cauchit), 
                        data = WindowDat1)
mod.binom.cloglog <- glm(formula = cbind(y, n - y) ~ temp, 
                         family = binomial(link=cloglog), 
                        data = WindowDat1)
mod.binom.loglog <- glm(formula = cbind(n-y, y) ~ temp, 
                         family = binomial(link=cloglog), 
                        data = WindowDat1)
AIC(mod.binom,mod.binom.probit,mod.binom.cauchit,mod.binom.cloglog,
    mod.binom.loglog)

par(mfrow=c(1,3),bg="white")
panel.fun.tmp <- function(x, y, ...)
              panel.smooth(x, y, iter=0, col.smooth=gray(0.5),lty=2,lwd=2,...)
plot(mod.binom.cauchit,pch=19,cex=1,which=1:3,panel=panel.fun.tmp)

par(mfrow=c(1,1),mar=c(4,5,0,0),oma=c(1,0,1,0),bg="white")
plot(y/n~temp, data = WindowDat1,ylim=c(0,1),xlab="Outdoor temperature",
     ylab="Relative frequency of \nwindow opening",pch=19)
# define and fit a logistic regression model
glmod<-glm(cbind(y,n-y)~temp, data = WindowDat1,family=binomial())
temp.plot <- seq(min(WindowDat1$temp),max(WindowDat1$temp),length=100)
lines(temp.plot,predict(glmod,newdata=data.frame(temp=temp.plot),type="response"),lwd=2)
lines(temp.plot,predict(mod.binom.cauchit,newdata=data.frame(temp=temp.plot),type="response"),col=gray(0.5),lwd=2,lty=2)
legend("bottomright",legend=c("logit-link","Cauchit-link"),col=c(1,gray(0.5)),lty=1:2,lwd=2)

mod.binom.cloglog <- glm(formula = cbind(y, n - y) ~ temp, 
                         family = binomial(link=cloglog), 
                        data = WindowDat1)
mod.binom.loglog <- glm(formula = cbind(n-y, y) ~ temp, 
                         family = binomial(link=cloglog), 
                        data = WindowDat1)

##################################################
## Section 4.7
##################################################

##################################################
## Example 4.19

dfcount <- read.csv(file = "data/clothing_count.csv",sep=";") 
dfcount$delta.t=dfcount$time/dfcount$nobs
head(dfcount[ ,-c(7:8)])


summary(dfcount[ ,c(3:5,9)])

fit.binom <- glm(cbind(clo, nobs - clo) ~ sex, data = dfcount, 
                 family = binomial)
exp(cumsum(coef(fit.binom))) / (1 + exp(cumsum(coef(fit.binom))))

fit.pois <- glm(clo ~ sex, data = dfcount, family = poisson)
exp(cumsum(coef(fit.pois)))


fit.poisOff <- glm(clo ~ sex + offset(log(time)), data = dfcount, 
                   family = poisson)
exp(cumsum(coef(fit.poisOff)))

AIC(fit.binom,fit.pois,fit.poisOff)

lambda.female <- exp(coef(fit.poisOff)[1])
lambda.male <- exp(sum(coef(fit.poisOff)))
p.female <- 1/(1+exp(-coef(fit.binom)[1]))
p.male <- 1/(1+exp(-sum(coef(fit.binom))))
par(mfrow=c(1,2),bg="white")
 matplot(cbind(0:5-0.1,0:5+0.1),cbind(dpois(0:5,lambda.male*6.7),
                   dbinom(0:5, size=5, 
                          prob=p.male)),
                   type="h",col=c(1,gray(0.5)),lty=1,
         main="Male",ylim=c(0,0.8),lwd=3,ylab="Probability",xlab="No. of events")
legend("topright",col=c(1,gray(0.5)),lwd=3,legend=c("Binomial","Poisson"))
matplot(cbind(0:5-0.1,0:5+0.1),cbind(dpois(0:5,lambda.female*6.7),
                   dbinom(0:5, size=5, 
                          prob=p.female)),
                   type="h",col=c(1,gray(0.5)),lty=1,
        main="Female",ylim=c(0,0.8),lwd=3,ylab="Probability",xlab="No. of events")


##################################################
## Section 4.8
##################################################

##################################################
## Example 4.20

modGLMEx11.gam.log <- glm(clo ~ tOut + I(log(tOut)) * sex + 
                              tInOp, family = Gamma(link=log), 
                          data = dfLMEx1)

modGLMEx12.gam.log <- glm(clo ~ tOut + I(log(tOut)) * sex,
                          family = Gamma(link=log), 
                          data = dfLMEx1)

anova(modGLMEx12.gam.log,modGLMEx11.gam.log,test="F")

##################################################
## Example 4.21

anova(mod.binom.cauchit, test = "Chisq")

1 - pchisq(14.7, df = 10)

##################################################
## Example 4.22

1 - pchisq(summary(fit.poisOff)$deviance, df = 134)

fit.poisOffSat <- glm(clo ~ factor(subjId) + offset(log(time)), 
     data = dfcount, family = poisson)

 anova(fit.pois,fit.poisOffSat,test="Chisq")

##################################################
## Section 4.9
##################################################

##################################################
## Example 4.23

mod.quasibinom <- glm(formula = cbind(y, n - y) ~ temp, 
                      family = quasibinomial(link=cauchit), 
                      data = WindowDat1)
summary(mod.quasibinom)

##################################################
## Example 4.24

set.seed(2352)
n <- WindowDat1$n
temp <- WindowDat1$temp
y <- WindowDat1$y

pred <- predict(mod.binom.cauchit, type= "response")
k <- 1000
disp <- numeric(k)
for(i in 1:k){
    y <- rbinom(length(n), prob = pred, size = n)
    disp[i] <- summary(glm(cbind(y, n - y) ~ temp,
                           family = quasibinomial(link = cauchit))
    )$dispersion
}

sum(disp > summary(mod.quasibinom)$dispersion) / k

##################################################
## Section 4.10
##################################################

##################################################
## Example 4.25

rP <- residuals(modGLMEx12.gam.log,type="pearson")
(ratio <- mean(rP[dfLMEx1$sex=="female"]^2)/
     mean(rP[dfLMEx1$sex=="male"]^2))

## Negative log.likelihood function
nll <- function(pars,X.mu,X.disp,y){
    ## mean value parameters
    beta <- pars[1:dim(X.mu)[2]]
    ## Dispersion
    a <- exp(X.disp %*% pars[-c(1:dim(X.mu)[2])])
    ## Linear predictor
    eta <- X %*% beta
    ## mean value
    mu <- exp(eta)
    ## scale
    s <- mu * a
    - sum(dgamma(y, shape = 1 / a, scale = s, log = TRUE))
}

## Design matrix for mean value model
X <- model.matrix(modGLMEx12.gam.log)
## Design matrix for dispersion model 
X.disp <- matrix(1,nrow=dim(X)[1],ncol=2)
X.disp[ ,2] <- dfLMEx1$sex=="female"

## Initial paramters 
pars <- rep(0,dim(X)[2]+dim(X.disp)[2])
## Find optimal paramters
opt.gam <- nlminb(pars, nll, X.mu = X, X.disp = X.disp, y = dfLMEx1$clo)
## Log-likelihood
-opt.gam$objective
## Campare with gamma model 
logLik(modGLMEx12.gam.log)

 library(numDeriv)
H <- hessian(nll ,opt.gam$par, X.mu = X, X.disp = X.disp, 
             y = dfLMEx1$clo)
se <- sqrt(diag(solve(H)))
par.tab <- cbind(opt.gam$par, se,opt.gam$par / se)
colnames(par.tab) <- c("Est", "se", "z.value")
rownames(par.tab) <- c(names(coef(modGLMEx12.gam.log)),
                       "log(lambda)", "log(gamma)")
round(par.tab, digits=3)

##################################################
## Section 4.11
##################################################

##################################################
## Example 4.26

## read data
dfCeilingFan <- read.csv(file = "data/exCeilingFan_data.csv") 

# tell R that its all factors
dfCeilingFan$fanSpeed <- factor(dfCeilingFan$fanSpeed, level=c("0","1","2")) #,"3"
dfCeilingFan$subjId <- factor(dfCeilingFan$subjId) #21 subjects
dfCeilingFan$TSV <- factor(dfCeilingFan$TSV, level=c("0", "1", "2"))
dfCeilingFan$fanType <- factor(dfCeilingFan$fanType)

summary(dfCeilingFan[ ,-1])

tab <- aggregate(fanSpeed ~ TSV, data = dfCeilingFan,table)
colnames(tab) <- c("TSV","fanSpeed")
tab
(chi.tab <- chisq.test(tab[ ,-1]))

## Warning message:
## In chisq.test(tab[, -1]) : Chi-squared approximation may be incorrect

## Observed proportions
p1 <- tab[1,-1] / sum(tab[1,-1])
p2 <- tab[2,-1] / sum(tab[2,-1])
p3 <- tab[3,-1] / sum(tab[3,-1])
(l1 <- sum(tab[1,-1] * log(p1)) + sum(tab[2,-1] * log(p2)) +
     sum(tab[3,-1] * log(p3)))

p <- colSums(tab[ ,-1]) / sum(colSums(tab[ ,-1]))
(l0 <- sum(tab[1,-1] * log(p)) + sum(tab[2,-1] * log(p))+
     sum(tab[3,-1] * log(p)))

(Q <- 2 * (l1 - l0))
1 - pchisq(Q, df = 6-2)

##################################################
## Example 4.27

fit1.nom <- clm(fanSpeed ~ 1, nominal = ~ TSV * fanType, 
                data = dfCeilingFan)
fit2.nom <- clm(fanSpeed ~ 1, nominal = ~ TSV + fanType, 
                data = dfCeilingFan)
fit3.nom <- clm(fanSpeed ~ 1, nominal = ~ TSV, 
                data = dfCeilingFan)
fit4.nom <- clm(fanSpeed ~ 1, nominal = ~ 1, 
                data = dfCeilingFan)

anova(fit1.nom, fit2.nom, fit3.nom, fit4.nom)

summary(fit3.nom)

## Just to compare not to be included
1/(1+exp(-coef(fit3.nom)[1]))-97/137
1/(1+exp(-sum(coef(fit3.nom)[2])))-117/137

1/(1+exp(-sum(coef(fit3.nom)[c(1,3)])))-40/74
1/(1+exp(-sum(coef(fit3.nom)[c(1,5)])))-8/26

1/(1+exp(-sum(coef(fit3.nom)[c(2,4)])))-64/74
1/(1+exp(-sum(coef(fit3.nom)[c(2,6)])))-16/26

##################################################
## Example 4.28

fit3.ord <- clm(fanSpeed ~ TSV, data = dfCeilingFan)
summary(fit3.ord)

dfCeilingFan$numTSV <- as.numeric(dfCeilingFan$TSV) - 1
fit32.ord <- clm(fanSpeed ~ numTSV, data = dfCeilingFan)
anova(fit32.ord,fit3.ord)

summary(fit32.ord)

pred32.ord <- predict(fit32.ord,newdata=data.frame(numTSV=(0:2)))
pred3.ord <- predict(fit3.ord,newdata=data.frame(TSV=factor(0:2)))
pred3.nom <- predict(fit3.nom,newdata=data.frame(TSV=factor(0:2)))


par(mfrow=c(1,2),bg="white")
col=gray((0:2)/4)
lty=1:3
pch=17:19
cex=1
p3.ord <- t(apply(pred3.ord$fit,1,cumsum))[ ,-3]
p32.ord <- t(apply(pred32.ord$fit,1,cumsum))[ ,-3]
p3.nom <- t(apply(pred3.nom$fit,1,cumsum))[ ,-3]
matplot(0:2,p3.ord,ylim=c(0,1),lty=lty[1],col=col[1],
        lwd=3,type="b",pch=pch[1],ylab="P(X<C_j)",xlab="TSV",cex=cex)
matlines(0:2,p32.ord,ylim=c(0,1),lty=lty[2],col=col[2],
        lwd=3,type="b",pch=pch[2],cex=cex)
matlines(0:2,p3.nom,ylim=c(0,1),lty=lty[3],col=col[3],
        lwd=3,type="b",pch=pch[3],cex=cex)
legend("bottomleft",legend=c("Ordinal numeric TSV","Ordinal","Nominal"),col=col[1:3],lty=lty[1:3],pch=pch[1:3],lwd=3,cex=cex)

odds32.ord <- p32.ord/(1-p32.ord)
odds3.ord <- p3.ord/(1-p3.ord)
odds3.nom <- p3.ord/(1-p3.nom)

OR32.ord <- odds32.ord[ ,1]/odds32.ord[ ,2]
OR3.ord <- odds3.ord[ ,1]/odds3.ord[ ,2]
OR3.nom <- odds3.nom[ ,1]/odds3.nom[ ,2]

matplot(0:2,log((cbind(odds32.ord,odds3.ord,odds3.nom))),
        type="b",col=col[c(1,1,2,2,3,3)],lwd=3,lty=lty[c(1,1,2,2,3,3)],
        pch=pch[c(1,1,2,2,3,3)],
        ylab="log-odds",xlab="TSV",cex=cex)



## Just to compare not to be included
pred32.ord <- predict(fit32.ord, newdata = data.frame(TSV = factor(0:2)))
pred3.ord <- predict(fit3.ord, newdata = data.frame(TSV = factor(0:2)))
pred3.nom <- predict(fit3.nom, newdata = data.frame(TSV = factor(0:2)))

## ProbOdds
1/(1+exp(-coef(fit32.ord)[1]))-pred32.ord$fit[1,1]
1/(1+exp(-coef(fit32.ord)[2]))-sum(pred32.ord$fit[1,1:2])

1/(1+exp(-coef(fit32.ord)[1]+coef(fit32.ord)[3]))-pred32.ord$fit[2,1]
1/(1+exp(-coef(fit32.ord)[2]+coef(fit32.ord)[3]))-sum(pred32.ord$fit[2,1:2])

1/(1+exp(-coef(fit32.ord)[1]+coef(fit32.ord)[3]*2))-pred32.ord$fit[3,1]
1/(1+exp(-coef(fit32.ord)[2]+coef(fit32.ord)[3]*2))-sum(pred32.ord$fit[3,1:2])

pred3.ord
1/(1+exp(-coef(fit3.ord)[1]))-pred3.ord$fit[1,1]
1/(1+exp(-coef(fit3.ord)[2]))-sum(pred3.ord$fit[1,1:2])

1/(1+exp(-coef(fit3.ord)[1]+coef(fit3.ord)[3]))-pred3.ord$fit[2,1]
1/(1+exp(-coef(fit3.ord)[2]+coef(fit3.ord)[3]))-sum(pred3.ord$fit[2,1:2])

1/(1+exp(-coef(fit3.ord)[1]+coef(fit3.ord)[4]))-pred3.ord$fit[3,1]
1/(1+exp(-coef(fit3.ord)[2]+coef(fit3.ord)[4]))-sum(pred3.ord$fit[3,1:2])

##################################################
## Section 4.12
##################################################
