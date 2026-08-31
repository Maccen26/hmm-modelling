library(MASS)
##################################################
## Section 7.1
##################################################

##################################################
## Example 7.1

##################################################
## Figure 7.1

x <- c(0.63, 0.58, 4.71, 0.58, 115.75, 0.71, 0.04, 0.67, 0.58, 115.75, 0.71, 0.04, 14.13,
       14.08, 27.04, 4.92, 36.83, 7, 1.63, 28, 6.92, 1.71, 1.75, 6.88, 0.63, 0.58, 0.54,
       0.54, 0.5, 0.5, 0.46, 0.63, 0.58, 0.46, 0.54, 6.88, 3.04, 3, 18.83, 18.79, 5.75,
       0.67, 2.13, 2.08, 8.75, 26.5, 1.83, 6.88, 13.83, 13.79, 18.83, 18.79, 5.75, 0.67,
       2.13, 2.08, 8.75, 26.5, 1.83, 6.88, 2.83, 0.71, 0.04, 0.88, 0.83, 0.79, 4.71, 4.67, 
       4.63, 0.92, 0.88, 6.88)

BinnedObservations = hist(x,freq = FALSE,
     main="",
     xlab="Time (days)", 
     border="black", 
     col="gray",
     xlim=c(0,50),
     ylim=c(0,0.15),
     las=1, 
     breaks=c(0,5,10,15,20,25,30,35,40,45,50,200),cex.lab=0.8, cex.axis=0.8, cex.main=0.8, cex.sub=0.8, plot = TRUE)

weibullParameters <- fitdistr(x, densfun="weibull", lower = 0)
weibullParameters = unlist(weibullParameters)
y <- dweibull(c(0.1:0.01:50), weibullParameters[1], 
              weibullParameters[2])

lines(c(0.1:0.01:50),y,col=gray(0.25), type="l",lwd=2,lty=2)



##################################################
## Figure 7.2

h <- hist(x, plot=FALSE, breaks=seq(-0.7, 200, 1))
cumProb  <- cumsum(h$density)
cumProb <- rep(1,length(cumProb)) - cumProb
Ticks  <- h$breaks
plot(Ticks[0:50],cumProb[0:50],col = "black",
     xlim=c(0, 50),ylim=c(0, 1),
     xlab = "Time (day)", ylab = "Probability",
     cex.lab=0.8, cex.axis=0.8, cex.main=0.8, cex.sub=0.8
     )

weibullParameters <- fitdistr(x, densfun="weibull", lower = 0)
weibullParameters = unlist(weibullParameters)

y <- pweibull(c(0.01:0.01:50), weibullParameters[1], 
              weibullParameters[2])
y <- rep(1,length(y)) - y
lines(c(0.01:0.01:50),y,lwd=2,col=gray(0.25))
lines(c(0,50),0.3*c(1,1),lty=4,lwd=2,col=gray(0.5))



##################################################
## Example 7.2

##################################################
## Figure 7.3

# Duration of absence periods
cumProb <- c(1, 0.96, 0.92, 0.585, 0.25, 0.165, 0.08, 0.06, 0.04, 0.025, 0.01, 0.005, 0,0)
Ticks <- c(0, 0.2, 0.7,	1.2, 1.7, 2.2,2.7, 3.2,	3.75, 4.25, 4.75, 5.25,	5.75, 6.25)

WeibullFittedTicks <- seq(0,6.5,by=0.1)

WeibullFitted <- c(0.96, 0.95, 0.93, 0.92, 0.9, 0.87, 0.85, 0.81, 0.77, 0.73, 0.68,
                   0.63, 0.57, 0.51, 0.46,
                   0.4, 0.34, 0.29, 0.25, 0.21, 0.17, 0.14, 0.11, 0.09, 0.07, 0.06,
                   0.05, 0.04, 0.03, 0.02,
                   0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

plot(Ticks, cumProb, col = "black", xlim = c(0, 7), ylim = c(0, 1),
     xlab = "Time (hr)", ylab = "Probability", lwd = 2, cex.lab = 0.8,
     cex.axis = 0.8, cex.main = 0.8, cex.sub=0.8)

lines(WeibullFittedTicks,WeibullFitted,col=gray(0.25))


##################################################
## Example 7.3

##################################################
## Figure 7.4
# Plug loads

FittedTicks <- c(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93, 99, 105,
                 111, 120)

Fitted <- c(10.6, 5.3, 4.7, 4.3, 4.1, 3.9, 3.7, 3.6, 3.5, 3.5, 3.3, 3.2, 3.1, 3, 3, 2.9, 2.9, 2.8, 2.8,
            2.7, 2.7, 2.7, 2.6, 2.6, 2.6)

plot(FittedTicks, Fitted, type = "l", col = "black", xlim=c(0, 120), ylim = c(0, 12),
     xlab = "Duration of absence (hr)",  ylab = expression("Plug load intensity (W/" ~ (m^{2})),
     lwd = 2, cex.lab=0.8, cex.axis=0.8, cex.main=0.8, cex.sub=0.8)

##################################################
## Figure 7.5
# Light switch off                
FittedTicks <- c(0.1, 0.2, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6,
                 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6, 5.8, 6, 6.2, 6.4, 6.6, 6.8, 7, 7.2, 7.4, 
                 7.6, 7.8, 8)

Fitted <- c(0.31, 0.35, 0.45, 0.49, 0.56, 0.63, 0.7, 0.76, 0.81, 0.85, 0.88, 0.91, 0.93, 0.95, 0.96, 0.97,
            0.98, 0.98, 0.99, 0.99, 0.99, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) 

plot(FittedTicks, Fitted, type = "l", col = "black", xlim = c(0, 7), ylim = c(0, 1), 
     xlab = "Duration of absence (day)", ylab = "Probability of light switch off", lwd = 2,
     cex.lab=0.8, cex.axis=0.8, cex.main=0.8, cex.sub=0.8)

##################################################
## Section 7.2
##################################################

##################################################
## Section 7.3
##################################################

##################################################
## Example 7.4

##################################################
## Section 7.4
##################################################

##################################################
## Example 7.5

## Extract a small example for illustration
dat.surv <- read.table(file="data/survocc.csv",sep=";")
dat.surv$status <- (dat.surv$censor==0)*1
dat.surv.ex <- dat.surv[dat.surv$room==4206, ][1:15, ]
dat.ex <- data.frame(time=dat.surv.ex$time, censor=dat.surv.ex$censor)

##
dat.ex


t <- sort(unique(dat.ex$time))
d <- t * 0
c <- t * 0
for(i in 1:length(t)){
    d[i] <- sum(dat.ex$censor[dat.ex$time==t[i]]==0)
    c[i] <- sum(dat.ex$censor[dat.ex$time==t[i]])
}


R <- rep(15,length(t))
R <- R - cumsum(d) - cumsum(c)
R <- c(15,R[-length(R)])

S <- cumprod(1-d/R)
survex.tab <- data.frame(t,d,c,R,S)
survex.tab

##################################################
## Example 7.6

attach(survex.tab)
se.loglog <- sqrt(1 / log(S)^2 * cumsum(d / (R * (R - d))))
lower.loglog <- log(-log(S)) - qnorm(0.975) * se.loglog
upper.loglog <- log(-log(S)) + qnorm(0.975) * se.loglog

survex.tab$lower <- exp(-exp(upper.loglog))
survex.tab$upper <- exp(-exp(lower.loglog))

survex.tab


 library(survival)
Surv.Ex <- survfit(Surv(time,censor == 0) ~ 1,
                   data = dat.ex, conf.type = "log-log")


##################################################
## Figure 7.6

par(mfrow=c(1,1),mar=c(4,5,2,2))
matplot(c(0,survex.tab$t),rbind(c(1,1,1),
                                survex.tab[ ,c("S","upper","lower")]),
        type="s",col=gray(c(0,0.2,0.2)),
        lty=c(1,2,2),lwd=2,xlab="Time",ylab=expression(hat(S)))
lines(c(0,4),0.5*c(1,1),lty=3,col=gray(0.4),lwd=2)
lines(1.5*c(1,1),c(0,1),lty=4,col=gray(0.5),lwd=2)
legend("topright",legend=c("KM-estimate","95% CI","S=0.5","t=3"),lty=c(1,2,3,4),col=gray(c(0,0.2,0.4,0.6)),lwd=2,bty="o",
       bg="white")


quantile(Surv.Ex)

##################################################
## Example 7.7

dat.surv.room <- dat.surv[dat.surv$room==5203 | dat.surv$room==4206, ]

par(mfrow=c(1,1),mar=c(4,5,2,2),bg="white")
## KM-plot
## library(survival)
Surv.ByRoom <- survfit(Surv(time,censor == 0) ~ room,
                       data = dat.surv.room, conf.type = "log-log")

plot(Surv.ByRoom, conf.int = TRUE, xlab = "Tine [Hours]", 
     col = gray(c(0, 0.5)), ylab=expression(hat(S)),lwd=2)

## Median 
abline(h = 0.5, col = gray(0.25), lwd = 2,lty=3)
## Time t=5
abline(v = 5, col = gray(0.5), lwd = 2,lty = 4)
legend("topright", col = gray(c(0,0.5)),lty=c(1,1), 
       c("Room 5203","Room 4206"), lwd = 2)

Surv.ByRoom

##################################################
## Section 7.5
##################################################

##################################################
## Example 7.8

survdiff(Surv(time, censor == 0) ~ room, data = dat.surv.room)

##################################################
## Section 7.6
##################################################

##################################################
## Example 7.9

n <- dim(dat.ex)[1]
nc <- sum(dat.ex$censor)
c(MLE = log(n / (n - nc)) + log(mean(dat.ex$time)), 
  SE = sqrt(1 / (n - nc)))

summary(survreg(Surv(time, censor == 0) ~ 1, data = dat.ex, 
                dist = "exponential"))

##################################################
## Example 7.10

##################################################
## Example 7.11

dat.surv.1room <- dat.surv[dat.surv$room==6207, ]
fit0.exp <- survreg(Surv(time,status) ~ 1,
                    data = dat.surv.1room, dist="exponential")
fit0.wei <- survreg(Surv(time,status) ~ 1, 
                    data = dat.surv.1room, dist = "weibull")
fit0.llogis<-survreg(Surv(time,status) ~ 1, 
                     data = dat.surv.1room, 
                     dist = "loglogistic")

nll.exp <- function(pars, t, d){
    theta <- exp(pars)
    h <- 1 / theta
    H <-  t / theta
    -  sum(d * log(h)) + sum(H)
}

pars <-0
opt.exp <- nlminb(pars, nll.exp, t = dat.surv.1room$time, d = dat.surv.1room$status)

nll.wei <- function(pars, t, d){
    theta <- exp(pars[1])
    sigma <- exp(pars[2])
    h <- t^(1/sigma-1)/(sigma * theta^(1/sigma))
    H <-  (t / theta)^(1/sigma)
    -  sum(d * log(h)) + sum(H)
}

pars <- c(0,log(1.19))
opt.wei <- nlminb(pars, nll.wei, t = dat.surv.1room$time, d = dat.surv.1room$status)

nll.llogis <- function(pars, t, d){
    theta <- exp(pars[1])
    sigma <- exp(pars[2])
    h <- 1/(sigma * t) * (t/theta)^(1/sigma)/(1+ (t / theta)^(1/sigma))
    H <-  log(1+(t/theta)^(1/sigma))
    -  sum(d * log(h)) + sum(H)
}

pars <- c(0,log(1.19))
opt.llogis <- nlminb(pars, nll.llogis, t = dat.surv.1room$time, d = dat.surv.1room$status)


anova(fit0.exp, fit0.wei)

AIC(fit0.wei, fit0.llogis)

theta <- exp(c(opt.exp$par[1],opt.wei$par[1],opt.llogis$par[1]))
sigma <- exp(c(NA,opt.wei$par[2],opt.llogis$par[2]))

E <- c(theta[1],theta[2]*gamma(1+sigma[2]),theta[3]*sigma[3]*pi/sin(pi*sigma[3]))
Med <- c(theta[1] * log(2), theta[2]*log(2)^sigma[2], theta[3])
ll <- -c(opt.exp$objective,opt.wei$objective,opt.llogis$objective)

tab <- data.frame(theta,sigma,ll,E,Med)
rownames(tab) <- c("Exponential","Weibull","log-logistic")
tab


t <- seq(0,max(dat.surv.1room$time),by=0.01)
h0.exp <- exp(-coef(fit0.exp)[1])*rep(1,length(t))
log.h.w <- function(theta, sigma, t, X){
    - log(sigma) + (1/sigma-1) * log(t) -theta /sigma
}

H.w <- function(theta,sigma,t){
    t^(1/sigma) * exp(- theta/sigma)
}

H.log <- function(theta,sigma,t){
    mu <-theta
    z <- (log(t) - mu) / sigma
    log(1+exp(z))
}

log.h.log <- function(theta, sigma, t){
    mu <- theta
    z <- (log(t) - mu) / sigma
    - log(sigma) + z - log(1+exp(z)) - log(t)
}

h0.wei <- exp(log.h.w(coef(fit0.wei)[1], fit0.wei$scale, t))
h0.llog <- exp(log.h.log(coef(fit0.llogis), fit0.llogis$scale, t))

col.tmp <- gray(c(0.2,0.35,0.5))
par(mfrow=c(1,2),bg="white")
matplot(t,cbind(h0.exp,h0.wei,h0.llog),type="l",lty=2:4,lwd=3,ylab="Hazard function",
        xlab="Time [hours]",col=col.tmp)
legend("topright",legend=c("Exponential","Weibull","Log-logistic"),
       col=col.tmp,lty=2:4,lwd=3)
Surv.fit <- survfit(Surv(time,status) ~ 1,
                        data = dat.surv.1room, conf.type = "log-log")
plot(Surv.fit,col=1,conf.int=FALSE,mark.time=TRUE,ylab="Survival function",
        xlab="Time [hours]",lwd=3)
lines(t,exp(-h0.exp*t),lwd=3,col=col.tmp[1],lty=2)
lines(t,exp(-H.w(coef(fit0.wei)[1], fit0.wei$scale, t)),col=col.tmp[2],lwd=3,lty=3)
lines(t,exp(-H.log(coef(fit0.llogis)[1], fit0.llogis$scale, t)),col=col.tmp[3],lwd=3,lty=4)

legend("topright",legend=c("KM-estimate"),col=1,lty=1,pch=3)


##################################################
## Example 7.12

fit0.llogish <- survreg(Surv(time,status) ~ hour, data = dat.surv.1room, 
                        dist = "loglogistic")

anova(fit0.llogis,fit0.llogish)

summary(fit0.llogish)

##################################################
## Example 7.13

fit1.llogis <- survreg(Surv(time, censor) ~ factor(room), 
                        data = dat.surv.room, dist = "loglogistic")
anova(fit1.llogis)

coef(fit1.llogis)

##################################################
## Section 7.7
##################################################

##################################################
## Example 7.14

exp(coef(fit0.llogish)[2])
exp(confint(fit0.llogish)[2, ])

exp(coef(fit0.llogish)[2] * fit0.llogish$scale)

##################################################
## Section 7.8
##################################################

##################################################
## Example 7.15

## CS from Dobson
rc.exp <- H.w(coef(fit0.exp)[1], 1, dat.surv.1room$time)
rc.exp[dat.surv.1room$status==0] <- rc.exp[dat.surv.1room$status==0] + log(2)

rc.wei <- H.w(coef(fit0.wei)[1], fit0.wei$scale, dat.surv.1room$time)
rc.wei[dat.surv.1room$status==0] <- rc.wei[dat.surv.1room$status==0] + log(2)


rc.log <- H.log(coef(fit0.llogis)[1], fit0.llogis$scale, dat.surv.1room$time)
rc.log[dat.surv.1room$status==0] <- rc.log[dat.surv.1room$status==0] + log(2)


rc.logh <- H.log(coef(fit0.llogish)[1]+coef(fit0.llogish)[2]*dat.surv.1room$hour, fit0.llogish$scale, dat.surv.1room$time)
rc.logh[dat.surv.1room$status==0] <- rc.logh[dat.surv.1room$status==0] + log(2)


n <- length(rc.exp)
x <- qexp(((1:n)-0.5)/n)

ylim <- range(c(rc.exp,rc.wei,rc.log,rc.logh))
color <- dat.surv.1room$status==0
color[ dat.surv.1room$status==0] <- gray(0.5)
color[ dat.surv.1room$status!=0] <- gray(0)
pty.tmp <- dat.surv.1room$status==0
pty.tmp[ dat.surv.1room$status==0] <- 4
pty.tmp[ dat.surv.1room$status!=0] <- 19


par(mfrow=c(1,4),mar=c(3,1,1,0),oma=c(3,4,3,3),bg="white")
Index <- sort(rc.exp,index.return=TRUE)$ix
plot(x,sort(rc.exp),ylim=ylim,main="Exponential",axes=FALSE,cex=0.5,col=color[Index],
     pch=pty.tmp[Index])
abline(a=0,b=1)
box()
axis(1)
axis(2)
mtext("Cox-Snell residuals",side=2,line=2,outer=TRUE)

Index <- sort(rc.wei,index.return=TRUE)$ix
plot(x,sort(rc.wei),ylim=ylim,main="Weibull",axes=FALSE,cex=0.5,col=color[Index], pch=pty.tmp[Index])
abline(a=0,b=1)
box()
axis(1)

Index <- sort(rc.log,index.return=TRUE)$ix
plot(x,sort(rc.log),ylim=ylim,main="Log-logistic",axes=FALSE,cex=0.5,col=color[Index], pch=pty.tmp[Index])
abline(a=0,b=1)
box()
axis(1)

Index <- sort(rc.logh,index.return=TRUE)$ix
plot(x,sort(rc.logh),ylim=ylim,main="Log-logistic 2",axes=FALSE,cex=0.5,col=color[Index], pch=pty.tmp[Index])
abline(a=0,b=1)
box()
axis(1)
mtext("Standard exponential quantiles",side=1,line=1,outer=TRUE)
legend("topleft",pch=c(19,4),col=gray(c(0,0.5)),legend=c("non-censored","censored"))



##################################################
## Section 7.9
##################################################

##################################################
## Example 7.16

 library(tramME)
dat.survReduced <- dat.surv[dat.surv$room=="5202" | dat.surv$room=="4205" | dat.surv$room=="4206" | dat.surv$room=="5201", ]
dat.survReduced$room <- factor(dat.survReduced$room)
dat.survReduced$weekday <- factor(dat.survReduced$weekday)
dat.survReduced$status <- (dat.survReduced$censor==0)*1
fit0Surv.wei<-survreg(Surv(time,status) ~ 1+room, 
                        data=dat.survReduced, dist="weibull")
logLik(fit0Surv.wei)
coef(fit0Surv.wei)

(fit1Tram.wei<-SurvregME(Surv(time, status) ~ 1 + room, 
                       data = dat.survReduced, dist = "weibull"))


fit1Tram.wei$opt$par

alpha <- fit1Tram.wei$opt$par[1:2]
beta <- fit1Tram.wei$opt$par[-(1:2)]
c(-alpha[1], beta) / alpha[2]

##################################################
## Example 7.17

dat.surv$room <- factor(dat.surv$room)
dat.surv$weekday <- factor(dat.surv$weekday)
dat.surv$status <- (dat.surv$censor==0)*1

fit0m.llogis<-survreg(Surv(time,status) ~ 1+room, data=dat.surv, dist="loglogistic")
fit0m.wei<-survreg(Surv(time,status) ~ 1+room, data=dat.surv, dist="weibull")
fit0m.exp<-survreg(Surv(time,status) ~ 1, data=dat.surv, dist="exponential")


fit1m.llogis<-SurvregME(Surv(time,status) ~ 1+room, data=dat.surv, dist="loglogistic")
fit1m.wei<-SurvregME(Surv(time,status) ~ 1+room, data=dat.surv, dist="weibull")
fit1m.exp<-SurvregME(Surv(time,status) ~ 1, data=dat.surv, dist="exponential")

alpha <- fit1m.llogis$opt$par[1:2]
beta <- fit1m.llogis$opt$par[-(1:2)]
fit1m.llogis<-SurvregME(Surv(time,status) ~ 1 +(1|room), data=dat.surv, dist="loglogistic")

fit1m.llogis<-SurvregME(Surv(time,status) ~ 1 , data=dat.surv, dist="loglogistic")
fit1m.exp<-SurvregME(Surv(time,status) ~ 1 , data=dat.surv, dist="exponential")
fit1m.wei<-SurvregME(Surv(time,status) ~ 1 , data=dat.surv, dist="weibull")
fit0m.wei<-survreg(Surv(time,status) ~ 1 , data=dat.surv, dist="weibull")

fit0ME.llogis<-SurvregME(Surv(time,status) ~ 1, data=dat.surv, dist="loglogistic")
fit0ME.wei<-SurvregME(Surv(time,status) ~ 1, data=dat.surv, dist="weibull")
fit0ME.exp<-SurvregME(Surv(time,status) ~ 1, data=dat.surv, dist="exponential")


fit1.llogis<-SurvregME(Surv(time,status) ~ 1 +(1|room), data=dat.surv, dist="loglogistic")
fit1.wei<-SurvregME(Surv(time,status) ~ 1 +(1|room), data=dat.surv, dist="weibull")
fit1.exp<-SurvregME(Surv(time,status) ~ 1 +(1|room), data=dat.surv, dist="exponential")

I <- dat.surv$room == "4202" | dat.surv$room == "4205" | dat.surv$room == "4206" | 
    dat.surv$room == "5201"  
dat.surv.4rooms <- dat.surv[I, ] 
fit0m.llogis<-survreg(Surv(time,status) ~ 1, data=dat.surv.4rooms, dist="loglogistic")
fit0m.wei<-survreg(Surv(time,status) ~ 1, data=dat.surv.4rooms, dist="weibull")
fit0m.exp<-survreg(Surv(time,status) ~ 1, data=dat.surv.4rooms, dist="exponential")

fit0ME.llogis<-SurvregME(Surv(time,status) ~ 1, data=dat.surv.4rooms, dist="loglogistic")
fit0ME.wei<-SurvregME(Surv(time,status) ~ 1, data=dat.surv.4rooms, dist="weibull")
fit0ME.exp<-SurvregME(Surv(time,status) ~ 1, data=dat.surv.4rooms, dist="exponential")

fit1.llogis<-SurvregME(Surv(time, status) ~ 1 + (1 | room), 
                       data = dat.surv.4rooms, dist = "loglogistic")
fit1.wei<-SurvregME(Surv(time,status) ~ 1 + (1|room),
                    data=dat.surv.4rooms, dist="weibull")
fit1.exp<-SurvregME(Surv(time,status) ~ 1 + (1|room), 
                    data=dat.surv.4rooms, dist="exponential")

ranef(fit1.llogis)


Surv.ByRoom <- survfit(Surv(time,status) ~ room,
                        data = dat.surv[ I,])
Surv.Est <- survfit(Surv(time,status) ~ 1,
                        data = dat.surv[ I,])
t <- seq(0,max(dat.surv[ I,"time"]),by=0.1)
par(mfrow=c(1,1),bg="white")
plot(Surv.ByRoom,col=gray(c(0.2,0.3,0.4,0.5)),xlab="Hours")
lines(Surv.Est,col=1,lty=1,lwd=4,conf.int=FALSE)

sig <- fit0ME.llogis$opt$par[2]
theta <- fit0ME.llogis$opt$par[1]
survLLogis.ME <- function(t,theta,sig){
    (1+exp((log(t)*sig+theta)))^-1
}

lines(t,survLLogis.ME(t,fit0ME.llogis$opt$par[1],fit0ME.llogis$opt$par[2]),
      col=1,lwd=4,lty=2)
lines(t,survLLogis.ME(t,fit1.llogis$opt$par[1]-
                        ranef(fit1.llogis)$room[[1]][1],
                      fit1.llogis$opt$par[2]),col=gray(0.2),lwd=2,lty=2)
lines(t,survLLogis.ME(t,fit1.llogis$opt$par[1]-
                        ranef(fit1.llogis)$room[[1]][2],fit1.llogis$opt$par[2]),col=gray(0.3),lwd=2,lty=2)
lines(t,survLLogis.ME(t,fit1.llogis$opt$par[1]-
                        ranef(fit1.llogis)$room[[1]][3],fit1.llogis$opt$par[2]),col=gray(0.4),lwd=2,lty=2)
lines(t,survLLogis.ME(t,fit1.llogis$opt$par[1]-
                        ranef(fit1.llogis)$room[[1]][4],
                      fit1.llogis$opt$par[2]),col=gray(0.5),lwd=2,lty=2)
len <- legend("topright",legend=c("Prior","Room 1","Room 2","Room 3","Room 4"),lty=c(1,1,1,1),col=gray(c(0,0.2,0.3,0.4,0.5)),lwd=c(4,1,1,1,1))
len <- legend(x=len$rect$left,
              y=len$rect$top,
              legend=c("Kaplan Meier","Model"),lty=c(1,2),col=c(1,1),
              xjust=1)



c(AIC(fit0m.llogis),AIC(fit1.llogis))


##################################################
## Section 7.10
##################################################
