par(mfrow=c(1,1),mar=c(0,0,0,0),oma=c(0,0,2,0))
lwd<-2
cex<-2
plot(c(0,1),c(0,1),col="white",axes=FALSE,xlab="",ylab="")


draw.box <- function(upper,lower,right,left,lwd,text="",cex=1){
    lines(c(left,right),c(upper,upper),lwd=lwd)
    lines(c(left,right),c(lower,lower),lwd=lwd)
    lines(c(left,left),c(upper,lower),lwd=lwd)
    lines(c(right,right),c(upper,lower),lwd=lwd)
    text((left+right)/2, (lower+upper)/2,labels=text,cex=cex)
}



plot(c(0,1),c(0,1),col="white",axes=FALSE,xlab="",ylab="")

upper2 <- 1
lower2 <- 1-1/7
right2 <- 1
left2 <- 0
draw.box(upper2,lower2,right2,left2,lwd,text="Chap 2, Likelihood",cex=2)


upper3 <- lower2-1/7
lower3 <- upper3-1/7
right3 <- 0.2+0.1
left3 <- 0+0.1
draw.box(upper3,lower3,right3,left3,lwd,text="Chap 3,  General \n Linear Models",cex=2)


upper4 <- 1-4/7
lower4 <- 1-5/7
right4 <- 0.2+0.1
left4 <- 0+0.1
draw.box(upper4,lower4,right4,left4,lwd,text="Chap 4:  Generalized \n Linear Models",cex=2)


upper5 <- 1-4/7
lower5 <- 1-5/7
right5 <- 0.5+0.1
left5 <- 0.3+0.1
draw.box(upper5,lower5,right5,left5,lwd,text="Chap 5: General Linear \n Mixed effect Models",cex=2)


upper6 <- 1-6/7
lower6 <- 1-7/7
right6 <- 0.35+0.1
left6 <- 0.1+0.15
draw.box(upper6,lower6,right6,left6,lwd,text="Chap 6: Hierarchical \n Models",cex=2)


upper7 <- upper6#lower2-1/7
lower7 <- lower6#upper3-1/7
right7 <- 0.2
left7 <- 0#1-0.2
draw.box(upper7,lower7,right7,left7,lwd,text="Chap 7: Survival\n analysis",cex=2)


upper8 <- lower2-1/7
lower8 <- upper8-1/7
right8 <- 0.8#left7-0.1
left8 <- right8-0.2
draw.box(upper8,lower8,right8,left8,lwd,text="Chap 8: Linear Time\n Series",cex=2)


upper9 <- lower8-1/7
lower9 <- upper9-1/7
right9 <- 1#right5+0.35
left9 <- right9-0.25
draw.box(upper9,lower9,right9,left9,lwd,text="Chap 9: Markov Chains",cex=2)


upper10 <- lower9-1/7
lower10 <- upper10-1/7
right10 <- right8 -0.1
left10 <- left8-0.1
draw.box(upper10,lower10,right10,left10,lwd,text="Chap 10: State space \n models",cex=2)



upper11 <- upper10
lower11 <- lower10
right11 <- right9
left11 <- left9
draw.box(upper11,lower11,right11,left11,lwd,text="Chap 11: Hidden Markov\n Models",cex=2)

## 2-3
arrows((right3+left3)/2,lower2,(right3+left3)/2,upper3,lwd=lwd)

## 3-4
arrows((right3+left3)/2,lower3,(right3+left3)/2,upper4,lwd=lwd)

## 3-5
arrows(right3,(lower3+upper3)/2,(right5+3*left5)/4,(lower3+upper3)/2,lwd=lwd,code=0)
arrows((right5+3*left5)/4,(lower3+upper3)/2,(right5+3*left5)/4,upper5,lwd=lwd,code=2)
## 3-8
arrows(right3,(lower3+upper3)/2,left8,(lower3+upper3)/2,lwd=lwd,code=2)

## 4-6
arrows((5*right4+left4)/6,lower4,(5*right4+left4)/6,upper6,lwd=lwd,code=2)
## arrows((right4+left4)/2,(upper6+lower6)/2,left6,(upper6+lower6)/2,lwd=lwd,code=2)

## 5-6
arrows((right5+5*left5)/6,lower5,(right5+5*left5)/6,upper6,lwd=lwd,code=2)

## 2-7
arrows((right7+2*left7)/3,lower2,(right7+2*left7)/3,upper7,lwd=lwd)

## 8-10
arrows((right8+2*left8)/3,lower8,(right8+2*left8)/3,upper10,lwd=lwd)

## 2-9
arrows((right9+left9)/2,lower2,(right9+left9)/2,upper9,lwd=lwd)

## 9-11
arrows((right9+left9)/2,lower9,(right9+left9)/2,upper11,lwd=lwd)

## 8-9
arrows((6*right8+left8)/7,lower8,(6*right8+left8)/7,upper9,lwd=lwd,lty=2)

## 10-11
arrows(right10,(lower10+upper10)/2,left11,(lower10+upper10)/2,lwd=lwd,lty=2)


## 6-7
arrows(left6,(lower6+upper6)/2,right7,(lower6+upper6)/2,lwd=lwd,lty=2)

## 4-7
arrows((2*left4+right4)/3,lower4,(2*left4+right4)/3,upper7,lwd=lwd,lty=2)






