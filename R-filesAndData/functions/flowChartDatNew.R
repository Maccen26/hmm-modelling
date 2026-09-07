par(mfrow=c(1,1),mar=c(0,0,0,0),oma=c(0,0,2,0))

lwd<-2
cex<-1
cex.arrow <- 0.125
plot(c(0,1),c(0,1),col="white",axes=FALSE,xlab="",ylab="")


draw.box <- function(upper,lower,right,left,lwd,text="",cex=1){
    lines(c(left,right),c(upper,upper),lwd=lwd)
    lines(c(left,right),c(lower,lower),lwd=lwd)
    lines(c(left,left),c(upper,lower),lwd=lwd)
    lines(c(right,right),c(upper,lower),lwd=lwd)
    text((left+right)/2, (lower+upper)/2,labels=text,cex=cex)
}


nrow <- 5
d <- 2*nrow-1
w1 <- 0.25
w2 <- 0.15

plot(c(0,1),c(0,1),col="white",axes=FALSE,xlab="",ylab="")

upper2 <- 1
lower2 <- 1-1/d
right2 <- 1
left2 <- 0
draw.box(upper2,lower2,right2,left2,lwd,text="Can you assume Gaussian data (possibly after transformation)?",cex=cex)


upper3 <- lower2-1/d
lower3 <- upper3-1/d
right3 <- w1
left3 <- 0
draw.box(upper3,lower3,right3,left3,lwd,text="Independent observations?",cex=cex)


upper3d <- lower2-1/d
lower3d <- upper3-1/d
left3d <- right3+0.05
right3d <- left3d+w2
draw.box(upper3d,lower3d,right3d,left3d,lwd,text="Chap 3: General \n Linear Models",cex=cex)


upper4 <- 1-4/d
lower4 <- 1-5/d
right4 <- w1
left4 <- 0
draw.box(upper4,lower4,right4,left4,lwd,text="Repeated Measurements?",cex=cex)


upper4d <- 1-4/d
lower4d <- 1-5/d
left4d <- right4+0.05
right4d <- left4d+w2
draw.box(upper4d,lower4d,right4d,left4d,lwd,text="Chap 5: Linear Mixed \n Effect Models",cex=cex)


upper5 <- 1-6/d
lower5 <- 1-7/d
right5 <- w1
left5 <- 0
draw.box(upper5,lower5,right5,left5,lwd,text="Directly observed Time-series?",cex=cex)


upper5d <- 1-6/d
lower5d <- 1-7/d
left5d <- right5+0.05
right5d <- left5d+w2
draw.box(upper5d,lower5d,right5d,left5d,lwd,text="Chap 8: Linear Time\n Series Models",cex=cex)


upper6 <- 1-8/d
lower6 <- 1-9/d
right6 <- w1
left6 <- 0
draw.box(upper6,lower6,right6,left6,lwd,text="Indirectly observed Time series?",cex=cex)

upper6d <- 1-8/d
lower6d <- 1-9/d
left6d <- right6+0.05
right6d <- left6d+w2
draw.box(upper6d,lower6d,right6d,left6d,lwd,text="Chap 10: State\n  space models",cex=cex)



upper7 <- lower2-1/d
lower7 <- upper3-1/d
left7 <- 0.55
right7 <- left7+w1
draw.box(upper7,lower7,right7,left7,lwd,text="Independent observations?",cex=cex)


upper7d <- lower2-1/d
lower7d <- upper3-1/d
left7d <- right7+0.05
right7d <- left7d+w2
draw.box(upper7d,lower7d,right7d,left7d,lwd,text="Chap 4/7: Generalized\n Linear or Survival\n Models",cex=cex)


upper8 <- 1-4/d
lower8 <- 1-5/d
left8 <- 0.55
right8 <- left8+w1
draw.box(upper8,lower8,right8,left8,lwd,text="Repeated Measurements?",cex=cex)


upper8d <- 1-4/d
lower8d <- 1-5/d
left8d <- right8+0.05
right8d <- left8d+w2
draw.box(upper8d,lower8d,right8d,left8d,lwd,text="Chap 6: Hierarchical\n Models ",cex=cex)


upper9 <- 1-6/d
lower9 <- 1-7/d
left9 <- 0.55
right9 <- left7+w1
draw.box(upper9,lower9,right9,left9,lwd,text="Directly observed Time-series?",cex=cex)


upper9d <- 1-6/d
lower9d <- 1-7/d
left9d <- right9+0.05
right9d <- left9d+w2
draw.box(upper9d,lower9d,right9d,left9d,lwd,text="Chap 9: Markov\n Models",cex=cex)


upper10 <- 1-8/d
lower10 <- 1-9/d
left10 <- 0.55
right10 <- left7+w1
draw.box(upper10,lower10,right10,left10,lwd,text="Indirectly Observed Time-series?",cex=cex)

upper10d <- 1-8/d
lower10d <- 1-9/d
left10d <- right10+0.05
right10d <- left10d+w2
draw.box(upper10d,lower10d,right10d,left10d,lwd,text="Chap 11: Hidden\n Markov Models",cex=cex)

lines(c(0.5,0.5),c(0,1-2/d),lwd=3,lty=2)



## 2-3
arrows((right3+left3)/2,lower2,(right3+left3)/2,upper3,lwd=lwd,length=cex.arrow)
text((right3+left3)/2,(lower2+upper3)/2,labels="Y",pos=4,cex=cex)

arrows((right3+left3)/2,lower3,(right3+left3)/2,upper4,lwd=lwd,length=cex.arrow)
text((right3+left3)/2,(lower3+upper4)/2,labels="N",pos=4,cex=cex)

arrows((right3+left3)/2,lower4,(right3+left3)/2,upper5,lwd=lwd,length=cex.arrow)
text((right3+left3)/2,(lower4+upper5)/2,labels="N",pos=4,cex=cex)

arrows((right3+left3)/2,lower5,(right3+left3)/2,upper6,lwd=lwd,length=cex.arrow)
text((right3+left3)/2,(lower5+upper6)/2,labels="N",pos=4,cex=cex)


arrows((right7+left7)/2,lower2,(right7+left7)/2,upper7,lwd=lwd,length=cex.arrow)
text((right7+left7)/2,(lower2+upper7)/2,labels="N",pos=4,cex=cex)

arrows((right7+left7)/2,lower7,(right7+left7)/2,upper8,lwd=lwd,length=cex.arrow)
text((right7+left7)/2,(lower7+upper8)/2,labels="N",pos=4,cex=cex)

arrows((right7+left7)/2,lower8,(right7+left7)/2,upper9,lwd=lwd,length=cex.arrow)
text((right7+left7)/2,(lower8+upper9)/2,labels="N",pos=4,cex=cex)

arrows((right7+left7)/2,lower9,(right7+left7)/2,upper10,lwd=lwd,length=cex.arrow)
text((right7+left7)/2,(lower9+upper10)/2,labels="N",pos=4,cex=cex)


arrows(right3,(lower3+upper3)/2,left3d,(lower3+upper3)/2,lwd=lwd,length=cex.arrow)
text((right3+left3d)/2,(lower3+upper3)/2,labels="Y",pos=3,cex=cex)

arrows(right3,(lower4+upper4)/2,left3d,(lower4+upper4)/2,lwd=lwd,length=cex.arrow)
text((right3+left3d)/2,(lower4+upper4)/2,labels="Y",pos=3,cex=cex)


arrows(right3,(lower5+upper5)/2,left3d,(lower5+upper5)/2,lwd=lwd,length=cex.arrow)
text((right3+left3d)/2,(lower5+upper5)/2,labels="Y",pos=3,cex=cex)

arrows(right3,(lower6+upper6)/2,left3d,(lower6+upper6)/2,lwd=lwd,length=cex.arrow)
text((right3+left3d)/2,(lower6+upper6)/2,labels="Y",pos=3,cex=cex)


arrows(right7,(lower3+upper3)/2,left7d,(lower3+upper3)/2,lwd=lwd,length=cex.arrow)
text((right7+left7d)/2,(lower3+upper3)/2,labels="Y",pos=3,cex=cex)

arrows(right7,(lower4+upper4)/2,left7d,(lower4+upper4)/2,lwd=lwd,length=cex.arrow)
text((right7+left7d)/2,(lower4+upper4)/2,labels="Y",pos=3,cex=cex)

arrows(right7,(lower5+upper5)/2,left7d,(lower5+upper5)/2,lwd=lwd,length=cex.arrow)
text((right7+left7d)/2,(lower5+upper5)/2,labels="Y",pos=3,cex=cex)

arrows(right7,(lower6+upper6)/2,left7d,(lower6+upper6)/2,lwd=lwd,length=cex.arrow)
text((right7+left7d)/2,(lower6+upper6)/2,labels="Y",pos=3,cex=cex)


