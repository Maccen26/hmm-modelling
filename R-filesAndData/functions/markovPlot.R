par(mar=c(0,0,0,0),oma=c(0,0,0,0))
r <- 0.25
text.cex <- 1.5
ar.len <- 0.1
row <- c(4.5,3.5,2.5,1.5)
col<-c(1,2,3,4,5,6,7,8)
col<-seq(0.5,8,by=0.75)

plot(1,1,xlim=c(0,7),ylim=c(0.5,5),col="white",axes=FALSE,ylab="",xlab="")

for(i in 1:5){
    polygon(col[i]+r*c(-1,1,1,-1),row[1]+ r*c(1,1,-1,-1))
    arrows(x0=col[i]+r,y0=row[1],x1=col[i+1]-r,y1=row[1],length=ar.len)
}

text(col[1],row[1],expression(Y["t-2"]),cex=text.cex)
text(col[2],row[1],expression(Y["t-1"]),cex=text.cex)
text(col[3],row[1],expression(Y["t"]),cex=text.cex)
text(col[4],row[1],expression(Y["t+1"]),cex=text.cex)
text(col[5],row[1],expression(Y["t+2"]),cex=text.cex)
arrows(x0=0,y0=row[1],x1=col[1]-r,y1=row[1],length=ar.len)




## Row 2

row <- c(4.5,3.5,2.5,1.5)
row[2] <-3
for(i in 1:5){
    polygon(col[i]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
    arrows(x0=col[i]+r,y0=row[2],x1=col[i+1]-r,y1=row[2],length=ar.len)
}


text(col[1],row[2],expression(Y["t-2"]),cex=text.cex)
text(col[2],row[2],expression(Y["t-1"]),cex=text.cex)
text(col[3],row[2],expression(Y["t"]),cex=text.cex)
text(col[4],row[2],expression(Y["t+1"]),cex=text.cex)
text(col[5],row[2],expression(Y["t+2"]),cex=text.cex)
arrows(x0=0,y0=row[2],x1=col[1]-r,y1=row[2],length=ar.len)

lines(c(0,col[1]-r),row[2]+3/2*r*c(1,1))
arrows(x0=col[1]-r,y0=row[2]+3/2*r,x1=col[1]-r/2,y1=row[2]+r,length=ar.len)

lines(c(col[1]+r/2,col[1]+r),c(row[2]+r,row[2]+3/2*r))
lines(c(col[1]+r,col[3]-r),row[2]+3/2*r*c(1,1))
arrows(x0=col[3]-r,y0=row[2]+3/2*r,x1=col[3]-r/2,y1=row[2]+r,length=ar.len)

lines(c(col[3]+r/2,col[3]+r),c(row[2]+r,row[2]+3/2*r))
lines(c(col[3]+r,col[5]-r),row[2]+3/2*r*c(1,1))

arrows(x0=col[5]-r,y0=row[2]+3/2*r,x1=col[5]-r/2,y1=row[2]+r,length=ar.len)

lines(c(col[5]+r/2,col[5]+r),c(row[2]+r,row[2]+3/2*r))
arrows(x0=col[5]+r,y0=row[2]+3/2*r,x1=col[6]-r,y1=row[2]++3/2*r,length=ar.len)


lines(c(0,col[2]-r),row[2]-3/2*r*c(1,1))
arrows(x0=col[2]-r,y0=row[2]-3/2*r,x1=col[2]-r/2,y1=row[2]-r,length=ar.len)

lines(c(col[2]+r/2,col[2]+r),c(row[2]-r,row[2]-3/2*r))
lines(c(col[2]+r,col[4]-r),row[2]-3/2*r*c(1,1))
arrows(x0=col[4]-r,y0=row[2]-3/2*r,x1=col[4]-r/2,y1=row[2]-r,length=ar.len)

lines(c(col[4]+r/2,col[4]+r),c(row[2]-r,row[2]-3/2*r))
lines(c(col[4]+r,col[6]-r),row[2]-3/2*r*c(1,1))
arrows(x0=col[4]+r,x1=col[6]-r,y0=row[2]-3/2*r,y1=row[2]-3/2*r,length=ar.len)





## Row 3
row <- c(4.5,3.5,2.5,1.5)
row[3] <- 1.5

for(i in 1:5){
    polygon(col[i]+r*c(-1,1,1,-1),row[3]+ r*c(1,1,-1,-1))
    arrows(x0=col[i]+r,y0=row[3],x1=col[i+1]-r,y1=row[3],length=ar.len)
}

text(col[1],row[3],expression(Y["t-2"]^"*"),cex=text.cex)
text(col[2],row[3],expression(Y["t-1"]^"*"),cex=text.cex)
text(col[3],row[3],expression(Y["t"]^"*"),cex=text.cex)
text(col[4],row[3],expression(Y["t+1"]^"*"),cex=text.cex)
text(col[5],row[3],expression(Y["t+2"]^"*"),cex=text.cex)
arrows(x0=0,y0=row[3],x1=col[1]-r,y1=row[3],length=ar.len)






##################################################
##
##################################################
row <- c(4.5,3.5,2.5,1.5)


polygon(col[7]+r*c(-1,1,1,-1),row[1]+ r*c(1,1,-1,-1))
text(col[7],row[1],expression(0),cex=text.cex)

polygon(col[9]+r*c(-1,1,1,-1),row[1]+ r*c(1,1,-1,-1))
text(col[9],row[1],expression(1),cex=text.cex)

arrows(x0=col[7]+r,y0=row[1]+r/2,x1=col[9]-r,y1=row[1]+r/2,length=ar.len)
arrows(x1=col[7]+r,y1=row[1]-r/2,x0=col[9]-r,y0=row[1]-r/2,length=ar.len)


theta <- seq(-pi,0.5*pi,length=100)
lines(sin(theta)*r*0.5+col[7]-r,cos(theta)*r*0.5+row[1]+r)
arrows(x0=col[7]-r-0.001,y0=row[1]+0.5*r,x1=col[7]-r,y1=row[1]+0.5*r,length=ar.len)


theta <- seq(-0.5*pi,1*pi,length=100)
lines(sin(theta)*r*0.5+col[9]+r,cos(theta)*r*0.5+row[1]+r)
arrows(x0=col[9]+r+0.001,y0=row[1]+0.5*r,x1=col[9]+r,y1=row[1]+0.5*r,length=ar.len)



##################################################
##
##################################################


polygon(col[7]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
text(col[7],row[2],expression("00"),cex=text.cex)


theta <- seq(-pi,0.5*pi,length=100)
lines(sin(theta)*r*0.5+col[7]-r,cos(theta)*r*0.5+row[2]+r)
arrows(x0=col[7]-r-0.001,y0=row[2]+0.5*r,x1=col[7]-r,y1=row[2]+0.5*r,length=ar.len)


polygon(col[9]+r*c(-1,1,1,-1),row[2]+ r*c(1,1,-1,-1))
text(col[9],row[2],expression("01"),cex=text.cex)

arrows(x0=col[7]+r,y0=row[2],x1=col[9]-r,y1=row[2],length=ar.len)


polygon(col[7]+r*c(-1,1,1,-1),row[3]+r+ r*c(1,1,-1,-1))
text(col[7],row[3]+r,expression("10"),cex=text.cex)

arrows(x0=col[7],y0=row[3]+2*r,x1=col[7],y1=row[2]-r,length=ar.len)
arrows(x0=col[7]+r,y0=row[3]+1.5*r,x1=col[9]-r,y1=row[2]-0.5*r,length=ar.len)
arrows(x1=col[7]+r,y1=row[3]+r,x0=col[9]-0.5*r,y0=row[2]-r,length=ar.len)


polygon(col[9]+r*c(-1,1,1,-1),row[3]+r+ r*c(1,1,-1,-1))
text(col[9],row[3]+r,expression("11"),cex=text.cex)

arrows(x1=col[9],y1=row[3]+2*r,x0=col[9],y0=row[2]-r,length=ar.len)
arrows(x0=col[9]-r,y0=row[3]+r,x1=col[7]+r,y1=row[3]+r,length=ar.len)


theta <- seq(0,1.5*pi,length=100)
lines(sin(theta)*r*0.5+col[9]+r,cos(theta)*r*0.5+row[3])
arrows(x0=col[9]+0.5*r,y0=row[3]-0.001,x1=col[9]+0.5*r,y1=row[3],length=ar.len)


##################################################
##
##################################################


polygon(col[7]+r*c(-1,1,1,-1),row[4]+2*r+ r*c(1,1,-1,-1))
text(col[7],row[4]+2*r,expression(1),cex=text.cex)
 
theta <- seq(-pi,0.5*pi,length=100)
lines(sin(theta)*r*0.5+col[7]-r,cos(theta)*r*0.5+row[4]+3*r)
arrows(x0=col[7]-r-0.001,y0=row[4]+2.5*r,x1=col[7]-r,y1=row[4]+2.5*r,length=ar.len)
 
 
polygon(col[9]+r*c(-1,1,1,-1),row[4]+2*r+ r*c(1,1,-1,-1))
text(col[9],row[4]+2*r,expression("2"),cex=text.cex)
 
arrows(x0=col[7]+r,y0=row[4]+2*r,x1=col[9]-r,y1=row[4]+2*r,length=ar.len)
 

polygon(col[7]+r*c(-1,1,1,-1),row[4]-2*r+ r*c(1,1,-1,-1))
text(col[7],row[4]-2*r,expression("3"),cex=text.cex)

arrows(x0=col[7],y0=row[4]-r,x1=col[7],y1=row[4]+r,length=ar.len)
arrows(x0=col[7]+r,y0=row[4]-1.5*r,x1=col[9]-r,y1=row[4]+1.5*r,length=ar.len)
arrows(x1=col[7]+r,y1=row[4]-2*r,x0=col[9]-0.5*r,y0=row[4]+r,length=ar.len)
 
polygon(col[9]+r*c(-1,1,1,-1),row[4]-2*r+ r*c(1,1,-1,-1))
 text(col[9],row[4]-2*r,expression("4"),cex=text.cex)
 
arrows(x0=col[9],y0=row[4]+r,x1=col[9],y1=row[4]-r,length=ar.len)
arrows(x0=col[9]-r,y0=row[4]-2*r,x1=col[7]+r,y1=row[4]-2*r,length=ar.len)
  
 theta <- seq(0,1.5*pi,length=100)
 lines(sin(theta)*r*0.5+col[9]+r,cos(theta)*r*0.5+row[4]-3*r)
 arrows(x0=col[9]+0.5*r,y0=row[4]-3*r-0.001,x1=col[9]+0.5*r,y1=row[4]-3*r,length=ar.len)

