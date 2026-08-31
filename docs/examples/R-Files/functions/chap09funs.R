
stat.dist <- function(Gamma){
    m <- dim(Gamma)[1]
    E <- matrix(1, ncol = m, nrow = m)
    rep(1, m) %*% solve(diag(m) - Gamma + E)
}    

Markov.link <- function(Gamma){
    m <- dim(Gamma)[1]
    diag(Gamma) <- 0
    beta <-t( log(Gamma/(1-rowSums(Gamma))))
    diag(beta) <- NA
    beta[!is.na(beta)]   
}

Markov.Inv.Link <- function(pars,m){
    exp.pars <- exp(pars)
    Gamma <- diag(m)
    Gamma[Gamma != 1] <- exp.pars
    t(Gamma) / (colSums(Gamma))
}



nll1.markov <- function(pars,y,m){
    F <- trans.count(y, m)
    Gamma <- Markov.Inv.Link(pars, m)
    delta <- stat.dist(Gamma)
    ll <- log(delta[y[1]]) + sum(F * log(Gamma))
    - ll
}

Markov.Inv.Link.Reg <- function(pars, m, X){
    p <- dim(X)[2]; n <- dim(X)[1]
    beta <- matrix(pars[1 : (p * m)], ncol = m)
    exp.eta <- exp(X %*% beta)
    exp.tau <- exp(pars[-(1: (p * m))])
    Tau.mat <- diag(m)
    Tau.mat[Tau.mat != 1] <- exp.tau
    Gamma <- array(dim=c(n, m, m))
    for(i in 1:n){
        diag(Tau.mat) <- exp.eta[i, ]
        Gamma[i, , ] <- t(Tau.mat) / colSums(Tau.mat)
    }
    Gamma
}

nll.markov.reg <- function(pars, y, X, mm){
    n <- length(y)
    Gamma <- Markov.Inv.Link.Reg(pars, mm, X)
    lGamma <- log(Gamma)
    ll <- 0
    for(i in 2:n){
        ll <- ll + lGamma[i - 1, y[i - 1], y[i]]
    }
    -ll
}

sim.markovReg <- function(Gamma, y1){
    T <- dim(Gamma)[1]
    m <- dim(Gamma)[2]; y <- numeric(T)
    y[1] <- y1
    for(i in 2:T){
        y[i] <- sample(1:m, size = 1, prob = Gamma[i,y[i-1], ])
    }
    y
}
