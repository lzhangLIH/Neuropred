rm(list=ls())
# input data
dat.orig <- read.csv('data_imp40_mod.csv')
dim(dat.orig)

############
# more than ou equal to 60 year old
## index of age >= 60 from original dataset (.imp = 0)
idx.60 <- which(dat.orig[dat.orig$.imp == 0, "age2"] >= 60)
head(idx.60)

dat.orig.60 <- NULL
for (i in c(0:40)) {
  dat. <- dat.orig[dat.orig$.imp == i,]
  dat. <- dat.[idx.60,]
  dat.orig.60 <- rbind(dat.orig.60,dat.)
}

summary(dat.orig.60[dat.orig.60$.imp == 1,"age2"])
head(dat.orig.60[dat.orig.60$.imp == 0,".id"])



# initial variables
variables <- read.csv('', header = F)
head(variables)
variables <- variables[,1]
head(variables)
length(variables)

dat.mod <- dat.orig[,c(".id",".imp","id","timeYear","event",variables)]

# log2 transformation
dat.log2 <- dat.mod
dat.log2[,variables] <- log2(dat.log2[,variables]+1)

# normalisation
min.max <- function(data) {
  # data: vector of numbers
  min. <- min(data,na.rm = T)
  max. <- max(data,na.rm = T)
  scale. <- (data - min.)/(max. - min.)
  return(scale.)
}

dat.01 <- dat.log2
for (i in c(0:40)) {
  for (v in variables) {
    scale. <- min.max(dat.01[dat.01$.imp == i,v])
    dat.01[dat.01$.imp == i,v] <- scale.
  }
}

library(mice)
mids.01 <- as.mids(dat.01)
summary(dat.01[dat.01$.imp == 1,"bmi2"])
summary(complete(mids.01)[,'bmi2'])

# collinearity
### collinearity

library(rms)
library(survival)


var. <- variables
dat <- dat.01
n.imp <- length(unique(dat$.imp))-1

vif.max. <- NULL

for (i in c(1:n.imp)) {
  dat. <- as.data.frame(scale(dat[dat$.imp==i, var.]))
  dat.$event <- dat[dat$.imp==i, "event"]
  dat.$timeYear <- dat[dat$.imp==i, "timeYear"]
  
  fit.cox. <-coxph(as.formula(paste("Surv(timeYear, event)", 
                                          paste(var., collapse = " + "), sep = " ~ ")), 
                         data = dat., ties = "breslow")
  max. <- sort(vif(fit.cox.), decreasing = T)[1]
  vif.max. <- c(vif.max., max.)
}

vif.max. # sysval2 having the highest vif (>10)
var. <- var.[var. != "sysval2"]

vif.max. # cholesterol2 vif > 10 when using variables without sysval2
var. <- var.[var. != "cholesterol2"]


vif.max. # cardiov2 vif > 10 when using variables without sysval2 and cholesterol2
sum(vif.max. > 10)
var. <- var.[var. != "cardiov2"]

vif.max. # liv.alone2 vif > 10 when using variables without sysval2, cholesterol2 and cardiov2
var. <- var.[var. != "liv.alone2"]
# 

write.table(var., file = 'var_woMulticoll10.csv',
            row.names = F, col.names = F)

###################################################################
# feature selection on traning datasets
# using 2 repeared 5 fold cross validation
library(c060)
library(glmnet)
library(caret)

set.seed(123)

# create 2 times 5 fold split dataset
idx.test.10fold = NULL

dat <- dat.log2

for (i in c(0:40)) {
  y. <- dat[dat$.imp == i,"event"]
  set.seed(123)
  idx.test.1 <- createFolds(y., k = 5, list = F)
  set.seed(1234)
  idx.test.2 <- createFolds(y., k = 5, list = F)
  idx.test.2 <- idx.test.2 + 5
  
  name. <- paste0("imp",i)
  idx.test.10fold <- cbind(idx.test.10fold,c(idx.test.1, idx.test.2))
}

class(idx.test.10fold)
idx.test.10fold <- as.data.frame(idx.test.10fold)
dim(idx.test.10fold)
colnames(idx.test.10fold) <- paste0('imp',seq(0,40))

dat <- read.csv('../dataset/datafinal.csv', stringsAsFactors = T)
idx.test.10fold$id <- c(dat[dat$age2 >= 60, 'idauniq'],
                        dat[dat$age2 >= 60, 'idauniq'])
head(idx.test.10fold)

write.csv(idx.test.10fold, file = "../results/imputation/m40i20/idx_test_2times_5fold_age60.csv", row.names = F)


## parameter selection (alpha for elastic net)

scale.row <- function(x, mean, sd) {
  scaled <- (x - mean)/sd
  return(scaled)
}

min.max.row <- function(data, min, max) {
  # data: vector of numbers
  # min. <- min(data)
  # max. <- max(data)
  scale. <- (data - min)/(max - min)
  return(scale.)
}

bounds <- t(data.frame(alpha=c(0, 1)))
colnames(bounds)<-c("lower","upper")

lst.var.sel.5fold. <- list()
lst.var.sel.4times. <- list()
lst.param.opt.5fold. <- list()

lst.cidx.elnet.5fold. <- list()

cidx.elnet.5fold. <- data.frame(train=numeric(),test=numeric())

dat. <- dat.log2
# dat.$id <- rep(dat[dat$age2 >= 60,'idauniq'],41)

for (i in c(1:40)) {
  name. <- paste0("imp",i)
  print(name.)
  
  dat.i. <- dat.[dat.$.imp == i,c(var., "event", "timeYear", "id")]
  test.idx. <- idx.test.10fold[,name.]
  
  params.opt. <- data.frame(alpha=numeric(),
                            lambda=numeric(),
                            error=numeric())
  
  lst.var.sel. <- list()
  
  cidx.elnet. <- data.frame(train=numeric(),test=numeric())
  
  for (f. in c(1:10)) {
    idx. <- idx.test.10fold[idx.test.10fold[,name.] == f.,"id"]
    # training dataset
    y.glmnet <- as.matrix(dat.i.[!(dat.i.$id %in% idx.),c("timeYear","event")])
    colnames(y.glmnet) <- c("time", "status")
    X.train <- as.matrix(dat.i.[!(dat.i.$id %in% idx.),var.])
    
    X.test <- as.matrix(dat.i.[dat.i.$id %in% idx.,var.])
    
    y.glmnet.test <- as.matrix(dat.i.[dat.i.$id %in% idx.,c("timeYear","event")])
    colnames(y.glmnet.test) <- c("time", "status")
    
    # normalisation (0,1)
    ## min and max value of taining data
    min. <- apply(X = X.train, MARGIN = 2, FUN = min)
    max. <- apply(X = X.train, MARGIN = 2, FUN = max)
    
    X.test <- t(apply(X = X.test, MARGIN = 1, FUN = min.max.row, 
                      min=min., max=max.))
    X.train <- t(apply(X = X.train, MARGIN = 1, FUN = min.max.row, 
                      min=min., max=max.))
    
    
    # alpha.opt = 0
    # lambda.opt = 0
    # error.opt = 0
    
    foldid <- balancedFolds(y.glmnet[,"status"], 5)
    
    # inner loop, 5 fold cross validation
    fit <- epsgo(Q.func = "tune.glmnet.interval", bounds = bounds,  
                 seed=123, verbose = F, nfolds=5, foldid=foldid,
                 x = X.train, y = y.glmnet,
                 type.min = "lambda.min", type.measure = "deviance", family = "cox")
    fit.summ. <- summary(fit, verbose = F)
    alpha. <- fit.summ.$opt.alpha
    lambda. <- fit.summ.$opt.lambda
    error. <- fit.summ.$opt.error
    
    # if (s == 1) {
    #   alpha.opt = alpha.
    #   lambda.opt = lambda.
    #   error.opt = error.
    # }
    # else {
    #   if (error. < error.opt) {
    #     error.opt = error.
    #     alpha.opt = alpha.
    #     lambda.opt = lambda.
    #   }
    # }
    
    
    params.opt. <- rbind(params.opt., data.frame(alpha=alpha.,
                                                 lambda=lambda.,
                                                 error=error.))
    
    fit.glmnet <- glmnet(X.train, y.glmnet, family = "cox", 
                         alpha = alpha.)
    
    coef.min <- coef(fit.glmnet, s = lambda.)
    index.min <- which(coef.min != 0)
    var.sel <- rownames(coef.min)[index.min]
    
    pred.train. <- predict(fit.glmnet, newx = X.train, s = lambda.)
    cidx.train. <- Cindex(pred = pred.train., y = y.glmnet)
    
    pred.test. <- predict(fit.glmnet, newx = X.test, s = lambda.)
    cidx.test. <- Cindex(pred = pred.test., y = y.glmnet.test)
    
    lst.var.sel.[[f.]] <- var.sel
    
    cidx.elnet. <- rbind(cidx.elnet.,data.frame(train=cidx.train.,test=cidx.test.))
  }
  
  lst.var.sel.5fold.[[name.]] <- lst.var.sel.
  lst.param.opt.5fold.[[name.]] <- params.opt.
  lst.cidx.elnet.5fold.[[name.]] <- cidx.elnet.
  cidx.elnet.5fold. <- rbind(cidx.elnet.5fold., colMeans(cidx.elnet.))
  
  
  # varibles slected 8 times (80%)
  num.var. <- NULL
  for (i in var.) {
    num.var.[i] <- 0
    for (f. in c(1:10)) {
      if (i %in% lst.var.sel.[[f.]]) {
        num.var.[i] <- num.var.[i] + 1
      }
    }
  }
  
  lst.var.sel.4times.[[name.]] <- names(num.var.)[num.var. >= 8]
  
}

# optimal parameters
df.params <- NULL
for (i in names(lst.param.opt.5fold.)) {
  params <- as.data.frame(lst.param.opt.5fold.[[i]])
  params$imp <- rep(i,nrow(params))
  df.params <- rbind(df.params,params)
}
dim(df.params)
write.csv(df.params, file = 'opt_params.csv', row.names = F)

# variables selected at least 8 times
variables.times <- data.frame(variable=character())
for (i in names(lst.var.sel.4times.)) {
  df.var <- as.data.frame(lst.var.sel.4times.[[i]])
  df.var[,2] <- rep(1,nrow(df.var))
  colnames(df.var) <- c('variable',i)
  variables.times <- merge(variables.times,df.var, by='variable', all = T)
}

dim(variables.times)
head(variables.times)

# selected variables
variables.times <- data.frame(variable=character())
for (i in names(lst.var.sel.5fold.)) {
  var.fold = data.frame(variable=character())
  for (j in c(1:10)) {
    df.var <- as.data.frame(lst.var.sel.5fold.[[i]][[j]])
    df.var[,2] <- rep(1,nrow(df.var))
    colnames(df.var) <- c('variable',paste0('fold',j))
    var.fold <- merge(var.fold,df.var,by='variable',all=T)
  }
  df.var.times <- as.data.frame(var.fold[,'variable'])
  df.var.times[,2] <- rowSums(var.fold[,colnames(var.fold) != 'variable'], na.rm = T)
  colnames(df.var.times) <- c('variable',i)
  variables.times <- merge(variables.times,df.var.times,by='variable',all=T)
}


lst.var.sel. <- lst.var.sel.4times.
num.var. <- NULL
for (i in var.) {
  num.var.[i] <- 0
  for (j in c(1:40)) {
    if (i %in% lst.var.sel.[[j]]) {
      num.var.[i] <- num.var.[i] + 1
    }
  }
}

##################################################################
# compare different number of variables

library(survival)
library(mice)
mids. <- as.mids(dat.01)

lst.var.sel.perc80. <- list()
for (i in c(32:40)) {
  times.sel. <- i
  num. <- sum(num.var. >= times.sel.)
  name. <- paste0("var.",num.)
  if (!(name. %in% names(lst.var.sel.perc80.))) {
    lst.var.sel.perc80.[[name.]] <- names(num.var.)[num.var. >= times.sel.]
  }
}

lst.coxph.pool <- list()

for (name. in names(lst.var.sel.perc80.)) {
  var.sel. <- lst.var.sel.perc80.[[name.]]
  coxph.var. <- with(data = mids., 
                     expr=coxph(as.formula(paste("Surv(timeYear, event)", 
                                                 paste(var.sel., collapse = " + "),
                                                 sep = " ~ ")), method="breslow"))
  lst.coxph.pool[[name.]] <- coxph.var.
  
}

# compute D1 pvalues, if length(lst.coxph.pool) > 1
len. <- length(lst.coxph.pool)
p.d1. <- as.data.frame(matrix(nrow = len., ncol = len.))
colnames(p.d1.) <- names(lst.coxph.pool)
rownames(p.d1.) <- names(lst.coxph.pool)

for (i. in c(1:(len.-1))) {
  for (j. in c((i.+1):len.)) {
    name.more. <- names(lst.coxph.pool)[i.]
    name.less. <- names(lst.coxph.pool)[j.]
    comp. = D1(lst.coxph.pool[[name.more.]], 
               lst.coxph.pool[[name.less.]])
    p. <- comp.$result[,4]
    p.d1.[name.more.,name.less.] <- p.
  }
}

dim(p.d1.)
p.d1.

write.csv(p.d1., file = "p.d1.csv")

write.table(as.data.frame(lst.var.sel.perc80.[["var.8"]]), 
          file = 'var_8.csv',
          row.names = F, col.names = F, sep = '\t', quote = F)

