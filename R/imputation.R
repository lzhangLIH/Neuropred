# original data
dat <- read.csv('', stringsAsFactors = T)

# initial variables selected from literature, csv files with a list of variables
variable <- read.csv('')
dim(variable)
variable <- variable[,1]


# imputation

## data for imputation
### mortstatus, mif2 and execindex2 will be used for imputation model
dat. <- dat[,c(variable,"mortstatus", "mfi2", "execindex2", "timeYear","event")]
dim(dat.)

## sort the variables by the increasing number of missing values
miss.perc <- colSums(is.na(dat.))/nrow(dat.)
max(miss.perc) # 38.5%, so I will use 40 imputation
dat. <- dat.[,names(sort(miss.perc))]

library(mice)

fluxplot(data = dat.)
fx <- flux(data = dat.)
sort(fx$outflux) ### I exclude the variables with outflux < 0.3 (it depends on the variables) when they don't have missing values
summary(dat.[,colnames(dat.)[fx$outflux < 0.3]]) # they all have missing values

### dry run imputation
ini <- mice(dat., maxit = 0, printFlag = F)
##constant and collinear varibles
out.logged <- as.character(ini$loggedEvents[, "out"])
length(out.logged) # 0

### prediction matrix
## the variables should be included in imputation models
inlist <- c("age2","sex2","event", "timeYear","education2","mortstatus",
            "anxiety2","medic.take2", "incont2", "mfi2", "gaitspeed2", 
            "execindex2", "pa2", "lowincome2", "hearing2")
length(inlist)
sum(inlist %in% colnames(dat.))

pred <- quickpred(dat., include = inlist, mincor = 0.3)
table(rowSums(pred)) ## I modified the mincor according to the number of variables included in the models

### imputation
imp.m40.i20 <- mice(dat., pred = pred, seed = 123, m = 40, maxit = 20, remove.collinear = FALSE)

long.incl <- complete(imp.m40.i20,action = 'long', include = T)

# output the imputed dataset
write.csv(long.incl, file = 'data_imp40_mod.csv')
