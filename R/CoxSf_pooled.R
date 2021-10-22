rm(list=ls())

library(mice)
# input data
dat.orig <- read.csv('data_imp40_mod.csv')
dim(dat.orig)

# input selected variables
variable <- read.csv('var_8.csv', header = F)
variable <- variable[,1]
head(variable)
length(variable)

# log2 transformation
dat.log2 <- dat.orig[dat.orig$age2 >= 60,]
dim(dat.log2)
dat.log2[,variable] <- log2(dat.log2[,variable]+1)

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
  for (v in variable) {
    scale. <- min.max(dat.01[dat.01$.imp == i,v])
    dat.01[dat.01$.imp == i,v] <- scale.
  }
}

mids.01 <- as.mids(dat.01[,colnames(dat.01) != 'id'])
summary(dat.01[dat.01$.imp == 1,"gaitspeed2"])
summary(complete(mids.01)[,'gaitspeed2'])


# pooled Cox regression model using the whole dataset
coxph. <- with(data = mids.01, 
               expr=coxph(as.formula(paste("Surv(timeYear, event)", 
                                           paste(variable, collapse = " + "),
                                           sep = " ~ ")), method="breslow"))
pool.coxph. <- pool(coxph.)
summary(pool.coxph., conf.int = TRUE, exponentiate = TRUE)

# forest plot
library(ggfortify)
dat.plot. <- as.data.frame(summary(pool.coxph., conf.int = TRUE, exponentiate = TRUE))
dat.plot. <- dat.plot.[,c("term","estimate","2.5 %","97.5 %")]
colnames(dat.plot.) <- c("variable","HR","Lower.CI","Upper.CI")
rownames(dat.plot.) <- c('Age','BMI', 'current smoker','Executive Cognition Index',
                        'Gait speed', 'Memory Function Index', 
                        'Difficulty picking up coin',
                         'Poor hearing','Diabetic eye','Weight loss','Vigorous activities')


dat.plot.$indx <- seq(1,nrow(dat.plot.),1)

p <- ggplot(data=dat.plot., aes(y=indx, x=HR, xmin=Lower.CI, xmax=Upper.CI))+ 
  
  #the effect sizes to the plot
  geom_point()+
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank(),
        axis.text.x = element_text(color = "black"),
        axis.text.y = element_text(color = "black"))+
  
  #adds the CIs
  geom_errorbarh(height=.1)+
  
  #sets the scales
  scale_x_continuous(trans='log10')+
  scale_y_continuous(name = "", breaks=1:nrow(dat.plot.), labels = rownames(dat.plot.), trans="reverse")+
  
  #adding a vertical line at the effect = 0 mark
  geom_vline(xintercept=1, color="black", linetype="dashed", alpha=.5)

p

# evaluation on test datasets from 70 imputed datasets: Python

