
#### Setting up working Directory
setwd("F:/r")

library(caret) # for dummyVars
library(RCurl) # download https data
library(Metrics) # calculate errors
library(xgboost)



#  Importing both Train and Test Dataset
ortrain2016<-read.csv("train2016.csv",na.strings = c("NA",""))
ortest2016<-read.csv("test2016.csv",na.strings = c("NA",""))


# train2016$USER_ID<-NULL
# test2016$USER_ID<-NULL


## Using random miss to impute NA values

# For train dataset
miceMod <- missForest(train2016)

trainoutput <-miceMod$ximp  # generate the completed data.
write.csv(trainoutput,"missforesttrainnew.csv")


# For test dataset

missf2<-missForest(test2016)
testoutput <- missf2$ximp
write.csv(testoutput,"missforesttestnew.csv")





### Reimporting the na's imputed data

ntrain2016<-read.csv("missforesttrainnew.csv")
ntest2016<-read.csv("missforesttestnew.csv")

names(ntrain2016)

ntrain2016<-ntrain2016[,-1]
ntest2016<-ntest2016[,-1]

names(ntrain2016)
table(ntrain2016$Party)   

ntrain2016$Party<-factor(ifelse(ntrain2016$Party=="Democrat",1,0),levels = c(0,1))      



#### Dummy variable creation
names(ntrain2016)

dmy<-dummyVars("~.",data=ntrain2016[,-7])
ntraindmy<-data.frame(predict(dmy,newdata=ntrain2016[,-7]))    

names(ntraindmy)

dim(ntrain2016)

ntraindmy$Party<-ntrain2016$Party


names(ntraindmy) 

class(ntraindmy$Party)
sample(5375,(5375/30))

dim(ntraindmy)


##Implementing xgboost model
ntraindmy=ntraindmy[sample(5375,(5375/10)),]
cv <- 30
trainSet <-ntraindmy
names(trainSet)
cvDivider <- floor(nrow(trainSet) / (cv+2))
cvDivider
ch<-data.frame(depth=numeric(),rounds=numeric(),accuracy=numeric())


for (depth in seq(1,10,1)) { 
        for (rounds in seq(1,20,1)) {
                
                indexCount <- 1
                for (cv in seq(1:cv)) {
                        # assign chunk to data test
                        dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
                        dataTest <- trainSet[dataTestIndex,]
                        # everything else to train
                        dataTrain <- trainSet[-dataTestIndex,]
                        
                        model<-xgboost(data=data.matrix(dataTrain[,-225]),label=data.matrix(dataTrain[,225]),
                                       max.depth=depth, nround=rounds,
                                       objective="binary:logistic"
                                       ,verbose=0)   
                        
                        pp<-predict(model,data.matrix(dataTest[,-225]))
                        
                        o1<-as.matrix(table(pp>0.5,dataTest$Party))
                        accuracy<-(o1[1,1]+o1[2,2])/(sum(o1))   
                        
                        ch<-rbind(ch,c(depth,rounds,accuracy,cv))
                }
                
        }               
        
}

ch[1:20,]
library(dplyr)
names(ch)
desc<-arrange(ch,desc(X0.470588235294118))       
desc[1:5,]
which.max(ch$X0.548571428571429)
ch[1524,]

dim(ch)/30    


dataTestIndex <- c((3 * cvDivider):(3 * cvDivider + cvDivider))

dataTest <- trainSet[dataTestIndex,]

# everything else to train
dataTrain <- trainSet[-dataTestIndex,]

ntraindmy=dataTrain
names(ntraindmy)

##### creating model         
model<-xgboost(data=data.matrix(ntraindmy[,-225]),label=data.matrix(ntraindmy[,225]),
               max_depth = 3, 
               nround=11, 
               objective="binary:logistic"
)               

pp<-predict(model,data.matrix(dataTest[,-225]))

o1<-as.matrix(table(pp>0.5,dataTest$Party))
o1
accuracy<-(o1[1,1]+o1[2,2])/(sum(o1))   




## Applying on actual test set


##Changing it to dummy var and matrix format
newtest<-ntest2016
names(newtest)
dim(newtest)


dmytest<-dummyVars("~.",data=newtest)
ntestdmy<-data.frame(predict(dmytest,newdata=newtest))    
ntestdmy

pg<-predict(model,data.matrix(ntestdmy))  


#submission

rt<-ifelse(pg>0.5,"Democrat","Republican")

xgboostnew<-as.data.frame(cbind(ortest2016$USER_ID,rt))

colnames(xgboostnew)<-c("USER_ID","Predictions")

write.csv(xgboostnew,"xgboostlat.csv")









