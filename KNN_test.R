## Testing the K-Nearest-Neighbors algorithm ##

#Created by A.Sosa-Costa 25/06/2017

#This script tests the performance of the KNN algorithm.
#-First it creates two random samples from two different multivariate Normal distributions,
# whose parameters (mean and covariance matrix) can be individually adjusted
#-Then the data randomly generated is divided in two subgroups, which will be
# used later as the training subset and the test subset respectively
#-The KNN algorithm is applied on the test data subset two classify it as Group A or Group B
#-Finally the decision boundary obtained from the KNN algorithm is plotted together with
# Bayes decision boundary and the classified points (using KNN)
#-The output TT indicates the classifications successfully assigned to groups A and B.

rm(list=ls())
library(mvtnorm)  #random number generator for the multivariate normal distribution
library(class)    #to use the knn function

#Declare parameters to be used
seed1=40        #Seed value for random number generator
N1=100          #Number of random points to be generated for GroupA
N2=100          #Number of random points to be generated for GroupB
mu1=c(0.5,0.5)  #Mean values of the multivariate distribution (GroupA)
mu2=c(-1,-1)    #Mean values of the multivariate distribution (GroupB)
var1=matrix(c(1, 0.5, 0.5, 1), nrow = 2)   #covariance matrix (GroupA)
var2=matrix(c(0.5, 0, 0, 0.5), nrow = 2)   #covariance matrix (GroupA)
a=65  #define the data subset used as training data a*(N1+N2)/100
k1=9  #number of neighbours considered

#Creates two distinguishable random set of points in 2D (Training data set)
set.seed(seed1)
groupA<-rmvnorm(N1, mean =mu1 , sigma = var1)
groupB<-rmvnorm(N2, mean =mu2 , sigma = var2)
# Plot both groups of points to check that everything is correct
# plot(groupA,type='p',col='blue',xlim=c(min(groupB[,1]),max(groupA[,1])),ylim=c(min(groupB[,2]),max(groupA[,2])))
# points(groupB,type='p',col='red')


# Divide the data in a training subset and a test subset (to a ratio of a:b=65:35)
a1=round(a*N1/100)
a2=round(a*N2/100)
train_data<-rbind(groupA[1:a1,],groupB[1:a2,]) 
test_data<-rbind(groupA[(a1+1):N1,],groupB[(a2+1):N2,]) 

#Create a label vector indicating which data subset corresponds to each subgroup
train_label<-factor(c(rep('GroupA',a1),rep('GroupB',a2)))
test_label<-factor(c(rep('GroupA',N1-a1),rep('GroupB',N2-a2)))


##Implement kNN algorithm
knn_out<- knn(train = train_data, test = test_data,cl = train_label, k=k1,prob=TRUE)
#attributes(.Last.value)
pp<-attr(knn_out,'prob')

#Calculate the proportion of correct classification
100*sum(test_label==knn_out)/length(pp)

#Success rate
TT<-table(knn_out,test_label)


## KNN-based decision boundary ##

#First I create a grid with xy pairs within the range of data (training+test in both groups)
datat<-rbind(groupA,groupB)
xl<-range(datat[,1])
yl<-range(datat[,2])
px<-seq(from=xl[1],to=xl[2],by=0.1)
py<-seq(from=yl[1],to=yl[2],by=0.1)
xydata<-expand.grid(x=px,y=py) 

#Then create the countor map
knn_map<- knn(train = train_data, test = xydata,cl = train_label, k=k1,prob=TRUE)
prob <- attr(knn_map, "prob")
prob <- ifelse(knn_map=="GroupA", prob, 1-prob)
prob_bound <- matrix(prob, nrow = length(px), ncol = length(py))

#plot decision boundary using KNN algorithm
# contour(px, py, prob_bound,level=0.5,labels="", xlab="", ylab="", main=
#           "10-nearest neighbour", axes=TRUE)
# points(train_data,col=ifelse(train_label=='GroupA','blue','red'))
# points(test_data,col=ifelse(test_label=='GroupA','cornflowerblue','coral'))
# points(xydata, pch=".", cex=1.2, col=ifelse(prob_bound<0.5, "coral", "cornflowerblue"))
# box()


## Bayes decision boundary (BDB) ##
#(Since we know the distributions from where the random points were simulated
#we can caclculate with precision what is the probability of a given xy pair
#to belong either to GroupA or GroupB)

prob1<-dmvnorm(xydata, mean = mu1, sigma = var1, log=FALSE) #multivariate normal distribution
prob2<-dmvnorm(xydata, mean = mu2, sigma = var2, log=FALSE) #multivariate normal distribution
prob12 <- matrix(prob2-prob1, nrow = length(px), ncol = length(py))

## Master Final Plot ##
contour(px, py, prob12,level=0,labels="", xlab="X", ylab="Y") #plot Bayes decision boundary
contour(px, py, prob_bound,level=0.5,labels="", xlab="", ylab="", main=
          "10-nearest neighbour", axes=TRUE, add=TRUE,lty="dotted")  #plot KNN decision boundary

points(train_data,col=ifelse(train_label=='GroupA','blue','red'))    #plot training data points
points(test_data,col=ifelse(test_label=='GroupA','blue','red'),pch=10)#plot test data points
points(xydata, pch=".", cex=1.2, col=ifelse(prob12>0, "coral", "cornflowerblue")) #GroupA and GroupB spaces,

par(xpd=TRUE)                                                                                  #defined by BDB
legend(1.5,6, legend=c("Group A (training)", "Group B (training)","GroupA (test)","Group B (test)" ),
       col=c("blue", "red","blue", "red"), pch=c(1,1,10,10), cex=0.8,bty="n",y.intersp=0.25)
legend(-2,6, legend=c("Bayes Decision Boundary","KNN Decision Boundary" ),
       lty=c("solid","dotted"), cex=0.8,bty="n",y.intersp=0.25)
box()

