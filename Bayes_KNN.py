# -*- coding: utf-8 -*-
# This part exploring the KNN and simple Bayes technique

# This is the area of import package
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as multn
import mnist_reader
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# We first use the mnist_reader to store into the numpy arrays
trainData, trainLabels = mnist_reader.load_mnist('data', kind='train')
testData, testLabels = mnist_reader.load_mnist('data', kind='t10k')

# This function trains the data set with bayes rule
# and return the mean, covarience and 
def bayesFit(x, y):
    shape = x.shape[1]
    tMean = dict()
    tCov = dict()
    tPrior = dict()
    
    for i in range(10):
        curr = x[y == i]
        classSize = float(y[y == i].size)
        tMean[i] = curr.mean(axis=0)
        tCov[i] = np.cov(curr.T) + np.eye(shape)
        tPrior[i] = classSize / y.size
    
    return (tMean, tCov, tPrior)

# This function test the data and return the predict label
# Using c∗ = argmax(logP(x|c) + logP(c))
def bayesPredict(x, tMean, tCov, tPrior):
    # initialize the variables and results
    n = x.shape[0]
    results = np.zeros((n, 10))
    
    # Use the logpdf function to calculate the pdf
    for j in range(10):
        results[:,j] = multn.logpdf(x, mean=tMean[j], cov=tCov[j]) 
        + np.log(tPrior[j])
    
    # The result need to be argmax
    return np.argmax(results, axis=1)

# This function give the actual label and predict label and print out the
# summary information
def printSummary(predict, actual):
    check = (predict == actual)
    n = actual.size
    t = np.sum(check)
    percent = t/n
    print("True prediction: %d" % t)
    print("False prediction: %d" % (n - t))
    print("Correctness: " + "{:.2%}".format(percent))
    print()
    
    # return the percentage
    return percent

# This function construct the pca reduction on multiple components
# And do bayes and knn classification
def pcaReduction(start, stop, step):
    compArr = np.arange(start, stop, step)
    bayes = np.zeros(compArr.size)
    knns = np.zeros(compArr.size)
    j = 0
    
    for i in compArr:
        pca = PCA(n_components = i)
        pcaTrain = pca.fit_transform(trainData)
        pcaTest = pca.transform(testData)

        tMean, tCov, tPrior = bayesFit(pcaTrain, trainLabels)
        bayesRe = bayesPredict(pcaTest, tMean, tCov, tPrior)
        print("After %d component PCA reduction and bayes rule for classification, we have: " % i)
        bayes[j] = printSummary(bayesRe, testLabels)
        
        neigh = knn(n_neighbors = 3)
        neigh.fit(pcaTrain, trainLabels)
        knnRe = neigh.predict(pcaTest)
        print("After %d component PCA reduction and KNN for classification, we have: " % i)
        knns[j] = printSummary(knnRe, testLabels)
        j+=1
    
    return (compArr, bayes, knns)
    

# This function construct the pca reduction on multiple components
# And do bayes and knn classification
def ldaReduction(start, stop, step):
    compArr = np.arange(start, stop, step)
    bayes = np.zeros(compArr.size)
    knns = np.zeros(compArr.size)
    j = 0
    
    for i in compArr:
        lda = LDA(n_components = i)
        ldaTrain = lda.fit_transform(trainData, trainLabels)
        ldaTest = lda.transform(testData)

        tMean, tCov, tPrior = bayesFit(ldaTrain, trainLabels)
        bayesRe = bayesPredict(ldaTest, tMean, tCov, tPrior)
        print("After %d component LDA reduction and bayes rule for classification, we have: " % i)
        bayes[j] = printSummary(bayesRe, testLabels)
        
        neigh = knn(n_neighbors = 3)
        neigh.fit(ldaTrain, trainLabels)
        knnRe = neigh.predict(ldaTest)
        print("After %d component LDA reduction and KNN for classification, we have: " % i)
        knns[j] = printSummary(knnRe, testLabels)
        j+=1
    
    return (compArr, bayes, knns)


# (a) Bayes rules for classification on raw pixels
tMean, tCov, tPrior = bayesFit(trainData, trainLabels)
bayesRaw = bayesPredict(testData, tMean, tCov, tPrior)
print("After using bayes rule for classification, we have: ")
printSummary(bayesRaw, testLabels)

# (b) KNN for classification on raw pixels (k = 3)
# This may take long time to excute
neigh = knn(n_neighbors=3)
neigh.fit(trainData, trainLabels)
knnRaw = neigh.predict(testData)
print("After using KNN (k = 3) for classification, we have: ")
printSummary(knnRaw, testLabels)

# (c) Use the PCA demension reduction, use components from 10 to 110, step 20
# and use the LDA demension reduction, use components from 1 to 9, step 1
pcaComp, pcaB, pcaK = pcaReduction(10, 111, 20)
ldaComp, ldaB, ldaK = ldaReduction(1, 10, 1)

# (d) Plot the results in a graph
fig1 = plt.subplots()
plt.plot(pcaComp, pcaB, 'b', label = "Bayes Correctness")
plt.plot(pcaComp, pcaK, 'r', label = "KNN Correctness")
plt.legend(loc='upper left')
plt.title("Two Method Classification Correctness using PCA reduction")
plt.xlabel('Components')
plt.ylabel('Correctness')
plt.show()

fig2 = plt.subplots()
plt.plot(ldaComp, ldaB, 'b', label = "Bayes Correctness")
plt.plot(ldaComp, ldaK, 'r', label = "KNN Correctness")
plt.legend(loc='upper left')
plt.title("Two Method Classification Correctness using LDA reduction")
plt.xlabel('Components')
plt.ylabel('Correctness')
plt.show()
