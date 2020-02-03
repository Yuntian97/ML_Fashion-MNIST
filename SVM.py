# -*- coding: utf-8 -*-
# This part explore the SVM method and results

# This is the area of import package
import numpy as np
import matplotlib.pyplot as plt
import mnist_reader

# Import PCA and LDA package here
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Import SVM package here
from sklearn import svm

# We first use the mnist_reader to store into the numpy arrays
trainData, trainLabels = mnist_reader.load_mnist('data', kind='train')
testData, testLabels = mnist_reader.load_mnist('data', kind='t10k')

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

# This function calculate SVM result for different reduction method
def calcSVM(iterArr, trainArr, testArr, redType, svmObj, svmType):
    resultArr = np.zeros(iterArr.size)
    for i in range(iterArr.size):
        reTrain = trainArr[i]
        reTest = testArr[i]
        
        svmObj.fit(reTrain, trainLabels)
        svmRe = svmObj.predict(reTest)
        print("After %d component %s reduction and %s for classification, we have: " 
              % (iterArr[i], redType, svmType))
        resultArr[i] = printSummary(svmRe, testLabels)
    
    return resultArr

# This function get the LDA reduction dataset
def LDAreduct(ldaIter):
    ldaTrainArr = []
    ldaTestArr = []
    for i in ldaIter:
        thisLDA = LDA(n_components = i)
        ldaTrainArr.append(thisLDA.fit_transform(trainData, trainLabels))
        ldaTestArr.append(thisLDA.transform(testData))
    
    return (ldaTrainArr, ldaTestArr)

# This function get the PCA reduction dataset
def PCAreduct(pcaIter):
    pcaTrainArr = []
    pcaTestArr = []
    for i in pcaIter:
        thisPCA = PCA(n_components = i)
        pcaTrainArr.append(thisPCA.fit_transform(trainData))
        pcaTestArr.append(thisPCA.transform(testData))
    
    return (pcaTrainArr, pcaTestArr)

# 0. calculate the PCA and LDA reductions of the dataset
pcaIterArr = np.arange(70, 171, 20)
ldaIterArr = np.arange(3, 10, 1)
ldaTrainArr, ldaTestArr = LDAreduct(ldaIterArr)
pcaTrainArr, pcaTestArr = PCAreduct(pcaIterArr)

# Section 1: Use the different SVM without any dimension reduction and get the results
# 1. Use the Linear SVM without any dimension reduction
thisSVM = svm.LinearSVC()
thisSVM.fit(trainData, trainLabels)
svmRaw = thisSVM.predict(testData)
print("After using Linear SVM for classification, we have: ")
printSummary(svmRaw, testLabels)

# 2. Use the Polynomial kernal SVM without any dimension reduction
thisSVM = svm.SVC(kernel='poly')
thisSVM.fit(trainData, trainLabels)
svmRaw = thisSVM.predict(testData)
print("After using poly kernal SVM for classification, we have: ")
printSummary(svmRaw, testLabels)

# 3. Use the RBF kernal SVM without any dimension reduction
thisSVM = svm.SVC(kernel='rbf')
thisSVM.fit(trainData, trainLabels)
svmRaw = thisSVM.predict(testData)
print("After using rbf kernal SVM for classification, we have: ")
printSummary(svmRaw, testLabels)

# Section 2: Use LDA and PCA on different SVM and get the classification results
# 1. Use the PCA and LDA dimension reduction for Linear SVM
pcaL = calcSVM(pcaIterArr, pcaTrainArr, pcaTestArr, "PCA", svm.LinearSVC(), "Linear SVM")
ldaL = calcSVM(ldaIterArr, ldaTrainArr, ldaTestArr, "LDA", svm.LinearSVC(), "Linear SVM")

# 2. Use the PCA and LDA dimension reduction for Polynomial kernal SVM
pcaP = calcSVM(pcaIterArr, pcaTrainArr, pcaTestArr, "PCA", svm.SVC(kernel='poly'), "Poly Kernal SVM")
ldaP = calcSVM(ldaIterArr, ldaTrainArr, ldaTestArr, "LDA", svm.SVC(kernel='poly'), "Poly Kernal SVM")

# 3. Use the PCA and LDA dimension reduction for rbf kernal SVM
pcaR = calcSVM(pcaIterArr, pcaTrainArr, pcaTestArr, "PCA", svm.SVC(kernel='rbf'), "RBF Kernal SVM")
ldaR = calcSVM(ldaIterArr, ldaTrainArr, ldaTestArr, "LDA", svm.SVC(kernel='rbf'), "RBF Kernal SVM")

# 4. plot the graph
fig1 = plt.subplots()
plt.plot(pcaIterArr, pcaL, 'g', label = "Linear Kernal")
plt.plot(pcaIterArr, pcaP, 'b', label = "Poly Kernal")
plt.plot(pcaIterArr, pcaR, 'r', label = "RBF Kernal")
plt.legend(loc='upper left')
plt.title("Different SVM Kernal Classification Correctness using PCA reduction")
plt.xlabel('Components')
plt.ylabel('Correctness')
plt.show()

fig2 = plt.subplots()
plt.plot(ldaIterArr, ldaL, 'g', label = "Linear Kernal")
plt.plot(ldaIterArr, ldaP, 'b', label = "Poly Kernal")
plt.plot(ldaIterArr, ldaR, 'r', label = "RBF Kernal")
plt.legend(loc='upper left')
plt.title("Different SVM Kernal Classification Correctness using LDA reduction")
plt.xlabel('Components')
plt.ylabel('Correctness')
plt.show()





























