from sklearn import preprocessing
import pandas as pd
from numpy import genfromtxt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time
import random
import matplotlib.pyplot as resplt 
import matplotlib.pyplot as corrplt
import MatrixGeneration
from sklearn.impute import SimpleImputer
import sys, os




class DataSet():

    def __init__(self, name, datasetdir, labelcol, databegin, dataend, numclasses):
        self.name = name
        self.datasetdir = datasetdir
        self.labelcol = labelcol
        self.databegin = databegin
        self.dataend = dataend
        self.numclasses = numclasses

    def getname(self):
        return self.name
    def getdir(self):
        return self.datasetdir
    def getlabelcol(self):
        return self.labelcol
    def getbegin(self):
        return self.databegin
    def getend(self):
        return self.dataend
    def getnumclasses(self):
        return self.numclasses

class DataManager():

    def __init__(self):
        pass

    def getData(self, dataset, labelsColumn, dataBeginIndex, dataEndIndex):
        importedDataset = pd.read_csv(dataset, header=None)
        numColumns = len(importedDataset.columns)
        dataValues = genfromtxt(dataset, delimiter=',', usecols=range(
            dataBeginIndex, dataEndIndex)).tolist()

        # 1 == labels are in the first column. -1 == labels are in the last column
        if(labelsColumn == 1):
            labels = importedDataset.loc[:, 0].tolist()
        elif(labelsColumn == -1):
            labels = importedDataset.loc[:, (numColumns - 1)].tolist()

        return dataValues, labels

    def preprocessData(self, data):
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(data)
        imputedData = imputer.transform(data)  # nan values will take on mean
        scaledData = preprocessing.scale(imputedData).tolist()

        return scaledData

    def assignCodebook(self, labels, codebook):
        labelDictionary = {}
        updatedLabels = []
        # In order to randomize assignment of codewords.
        # random.shuffle(codebook)

        for originalLabel in labels:
            labelDictionary[originalLabel] = -1
        for label, codeword in zip(labelDictionary, codebook):
            labelDictionary[label] = codeword

        # Updating every original label to its corresponding codeword
        for label in labels:
            updatedLabels.append(labelDictionary[label])

        return updatedLabels, labelDictionary

    def assignCodeword(self, labels, codeword, index):
        labelDictionary = {}
        updatedLabels = []

        for originalLabel in labels:
            labelDictionary[originalLabel] = -1
        labelDictionary[labels[index]] = codeword

        # Updating every original label to its corresponding codeword
        for label in labels:
            updatedLabels.append(labelDictionary[label])

        return updatedLabels, labelDictionary

    def binarizeLabels(self, labelDictionary):
        updatedLabelsList = []
        classifierList = []
        for label in labelDictionary:
            updatedLabelsList.append(labelDictionary[label])

        codewordBits = len(updatedLabelsList[0])
        numClasses = len(updatedLabelsList)
        classifierIndex = 0
        classifierNumber = 0
        count = 1
        # The number of indices in a classifier is equal to the number of codewords/classes.
        # The number of classifiers is equal to the length or total number of bits in a codeword.
        # A classifier is made by getting a particular index's value for each and every codeword in
        # a codebook for every element in the codewords.
        while(classifierNumber < codewordBits):
            tempClassifier = []
            while(classifierIndex < numClasses):
                count += 1
                tempClassifier.append(
                    updatedLabelsList[classifierIndex][classifierNumber])
                classifierIndex += 1

            classifierIndex = 0
            classifierNumber += 1
            classifierList.append(tempClassifier)

        # Creating a dictionary of what all the original labels will be assumed to be using the classifiers.
        # Will use this later before training when updating all the labels to what their new binary value will be.
        tempDictionary = {}
        classifierDictionaryList = []
        for classifier in classifierList:
            for index, origLabel in zip(classifier, labelDictionary):
                tempDictionary[origLabel] = index

            classifierDictionaryList.append(tempDictionary)
            tempDictionary = {}

        return classifierDictionaryList

    # Updates the labels to their binary representation for a specific
    # column in the codebook
    def makeTrainingLabels(self, labelDictionaries, labels):
        allUpdatedLabels = []

        for dictionary in labelDictionaries:
            tempLabelList = []
            for label in labels:
                tempLabelList.append(dictionary[label])
            allUpdatedLabels.append(tempLabelList)

        return allUpdatedLabels

    def toCodeword(self, list):
        codeWordList = []
        tempList = []
        counter = 0

        while counter < len(list[0]):
            for prediction in list:
                tempList.append(prediction[counter])
            codeWordList.append(tempList)
            tempList = []
            counter += 1

        return codeWordList

    def originalY(self, updatedLabels, labelDictionary):
        originalLabels = []
        for label in updatedLabels:
            for key, value in labelDictionary.items():
                if value == label:
                    originalLabels.append(key)

        return originalLabels

    def getSmallClasses(self, data, labels):
        uniqueLabels = np.unique(labels)
        labelsToRemove = []
        dataToRemove = []
        indicesToRemove = []

        for uniqueLabel in uniqueLabels:
            indices = []
            index = 0
            for label in labels:
                if label == uniqueLabel:
                    indices.append(index)
                index += 1
            if (len(indices) < 3):
                for index in indices:
                    labelsToRemove.append(labels[index])
                    dataToRemove.append(data[index])
                    indicesToRemove.append(index)

        return indicesToRemove, dataToRemove, labelsToRemove

    def removeSmallClasses(self, data, labels, indicesToRemove):
        sortedIndicies = sorted(indicesToRemove, reverse=True)
        for index in sortedIndicies:
            del labels[index]
            del data[index]

        return data, labels

    def makeSplits(self, data, numSplits):
        dataSplits = []
        for row in data:
            newRow = []
            splitSize = len(row)/numSplits
            for split in range(numSplits):
                newRow.append(
                    row[int(splitSize*split):int(splitSize*(split+1))])
            dataSplits.append(newRow)
        return dataSplits

    def getDataFromSplits(self, dataSplits, indexToRemove):
        data = []
        for row in dataSplits:
            newRow = []
            for item in row[:indexToRemove] + row[indexToRemove+1:]:
                newRow += item
            data.append(newRow)
        return data

    def compilePredictions(self, *filenames):
        return np.hstack(list(genfromtxt(filename, dtype=int).reshape(-1, 1)
                              for filename in filenames)).tolist()

class Trainer():

    def __init__(self):
        pass

   # Return models so that predictions can be done later.
    def trainClassifiers(self, knownData, knownLabels, model):
        trainedModels = []

        for labels in knownLabels:
            if model == 1:
                classifier = svm.SVC(gamma='auto')
            elif model == 2:
                classifier = DecisionTreeClassifier(random_state=0)
            elif model == 3:
                classifier = LinearDiscriminantAnalysis()
            elif model == 4:
                classifier = KNeighborsClassifier(n_neighbors=2)
            elif model == 5:
                classifier = LogisticRegression(random_state=0, solver="lbfgs")
            elif model == 6:
                classifier = GaussianNB()
            elif model == 7:
                classifier = RandomForestClassifier(
                random_state=0, n_estimators=10)
            else:
                print("Specify Classifier")
            classifier = classifier.fit(knownData, labels)
            trainedModels.append(classifier)

        return trainedModels

    def unPackShuffled(self, packed, p, seed):
        first = []
        second = []
        random.seed(seed)
        sample = random.sample(packed, int(p*len(packed)))
        for pair in sample:
            first.append(pair[0])
            second.append(pair[1])
        return first, second

    def pack(self, setA, setB):
        packed = []
        for x, y in zip(setA, setB):
            packed.append((x, y))
        return packed

    def trainClassifiers_VD(self, knownData, knownLabels, model, p):
        trainedModels = []
        counter = 0
        for labels in knownLabels:
            packed = self.pack(knownData, labels)
            pdata, plabels = self.unPackShuffled(packed, p, counter)
            if model == 1:
                classifier = svm.SVC(gamma='auto')
            elif model == 2:
                classifier = DecisionTreeClassifier(random_state=0)
            elif model == 3:
                classifier = LinearDiscriminantAnalysis()
            elif model == 4:
                classifier = KNeighborsClassifier(n_neighbors=2)
            elif model == 5:
                classifier = LogisticRegression(random_state=0, solver="lbfgs")
            elif model == 6:
                classifier = GaussianNB()
            elif model == 7:
                classifier = RandomForestClassifier(
                    random_state=0, n_estimators=10)
            else:
                print("Specify Classifier")
            counter += 1
            classifier = classifier.fit(pdata, plabels)
            trainedModels.append(classifier)

        return trainedModels

    def trainClassifiers_VD_chosen_seed(self, knownData, knownLabels, model, p, seed):
        trainedModels = []
        for labels in knownLabels:
            packed = self.pack(knownData, labels)
            pdata, plabels = self.unPackShuffled(packed, p, seed)
            if model == 1:
                classifier = svm.SVC(gamma='auto')
            elif model == 2:
                classifier = DecisionTreeClassifier(random_state=0)
            elif model == 3:
                classifier = LinearDiscriminantAnalysis()
            elif model == 4:
                classifier = KNeighborsClassifier(n_neighbors=2)
            elif model == 5:
                classifier = LogisticRegression(random_state=0, solver="lbfgs")
            elif model == 6:
                classifier = GaussianNB()
            elif model == 7:
                classifier = RandomForestClassifier(
                random_state=0, n_estimators=10)
            else:
                print("Specify Classifier")
            classifier = classifier.fit(pdata, plabels)
            trainedModels.append(classifier)

        return trainedModels

    def trainClassifiersMultiple(self, knownData, knownLabels, modelArray):
        trainedModels = []
        counter = 0
        for labels in knownLabels:
            model = modelArray[counter]
            if model == 1:
                classifier = svm.SVC(gamma='auto')
            elif model == 2:
                classifier = DecisionTreeClassifier(random_state=0)
            elif model == 3:
                classifier = LinearDiscriminantAnalysis()
            elif model == 4:
                classifier = KNeighborsClassifier(n_neighbors=2)
            elif model == 5:
                classifier = LogisticRegression(random_state=0, solver="lbfgs")
            elif model == 6:
                classifier = GaussianNB()
            elif model == 7:
                classifier = RandomForestClassifier(
                    random_state=0, n_estimators=10)
            else:
                print("Specify Classifier")
            counter += 1
            classifier = classifier.fit(knownData, labels)
            trainedModels.append(classifier)

        return trainedModels



    def trainClassifiersMultiple_VD(self, knownData, knownLabels, modelArray, p):
        trainedModels = []
        counter = 0
        for labels in knownLabels:
            model = modelArray[counter]
            packed = self.pack(knownData, labels)
            pdata, plabels = self.unPackShuffled(packed, p, counter)
            if model == 1:
                classifier = svm.SVC(gamma='auto')
            elif model == 2:
                classifier = DecisionTreeClassifier(random_state=0)
            elif model == 3:
                classifier = LinearDiscriminantAnalysis()
            elif model == 4:
                classifier = KNeighborsClassifier(n_neighbors=2)
            elif model == 5:
                classifier = LogisticRegression(random_state=0, solver="lbfgs")
            elif model == 6:
                classifier = GaussianNB()
            elif model == 7:
                classifier = RandomForestClassifier(
                    random_state=0, n_estimators=10)
            else:
                print("Specify Classifier")
            counter += 1
            classifier = classifier.fit(pdata, plabels)
            trainedModels.append(classifier)

        return trainedModels

    def trainClassifier(self, knownData, knownLabels, model, deleteTernarySymbol, column):
        if model == 1:
            classifier = svm.SVC(gamma='auto', random_state=0)
        elif model == 2:
            classifier = DecisionTreeClassifier(random_state=0)
        elif model == 3:
            classifier = LinearDiscriminantAnalysis()
        elif model == 4:
            classifier = KNeighborsClassifier(
                n_neighbors=3)
        elif model == 5:
            classifier = LogisticRegression(random_state=0, solver="lbfgs")
        elif model == 6:
            classifier = GaussianNB()
        elif model == 7:
            classifier = RandomForestClassifier(
                random_state=0, n_estimators=10)
        else:
            print("Specify Classifier")

        dataCopy = knownData.copy()
        labelsCopy = knownLabels[column].copy()

        if deleteTernarySymbol:
            indicesToDelete = []
            # Find indices of samples that are marked with the ternary symbol
            for index, label in enumerate(knownLabels[column]):
                if label != 1 and label != 0:
                    indicesToDelete.append(index)

            # Delete indices from data and labels. Convert back to list (from np array) afterwards.
            sortedIndicies = sorted(indicesToDelete, reverse=True)
            for index in sortedIndicies:
                del dataCopy[index]
                del labelsCopy[index]

        return classifier.fit(dataCopy, labelsCopy)

    # Converts list containing multiple numpy arrays to list of lists containing codewords.
    def toCodeword(self, list):
        codeWordList = []
        tempList = []
        counter = 0

        while counter < len(list[0]):
            for prediction in list:
                tempList.append(prediction[counter])
            codeWordList.append(tempList)
            tempList = []
            counter += 1

        return codeWordList

    # Used trained classifiers to get predictions. Predictions will construct codewords.
    def getPredictions(self, validationData, trainedClassifiers):
        predictionList = []
        nonCodeworded = []
        for classifier in trainedClassifiers:
            predictions = classifier.predict(validationData)
            predictionList.append(predictions)
            nonCodeworded.append(predictions)

        predictionList = self.toCodeword(predictionList)

        return predictionList, nonCodeworded

    def getPrediction(self, validationData, trainedClassifier):
        return trainedClassifier.predict(validationData)

    def getProbabilities(self, missclassifieds, classifer1, classifier2):
        #for missclassifications in missclassifieds:
        numMissClass1 = 0
        numMissClass2 = 0
        numBoth = 0
        for code in missclassifieds:
            if code[classifer1] == 1:
                numMissClass1 += 1
            if code[classifier2] == 1:
                numMissClass2 += 1
            if code[classifer1] == 1 and code[classifier2] == 1:
                numBoth += 1
        probprod = ((numMissClass1)*(numMissClass2))/(len(missclassifieds)**2)
        probboth = numBoth/len(missclassifieds)
        return probboth, probprod, (numMissClass1/len(missclassifieds)), (numMissClass2/len(missclassifieds))


    def getProbabilitiesB(self, miss1, miss2, index1, index2):
        # for missclassifications in missclassifieds:
        numMissClass1 = 0
        numMissClass2 = 0
        numBoth = 0
        for code1, code2 in zip(miss1, miss2):
            if code1[index1] == 1:
                numMissClass1 += 1
            if code2[index2] == 1:
                numMissClass2 += 1
            if code1[index1] == 1 and code2[index2] == 1:
                numBoth += 1
        probprod = ((numMissClass1) * (numMissClass2)) / (len(miss1) ** 2)
        probboth = numBoth / len(miss1)
        return probboth, probprod, (numMissClass1 / len(miss1)), (numMissClass2 / len(miss1))

    def correlationMatrix(self, missclassifieds):
        matrix = []
        for i in range(len(missclassifieds)-1):
            matrix.append([])
            for j in range(len(missclassifieds)-1):
                #print(i, j)
                #print(matrix)
                a, b, c, d = self.getProbabilities(missclassifieds, i, j)
                if (((c*d*(1-c)*(1-d))**(1/2) != 0)):
                    matrix[i].append((a-b)/((c*d*(1-c)*(1-d))**(1/2)))
                else:
                    matrix[i].append(a-b)
        return matrix

    def correlationMatrixVaried(self, missclassified1, missclassified2):
        matrix = []
        k = len(missclassified1)
        for i in range(k - 1):
            matrix.append([])
            for j in range(k - 1):
                # print(i, j)
                # print(matrix)
                a, b, c, d = self.getProbabilities(missclassified1, missclassified2, i, j)
                matrix[i].append((a - b) / ((c * d * (1 - c) * (1 - d)) ** (1 / 2)))
        return matrix
    
    # Sort the answer key
    # Go one to one between what it has
    # and what the classifier says.
    def testClassifiers(self, trainedModels, unseenData, answerKey):
        accuracies = []#np.array([])
        guesses = []
        oneToOneComparison = np.transpose(answerKey)
        for classifier in trainedModels:
            guesses.append(classifier.predict(unseenData))
        for guess, actual in zip(guesses, oneToOneComparison):
            accuracies.append(1-(MatrixGeneration.hammingDistance(guess, actual)/len(guess)))
        accuraciesNP = np.array(accuracies)
        return accuraciesNP


    # Takes codewords (usually predicted codewords) and "updates" them to whatever codeword they are
    # closest to (with respect to hamming distance) in a given codebook. Will also return a list that
    # shows what the minimum hamming distances were when deciding which codeword to updated the predicted
    # codeword with.
    def hammingDistanceUpdater(self, codebook, codewords):
        minHamWord = []
        # List containing actual CW based off of shortest HD
        UpdatedList = []
        minHamList = []
        for predictedCode in codewords:
            minHam = len(predictedCode)
            for actualCode in codebook:
                hammingDistance = 0
                for counter in range(0, len(predictedCode)):
                    if actualCode[counter] != predictedCode[counter]:
                        hammingDistance += 1
                if hammingDistance < minHam:
                    minHam = hammingDistance
                    minHamWord = actualCode

            UpdatedList.append(minHamWord)
            minHamList.append(minHam)

        return UpdatedList, minHamList

    # Gets accuracy of predicted codewords when compared to
    # actual (i.e. validation) codewords

    def compare(self, predictions, actual):
        total = len(predictions)
        right = 0

        for (x, y) in zip(predictions, actual):
            if x == y:
                right += 1

        percentRight = right * 1.0 / total

        return percentRight

class CrossValidator:

    def __init__(self, trainingset, labels):
        self.trainingset = trainingset
        self.labels = labels
        self.n = len(trainingset)

    # Associates labels with data before processing
    def PackLabels(self):
        packed = []
        for i in range(self.n):
            packed.append((self.trainingset[i], self.labels[i]))
            # self.trainingset[i].append(self.labels[i])
        self.trainingset = packed

        return packed

    def unPack(self, pastry):
        trainingSet = []
        labelSet = []
        counter = 0
        for crisp in pastry:
            trainingSet.append([])
            labelSet.append([])
            for crunch in crisp:
                trainingSet[counter].append(crunch[0])
                labelSet[counter].append(crunch[1])
            counter += 1

        return trainingSet, labelSet

    def removeTestingLabel(self, testingsets):
        testing = []
        validating = []
        counter = 0
        for set in testingsets:
            testing.append([])
            validating.append([])
            for val in set:
                testing[counter].append(val[0])
                validating[counter].append(val[1])
            counter += 1
        return testing, validating

    # Returns training sets

    def ECOCValidator(self, k):
        self.PackLabels()
        random.seed(1)
        random.shuffle(self.trainingset)

        trainingsets = []
        testingsets = []
        foldsize = len(self.trainingset)/k

        for i in range(k):
            lower = int(foldsize*i)
            upper = int(foldsize*(i+1))
            testingsets.append(self.trainingset[lower:upper])
            trainingsets.append(
                self.trainingset[:lower] + self.trainingset[upper:])

        t, k = self.unPack(trainingsets)
        l, m = self.removeTestingLabel(testingsets)
        return t, k, l, m

    # Removes the labels from the data

    def RemoveLabels(self, labelledSet):
        labels = []
        for i in range(len(labelledSet)):
            labels.append(labelledSet[i][-1])
            labelledSet[i] = labelledSet[i][:-1]
        return labels

    def averageMatrices(self, matrixSet):
        l = len(matrixSet)
        iter = len(matrixSet[0])
        averaged = matrixSet[0]
        for i in range(1, l):
            for j in range(iter):
                for k in range(iter):
                    averaged[j][k] += matrixSet[i][j][k]
        for a in range(iter):
            for b in range(iter):
                averaged[a][b] = (averaged[a][b] / l)

        return averaged

    def averageTotal(self, matrix):
        average = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                average += abs(matrix[i][j])
        return average/(len(matrix)*len(matrix[0]))

    def processMatrix(self, matrix):
        for row in matrix:
            for item in row:
                if item == 1:
                    item = 0
        return matrix

    def matrixStats(self, matrix):
        matrixarray = []
        for row in matrix:
            for item in row:
                matrixarray.append(item)
        stdv = np.std(matrixarray)
        MAX = max(matrixarray)
        MIN = min(matrixarray)
        return stdv

# Starting from 1
models_String = ["SVM", "DT", "LDA", "KNN",
          "LogisticRegression", "GaussianNB", "RandomForest"]


'''
datasets_Reference = [
    {   # 0
        "name": "Pendigits [Pen-Based Recognition of Handwritten Digits]",
        "link": "pendigits.csv",
        "labelCol": -1,
        "beginData": 3,
        "endData": 13,
        "numClasses": 10
    },
    {   # 1
        "name": "Vowel [Connectionist Bench (Vowel Recognition - Deterding Data)]",
        "link": "vowel-context.data",
        "labelCol": -1,
        "beginData": 9,
        "endData": 13,
        "numClasses": 11
    },
    {  # 2
        "name": "Letter Recognition",
        "link": "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
        "labelCol": 1,
        "beginData": 1,
        "endData": 17,
        "numClasses": 26
    },
    {  # 3
        "name": "Auslan [Australian Sign Language signs]",
        "link": "signs.data",
        "labelCol": -1,
        "beginData": 0,
        "endData": 9,
        "numClasses": 95
    },
    {   # 4
        "name": "! Sector [Labor Negotiations?]",
        "link": "",
        "labelCol": 0,
        "beginData": 0,
        "endData": 0,
        "numClasses": 0
    },
    {   # 5
        "name": "Aloi",
        "link": "aloi.data",
        "labelCol": -1,
        "beginData": 1,
        "endData": 129,
        "numClasses": 1000
    },
    {  # 6
        "name": "Glass Identification",
        "link": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        "labelCol": -1,
        "beginData": 1,
        "endData": 10,
        "numClasses": 6
    },
    {   # 7
        "name": "Satimage [Statlog (Landsat Satellite)]",
        "link": "sat.trn",
        "labelCol": -1,
        "beginData": 0,
        "endData": 36,
        "numClasses": 6
    },
    {  # 8
        "name": "Usps",
        "link": "usps.data",
        "labelCol": 1,
        "beginData": 1,
        "endData": 257,
        "numClasses": 10
    },
    {   # 9
        "name": "Segment [Statlog (Image Segmentation)]",
        "link": "segment.data",
        "labelCol": -1,
        "beginData": 0,
        "endData": 19,
        "numClasses": 7
    },
    {   # 10
        "name": "Auslan (High Quality)",
        "link": "auslan.data",
        "labelCol": -1,
        "beginData": 0,
        "endData": 22,
        "numClasses": 96
    }
]
'''
#Nothing underscore means base, VC means varied classifier, VD means varied data, VCVD means both.
#First codebook is always hadamard, second is always random
#To make for multiple models, make a helper function which does the loop and calls base.
#Make sure model selection is native, edit it here.
#Iterate data by making data objects, indexing them, then iterating.

def getECOCBaselineAccuracies_VC_VD(dataset, listOfCBs, labelCol, beginData, endData, modelChoices, p):
    ts = time.time()
    RunLabel = "VC_VD_"
    dataset_str_array = dataset.split("/")
    dataset_str_name = dataset_str_array[len(dataset_str_array)-1].strip(".csv")
    print("RUNNING VARIED VARIED", modelChoices)
    dm = DataManager()
    trainer = Trainer()
    codebookNum = 1
    models = ["SVM", "DT", "LDA", "KNN"] # Used for printing

    for codebook in listOfCBs:
        # Lists to hold each iteration's accuracy for each model
        accuracies = []
        print("\tCodebook Number:", codebookNum)
        modelArray = []
        random.seed(1)
        for i in range(len(codebook[0])):
            modelArray.append(random.choice(modelChoices))

        # Get and preprocess the data
        dataraw, labels = dm.getData(dataset, labelCol, beginData, endData)

        indicesToRemove, dataToRemove, labelsToRemove = dm.getSmallClasses(dataraw, labels)
        data, labels = dm.removeSmallClasses(dataraw, labels, indicesToRemove)

        data = dm.preprocessData(data)


        # Give each label a codeword then split the data
        updatedLabels, labelDictionary = dm.assignCodebook(labels, codebook)

        CV = CrossValidator(data, updatedLabels)
        trainingsets, labels, testingsets, validsets = CV.ECOCValidator(10)
        matrices = []
        matrixholder = []
        folds = []
        counter = 0
        auxilliaryMatrixTests = []
        binaryClassifiers = dm.binarizeLabels(labelDictionary)
        foldedPredictions = []
        individualaccuracies_holder = []
        #print(x_train, y_train)
        for trainingset, label, predset, testset in zip(trainingsets, labels, testingsets, validsets):
            #print(t, l)
            # Create lists of what each class's new label should be based off of the
            # columns of the codewords

            # Since splitting happens after the assigning/updating of codwords, we need to get the
            # original labels back so that makeTrainingLabels works properly (This can be improved upon)

            originalTrainLabels = dm.originalY(label, labelDictionary)

            # Train the models
            trainingLabels = dm.makeTrainingLabels(binaryClassifiers, originalTrainLabels)


            # trainedModels = trainer.trainClassifiers(x_train, trainingLabels, i)

            folds.append(trainer.trainClassifiersMultiple_VD(trainingset, trainingLabels, modelArray, p))
            predictionsMixed, literalsMixed = trainer.getPredictions(predset, folds[counter])
            cMatrix = trainer.correlationMatrix(literalsMixed)
            matrixholder.append(cMatrix)
            auxilliaryMatrixTests.append(cMatrix)
            

            individualaccuracies = trainer.testClassifiers(folds[counter], predset, testset)
            individualaccuracies_holder.append(np.average(individualaccuracies))

            

            updatedPredictionsBase, minHams = trainer.hammingDistanceUpdater(codebook, predictionsMixed)
            foldedPredictions.append(updatedPredictionsBase)
            accuracyBase = trainer.compare(updatedPredictionsBase, testset)
            print("\t\tECOC Acuracy for fold " + str(counter+1) + ":", accuracyBase)
            print("\t\tAverage Individual Accuracy for fold " + str(counter+1) + ":", individualaccuracies_holder[counter])
            print("\t\tCorrelation: ", CV.averageTotal(cMatrix))
            print("\t\tStdDev Corr: ", CV.matrixStats(cMatrix))
            accuracies.append(accuracyBase)

            counter += 1

        ecoc_average = np.average(accuracies)
        ecoc_std = np.std(accuracies)
        print("Average Individual Accurcy: ", np.average(individualaccuracies_holder))
        print("Average ECOC Accuracy: ", np.average(accuracies))
        print("Runtime", time.time() - ts)
        '''
        count = 0
        for amat in auxilliaryMatrixTests:
            plt.clf()
            plt.figure(figsize=(20, 15))
            plt.set_cmap('BrBG_r')
            ax = plt.subplot()
            sns.heatmap(amat, annot=True, ax=ax)
            plt.suptitle("Fold" + str(count+1), fontsize=20)
            #plt.show()
            count += 1
            #plt.savefig("Fold" + str(count) + ".png")
        '''
        matrices.append(CV.averageMatrices(matrixholder))
        '''
        for m in matrices:
            plt.clf()
            plt.figure(figsize=(20, 15))
            plt.set_cmap('BrBG_r')
            ax = plt.subplot()
            sns.heatmap(m, annot=True, ax=ax)
            plt.suptitle("Varied Data/Classifier Averaged " + dataset_str_name + str(modelChoices), fontsize=15)
            plt.savefig(RunLabel + dataset_str_name + str(modelChoices) + ".png")
        '''
            
        corr_average = CV.averageTotal(matrices[0])
        corr_std = CV.matrixStats(matrices[0])
        ecoc_data = [round(ecoc_average, 3), round(ecoc_std, 3)]
        corr_data = [round(corr_average, 3), round(corr_std, 3)]
        print("CV Average Correlation: ", CV.averageTotal(matrices[0]))
        print("CV Average Corr Std", CV.matrixStats(matrices[0]))

        codebookNum += 1
    return ecoc_data, corr_data


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def run_test(datasetdir, label_col, data_begin, data_end, numclasses, model_array, graphing, printing, outfile):
    result_log = open(outfile, "w")
    corrplt = resplt.subplot()
    if(not printing):
        blockPrint()
    dataset_str_array = datasetdir.split("/")
    dataset_str_name = dataset_str_array[len(dataset_str_array)-1].strip(".csv")
    codebook = MatrixGeneration.GenerateMatrix(numclasses, numclasses)

    listOfCBs = [codebook]

    accuracy_array = [[]]
    correlation_array = [[]]

    for model in model_array:
        accuracy_array.append([])
        correlation_array.append([])
    print("starting run")

    pdomain = []
    p = 1.0
    while p >= 0.2:
        print("DATA PER CLASSIFIER " + str(p))
        pdomain.append(round(p, 2))
        counter = 0
        for model in model_array:
            temp = getECOCBaselineAccuracies_VC_VD(datasetdir, listOfCBs, label_col, data_begin, data_end, [model], p)
            accuracy_array[counter].append(temp[0])
            correlation_array[counter].append(temp[1])
            counter += 1
        mixed = getECOCBaselineAccuracies_VC_VD(datasetdir, listOfCBs, label_col, data_begin, data_end, model_array, p)
        accuracy_array[len(accuracy_array)-1].append(mixed[0])
        correlation_array[len(correlation_array)-1].append(mixed[1])
        p = p - 0.1

    labels = "P     " 
    accuracy_results = ""
    correlation_results = ""

    for j in range(len(model_array)):
        labels += models_String[model_array[j]-1] + "     "
    labels += "ALL\n"

    for i in range(len(pdomain)):
        accuracy_results += str(pdomain[i]) + "     "
        for j in range(len(model_array)+1):
            accuracy_results += str(accuracy_array[j][i]) + "     "
        accuracy_results += "\n"
    for i in range(len(pdomain)):
        correlation_results += str(pdomain[i]) + "     "
        for j in range(len(model_array)+1):
            correlation_results += str(correlation_array[j][i]) + "     "
        correlation_results += "\n"

    result_log.write(labels + "Accuracies\n" + accuracy_results + "Correlation\n" + correlation_results)
        
    print(accuracy_array)
    print(correlation_array)
    if (graphing):
        accplt = [[]]
        corrplt = [[]]
        for model in model_array:
            accplt.append([])
            corrplt.append([])
        counter = 0
        for acc, corr in zip(accuracy_array, correlation_array):
            for modelacc, modelcorr in zip(acc, corr):
                accplt[counter].append(modelacc[0])
                corrplt[counter].append(modelcorr[0])
            counter+=1
        line_references = ['-', ':', '-.', '--']
        resplt.suptitle(dataset_str_name + " Varied Data Accuracies")
        resplt.xlabel("Percent of Data Per Learner")
        resplt.ylabel("Accuracy")
        
        for i in range(len(model_array)):
            resplt.plot(pdomain, accplt[i], line_references[i], label= models_String[model_array[i]-1])

        resplt.plot(pdomain, accplt[len(accplt)-1], line_references[len(line_references)-1], label = "All")
        resplt.legend(loc= "lower right")
        resplt.savefig(dataset_str_name + " Varied Data Accuracies.png")

        resplt.clf()

        resplt.suptitle(dataset_str_name + " Varied Data Correlation")
        resplt.xlabel("Percent of Data Per Learner")
        resplt.ylabel("Correlation")
        for i in range(len(model_array)):
            resplt.plot(pdomain, corrplt[i], line_references[i], label= models_String[model_array[i]-1])
        
        resplt.plot(pdomain, corrplt[len(corrplt)-1], line_references[len(line_references)-1], label = "All")
        resplt.legend(loc= "lower right")
        resplt.savefig(dataset_str_name + " Varied Data Correlation.png")
    result_log.close()
    return result_log

def main():
    if(len(sys.argv) > 5):
        print("Running From the Command line")
        dataset = sys.argv[1]
        labelscol = int(sys.argv[2])
        databegin = int(sys.argv[3])
        dataend = int(sys.argv[4])
        numclasses = int(sys.argv[5])
        models_string = sys.argv[6]
        graphing = sys.argv[7]
        printing = sys.argv[8]
        fname = sys.argv[9]
        models = []
        print("Printing: " + str(bool(printing)))
        print("Graphing: " + str(bool(graphing)))
        for m in models_string:
            models.append(int(m))
        log = run_test(dataset, labelscol, databegin, dataend, numclasses, models, graphing, printing, fname)
        return log
    else:
        #Run play button code here.
        return 0




#run_test("datasets/pendigits.csv", -1, 0, 12, 10, [2, 4, 7], False, True, "pendigitsData.txt")


main() 

'''
python3 MasterTester_V2.py "datasets/pendigits.csv" -1 0 12 10 247 1 0 "./penDigits_cmd.txt"
'''