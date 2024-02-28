import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import pickle
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
# CREATING DATAFRAME

def createInputData(inputdata):
  dfClassifier = pd.read_csv(inputdata, header=0)
  dfInput = pd.get_dummies(dfClassifier)
  # print(dfClassifier.head())
  return dfInput

def createTrainTest(dfclassifier, groupby):
  # dfclassifier = pd.read_csv(dfclassifier, header=0)
  yvals = list(dfclassifier[groupby])
  print(yvals)
  dataDF = dfclassifier.drop(groupby, axis=1)
  print(dataDF.head())
  # dataDF = dfclassifier.drop(columns=[i for i in dfclassifier.columns if i.startswith("Group") is True])
  
  # colLables2 = dfclassifier["Group_VSSA"]
  # smote = SMOTE(sampling_strategy='auto', random_state=42)
  # X_resampled, y_resampled = smote.fit_resample(X, y)
  xtrain, xtest, ytrain, ytest = train_test_split(dataDF, yvals, test_size=0.25, random_state=0)
  return xtrain, xtest, ytrain, ytest, yvals


def logitClassifier(xtrain, xtest, ytrain, ytest):
    # REGRESSION CLASSIFIER
    print("###### Running a regression (logistic) classifier on the data ######")
    classifier = LogisticRegression(random_state=123456789, max_iter=10000)
    classifier.fit(xtrain, ytrain)
    y_trainedPred = classifier.predict(xtrain)
    cm_trained = confusion_matrix(ytrain, y_trainedPred)
    print("Confusion Matrix for training data: \n", cm_trained)
    y_pred_test = classifier.predict(xtest)
    cm_testing = confusion_matrix(ytest, y_pred_test)
    print("Confusion Matrix : \n", cm_testing)
    return y_pred_test, cm_testing


def modelEvaluation(confusionMatrix, ytest, y_pred_test):
    accuracyVal = accuracy_score(ytest, y_pred_test)
    tn, fp, fn, tp = confusionMatrix.ravel()
    print("Major error: ", fp)
    print("Very major error: ", fn)
    precisionVal = tn / (tn+fp)
    recallVal = tp / (tp+fn)
    f1val = (2*precisionVal*recallVal)/(precisionVal + recallVal)
    print(f"recall: {recallVal: .2f} \n"
          f"precision {precisionVal: .2f} \n"
          f"F1 score: {f1val: .2f} \n"
          f"Accuracy: {accuracyVal: .2f}")

# RANDOM FOREST CLASSIFIER


def randForest(xtrain, xtest, ytrain, ytest):
    print("###### Running RF classifier on given data ######")
    classifier = RandomForestClassifier(random_state=123456789, n_estimators=1000)
    classifier.fit(xtrain, ytrain)
    y_pred_test_rf = classifier.predict(xtest)
    # print(y_pred_test_rf)
    confusionMatrix = confusion_matrix(ytest, y_pred_test_rf)
    print("Confusion Matrix using RF on predicted data: \n", confusionMatrix)
    return y_pred_test_rf, confusionMatrix

def getFeaturesRF(xtrain, ytrain, xtest, ytest, est = RandomForestClassifier):
   sel = SelectFromModel(est(random_state=123456789, n_estimators=1000).fit(xtrain,ytrain))
   sel.fit(xtrain, ytrain)
   selectedFeatures = sel.get_support()
   importances = sel.estimator_.feature_importances_
  #  plt.bar(range(xtrain.shape[1]), importances[np.argsort(importances)[::-1]],
      #  color="r", align="center")
   selectedCols = xtrain.columns[selectedFeatures]
  #  print(selectedCols)
  #  plt.xticks(range(xtrain.shape[1]), np.argsort(importances)[::-1])
  #  plt.show()
   return selectedCols
   
def crossValidation(rf_classifier, X_resampled, y_resampled, cv, scoring="recall"):
  cvScores = cross_val_score(rf_classifier, X_resampled, y_resampled, cv=cv, scoring="recall")
  return cvScores
   
def smoteResampling(X, y):
   desired_ratios = {0: 100, 1: 100}
   smote = SMOTE(sampling_strategy=desired_ratios, random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   return X_resampled, y_resampled
   



#### code #####
# df = pd.read_csv("testing.csv", header=0)
# dfnew = df.drop(["Sample"], axis=1)
# # print(dfnew.head())
# dfnew.fillna("99", inplace=True)
# xtrain, xtest, ytrain, ytest, colLabels = createTrainTest(dfnew, groupby="Group")
# y_pred_randforest, cmRandForest = randForest(xtrain, xtest, ytrain, ytest)
# modelEvaluation(cmRandForest, ytest, y_pred_randforest)

# featureCols = getFeaturesRF(xtrain, ytrain, xtest, ytest)
# # print(len(featureCols))
# dfFeatures = dfnew.drop([i for i in dfnew.columns if i not in featureCols], axis=1)
# # print(dfFeatures.shape)
# xtrain2, xtest2, ytrain2, ytest2 = train_test_split(dfFeatures, colLabels, test_size=0.25, random_state=0)
# y_pred_randforest2, cmRandForest2 = randForest(xtrain2, xtest2, ytrain2, ytest2)
# modelEvaluation(cmRandForest2, ytest, y_pred_randforest2)
# # with open("hVISARandForest.pkl", 'wb') as logmodelFile:
# #     pickle.dump(classifier2, logmodelFile)
# #     print(f"RF Model saved in: {logmodelFile}")

# y_pred_LR, cmLR = logitClassifier(xtrain, xtest, ytrain, ytest)
# modelEvaluation(cmLR, ytest, y_pred_LR)

# y_pred_LRFeature, cmLRFeature = logitClassifier(xtrain2, xtest2, ytrain2, ytest2)
# modelEvaluation(cmLRFeature, ytest2, y_pred_LRFeature)
