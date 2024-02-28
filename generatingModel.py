import pandas as pd
from createModel import *
from sklearn.impute import SimpleImputer

## select statistically significant mutations
dfStats = pd.read_csv("statsData.csv", header=0)
finalCols = list(dfStats["Mutation"])

## mutations with odds ratio > 2 and p-value < 0.05
# finalCols = ["GraR_148_Q", "GraS_59_L",  "LytR_192_A", "MprF_171_A", "MprF_174_L", "MprF_194_Y", "MprF_223_V", "MprF_371_L",
            #  "MprF_400_Y", "MprF_406_I", "MprF_409_I", "MprF_413_L", "MprF_426_V", "MprF_430_A", "MprF_446_I", "MprF_451_L",
            #  "MprF_575_I", "MprF_692_E", "PhoR_101_K", "SaeS_268_E", "TcaA_237_H", "Group"]
print(len(finalCols))
finalCols.append("Group")
df = pd.read_csv("cleanedData_MLinput.csv", header=0)
for colname in df.columns:
  if colname not in finalCols:
    df = df.drop(columns=colname, axis=1)
print(df["Group"])
print(df.shape)
# print(df.head())
## no imputations
imputer = SimpleImputer(strategy='most_frequent') # using imputation for handling missing data
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
xtrain, xtest, ytrain, ytest, colLabels = createTrainTest(df_imputed, groupby="Group")
y_pred_randforest, cmRandForest = randForest(xtrain, xtest, ytrain, ytest)
modelEvaluation(cmRandForest, ytest, y_pred_randforest)

# print("Extracting features..")
featureCols = getFeaturesRF(xtrain, ytrain, xtest, ytest,est="RandomForestClassifier")
dfFeatures = df_imputed.drop([i for i in df_imputed.columns if i not in featureCols], axis=1)
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(dfFeatures, colLabels, test_size=0.25, random_state=0)
print("rf on extracted features")
y_pred_randforestfeature, cmRandForestfeature = randForest(xtrain2, xtest2, ytrain2, ytest2)
modelEvaluation(cmRandForestfeature, ytest2, y_pred_randforestfeature)
