## script to generate a probability data from original dataframe
import pandas as pd
import numpy as np

df = pd.read_csv("finalData_allmuts.csv", header=0)
df.fillna(np.nan)
# print(len(df.columns))

print(df.head())
## cleaning dataframe: Removing too many missing observations
empty_rows = pd.DataFrame(df.isna().sum()/df.shape[0], columns=["proportionMissing"])
empty_rows["colnames"] = empty_rows.index
# print(empty_rows.head())
nonMissingCols = np.array(empty_rows[empty_rows["proportionMissing"] <= 0.2]["colnames"])  # 20% or less missing values
# print(len(nonMissingCols))
noMissingdf = df[nonMissingCols]

# print(noMissingdf.head())

## create dummy variables
newdf = pd.get_dummies(noMissingdf, columns=noMissingdf.columns[2:]) # excluding sample name and group
# print(newdf.head())
mutcols = list(newdf.columns)[2:]

##### data cleaning ######
## remove rare (<1% of all samples)  varaints
for colname in mutcols:
    print(newdf[colname].sum(), newdf[colname].count())
    colprop = round(newdf[colname].sum() / newdf[colname].count(),2)
    # print(f"{colname}: Freq: {colprop}")
    if colprop <= 0.01 or colprop == 1: ## removing rare and mutation present in all isolates
        mutcols.remove(colname)
        print(f"removed {colname}")

## removing wild type mutations
mutsdf = pd.read_csv("mutPositions.csv", header=0)
nrows = mutsdf.shape[0]
# print(nrows)
for i in range(nrows):
    geneID = mutsdf.loc[i, "GENE"]
    posID = str(int(mutsdf.loc[i, "POS"]))
    wtallele = mutsdf.loc[i, "REF"]
    wtgenes = "_".join([geneID, posID, wtallele])
    print(f"wt gene {wtgenes}")
    if wtgenes in mutcols:
        mutcols.remove(wtgenes)
        print(f"removed wild type column {wtgenes}")    

# mutcols.append("group")
print(len(mutcols))
cleandf = newdf[mutcols]
cleandf["group"] = df["group"]
cleandf["sample"] = df["Sample"]

# cleandf["Group"] = noMissingdf["Group"]
# cleandf["SampleID"] = noMissingdf["Sample"]
cleandf.to_csv("cleanedData_MLinput.csv", index=False)
print(cleandf)

# ## calculating probabilities for hVISA and VSSA separately
# def getProbability(dataframeobj=cleandf, group="VSSA"):
#     dfCalc = dataframeobj[dataframeobj["Group"] == group]
#     dfnumeric = dfCalc.drop(["Group", "SampleID"], axis=1)
#     sumval = pd.DataFrame(round(dfnumeric.sum()/dfnumeric.count(),5), columns=["Probability"])
#     sumval["geneMut"] = sumval.index
#     sumval.replace("0.0", "1e-4")
#     # print(sumval)
#     return sumval
    # prob = round(dfCalc.sum/dfCalc.count(),2)
    # print(prob)

# vssaProbs = getProbability(dataframeobj=cleandf, group="VSSA")
# hvisaprobs = getProbability(dataframeobj=cleandf, group="HVISA")

# dfGroupProb = pd.merge(vssaProbs, hvisaprobs, on="geneMut")

# dfGroupProb.rename(columns={"Probability_x":"vssa_Prob", "Probability_y": "hvisa_Prob"}, inplace=True)
# print(dfGroupProb.head())
# dfGroupProb.to_csv("MutProbs.csv", index=False)
