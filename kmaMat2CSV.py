import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from itertools import permutations, product
### The script accepts a matrix file from KMA and returns a CSV file with all amino acid mutations. This CSV file can be ued for predicting hVISA ####
### The results should be used for research purposes only and cannot be used to guide treatment for patients ####

def read_matrix(matrix_file, outfile):
    """Remove gapped alignment values in the matrix file and write a TSV file for Pandas."""
    all_data = []

    # Read the matrix file
    with open(matrix_file, 'r', encoding="utf-8", newline="\n") as fin:
        data = [line.rstrip().strip() for line in fin]

    # Define base pairs
    bps = ['A', 'C', 'G', 'T', 'N', '-']

    # Define the header for the output TSV file
    matrix_header = ["Gene", "Position", "Reference"] + bps

    # Open the output file
    with open(outfile, 'w', encoding='utf-8', newline='\n') as fout:
        fout.write("\t".join(matrix_header) + "\n")

        indices = []

        # Find indices of comment lines
        for index, line in enumerate(data):
            if line.startswith("#"):
                indices.append(index)

        for i in range(len(indices) - 1):
            # The matrix file has columns in the order Ref, A, C, G, T, N, -(gaps)
            start = indices[i]
            end = indices[i + 1]

            data_gene = []
            data_new = data[start:end]
            gene_name = data_new[0][1:]
            pos = 1

            for j in data_new[1:]:
                if not j.startswith("-"):
                    data_write = [gene_name, str(pos)] + j.split("\t")
                    fout.write("\t".join(data_write) + "\n")
                    pos += 1


def getMutations(dataFile=None, genePos="mutPositions.csv"):
    mutsdict = dict()
    df = pd.read_csv(dataFile, header=0, sep="\t")
    # print(df.head())
    dfMutpos = pd.read_csv(genePos, header=0)
    # print(dfMutpos.head())
    geneNames = list(set(list(dfMutpos["GENE"])))
    print(f"Identifying mutations in {geneNames}")
    # getting gene wise data
    for geneName in geneNames:
        # for geneName in ["GraS"]:
        mutsdict[geneName] = []
        matrixdata = df[df["Gene"] == geneName]
        # print(matrixdata.tail())
        posIDS = dfMutpos[dfMutpos["GENE"] == geneName]["POS"]
        # print(posIDS)
        for varPosition in posIDS:
            codonStart = int(varPosition)*3 - 2
            codonStop = int(varPosition)*3
            subdf = matrixdata.iloc[codonStart-1:codonStop]
            # print(subdf)
            codons = []
            bpsList = ["A", "C", "G", "T", "N", "-"]
            subdfData = subdf.loc[:, bpsList]
            subdfData = subdfData.div(subdfData.sum(axis=1), axis=0)
            # print(subdfData)
            # codon = [subdf.loc[codonStart-1, "Reference"], subdf.loc[codonStart,
            #  "Reference"], subdf.loc[codonStart+1, "Reference"]]
            altcodon = []
            for index, row in subdfData.iterrows():

                # print(row)
                refBP = subdf.loc[index, "Reference"]
                refBPCov = subdfData.loc[index, refBP]
                if str(refBPCov) == "nan":
                    refBPCov = "No coverage"
                print(
                    f"The genome input at {index} has {refBP} with {refBPCov} coverage for gene {geneName}")
                altBP = subdfData.columns[row.apply(lambda x: x >= 0.1)]
                codons.append(altBP)

                for j in altBP:
                    if j != refBP:
                        # print(f"mutation detected was: {j} at position {index}")
                        altcodon.append(j)
                    else:
                        altcodon.append(refBP)
            # print(altcodon)
            # print(f"The detected codons are: {list(codons)}")
            # to generate possible codons from detected bases
            allCodons = [p for p in product(*codons)]
            altAAlist = []
            for sampleCodon in allCodons:
                altcodon = list(sampleCodon)
                altcodonStr = "".join(altcodon)
            # refCodon = "".join(codon)
                altAA = Seq(altcodonStr).translate(table=11)
                altAAlist.append(altAA)
                print(f"{altcodonStr} -> {altAA}")

            dataOutput = geneName + "_" + str(varPosition)
            mutsdict[geneName].append({varPosition: altAAlist})
    # print(mutsdict)
    return mutsdict


def mutsdict2df(mutsdict, genePos="mutPositions.csv", fout="predInput.csv", sampleName="XXX"):
    """To read a mutsdict variable and convert into a dataframe for prediction"""
    mutPosdf = pd.read_csv(genePos, header=0)
    nsample = 0
    emptydf = pd.DataFrame()
    emptydf.loc[nsample, "Sample"] = sampleName
    for gene in list(mutsdict.keys()):
        genesScreened = np.array(mutPosdf["GENE"] == gene)
        geneData = mutsdict[gene]
        for mutposition in geneData:
            # print(mutposition)
            posids = list(mutposition.keys())
            for posID in posids:
                allAAs = [str(i) for i in mutposition[posID]]
                allAAs = list(set(allAAs))
                print(
                    f"All amino acids detected at {posID} for gene {gene}: {allAAs}")
                for AAdetected in allAAs:
                    # AAdetected = mutposition[posID]
                    # print(AAdetected)
                    refAA = mutPosdf[(mutPosdf["GENE"] == gene) &
                                     (mutPosdf["POS"] == posID)]["REF"]
                    refAA = "".join(refAA.values)
                    altAA = mutPosdf[(mutPosdf["GENE"] == gene) &
                                     (mutPosdf["POS"] == posID)]["ALT"]
                # print(altAA)
                    altAA = "".join(altAA.values)

                    colname = gene + "_" + str(int(posID)) + "_" + altAA
                # print(
                    # f"the reference AA is {refAA} for gene {gene} at position {posID}")
                    if str(AAdetected) == refAA:
                        print(
                            f"no variant for gene {gene} at position {posID}")
                        dataval = 0
                    elif AAdetected == "":
                        print(
                            f"No read detected for gene {gene} at position {posID}: {refAA} -> None")
                        dataval = None
                    elif AAdetected == altAA:
                        print(
                            f"varaint detected for gene {gene} at position {posID}: {refAA} -> {AAdetected}")
                        dataval = 1
                    else:
                        print(
                            f" Novel varaint detected for gene {gene} at position {posID}: {refAA} | {altAA} -> {AAdetected}")
                        if altAA in allAAs:
                            dataval = str(1) + "_" + AAdetected
                        else:
                            dataval = AAdetected
                    # print(dataval)
                    emptydf.loc[nsample, colname] = dataval

    nsample += 1
    # print(emptydf.head())
    emptydf.to_csv(fout, index=False)
    print(f"dataframe created: {fout}")
