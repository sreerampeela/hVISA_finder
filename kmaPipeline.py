import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from itertools import permutations, product
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--readPrefix", help="Fastq file prefix (File should end in *_1.fastq.gz)")
parser.add_argument("--dbprefix", help="KMA db prefix")
parser.add_argument("--outPrefix", help="Output file prefix")
parser.add_argument("--kmaPath", help="kmaPath", default="/home/sreeram/kma")
parser.add_argument("--genePosFile", help="Gene position file in CSV")
parser.add_argument("--fastqPath", help="Fastq file path")
args = parser.parse_args()


def runKMA(readPrefix, dbprefix, outPrefix):
    kmaPath = args.kmaPath
    cmd = f"{kmaPath}/kma -ipe {args.fastqPath}/{readPrefix}_1.fastq.gz {args.fastqPath}/{readPrefix}_2.fastq.gz -t_db {dbprefix} -matrix -mp 20 -1t1 -o {outPrefix}"
    print(cmd)
    os.system(cmd)
    print("KMA run completed..Unzipping the matrix file")
    os.system(f"gunzip {outPrefix}.mat.gz")


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
        indices.append(len(data))

        print(f"total number of proteins: {len(indices)}")
        print(f"indices identified are: {indices}")
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


def getMutations(dataFile, genePos):
    mutsdict = dict()
    df = pd.read_csv(dataFile, header=0, sep="\t")

    dfMutpos = pd.read_csv(genePos, header=0)

    geneNames = list(set(list(dfMutpos["GENE"])))
    print(f"Identifying mutations in {geneNames}")
    # getting gene wise data
    for geneName in geneNames:

        mutsdict[geneName] = []
        matrixdata = df[df["Gene"] == geneName]
        print(matrixdata.shape)
        posIDS = dfMutpos[dfMutpos["GENE"] == geneName]["POS"]

        for varPosition in posIDS:
            codonStart = int(varPosition)*3 - 3
            codonStop = int(varPosition)*3
            subdf = matrixdata.iloc[codonStart:codonStop]
            print(subdf)
            codons = []
            bpsList = ["A", "C", "G", "T", "N", "-"]
            subdfData = subdf.loc[:, bpsList]
            subdfData = subdfData.div(subdfData.sum(axis=1), axis=0)
            altcodon = []
            for index, row in subdfData.iterrows():
                refBP = subdf.loc[index, "Reference"]
                refBPCov = subdfData.loc[index, refBP]
                if str(refBPCov) == "nan":
                    refBPCov = "No coverage"
                print(
                    f"The genome input at {index} has {refBP} with {refBPCov} coverage for gene {geneName}")
                altBP = subdfData.columns[row.apply(lambda x: x >= 0.05)]
                codons.append(altBP)

                for j in altBP:
                    if j != refBP:
                        altcodon.append(j)
                    else:
                        altcodon.append(refBP)

            # to generate possible codons from detected bases
            allCodons = [p for p in product(*codons)]
            altAAlist = []
            for sampleCodon in allCodons:
                altcodon = list(sampleCodon)
                altcodonStr = "".join(altcodon)

                if "-" not in altcodonStr:
                    altAA = Seq(altcodonStr).translate(table=11)
                else:
                    altAA = ""
                altAAlist.append(altAA)
                print(f"{altcodonStr} -> {altAA}")

            dataOutput = geneName + "_" + str(varPosition)
            mutsdict[geneName].append({varPosition: altAAlist})
    print(mutsdict)
    return mutsdict


def mutsdict2df(mutsdict, genePos, sampleName):
    """To read a mutsdict variable and convert into a dataframe for prediction"""

    output = []
    mutPosdf = pd.read_csv(genePos, header=0)
    nsample = 0
    geneNames = list(mutPosdf["GENE"])
    posIDS = list(mutPosdf["POS"])
    colnames = []
    for i in range(len(geneNames)):
        colid = "_".join([geneNames[i], str(int(posIDS[i]))])
        colnames.append(colid)
    emptydf = pd.DataFrame(columns=colnames)
    emptydf.loc[nsample, "Sample"] = sampleName
    print(emptydf.head())
    for gene in list(mutsdict.keys()):
        print(gene)
        geneData = mutsdict[gene]
        for mutposition in geneData:
            posids = list(mutposition.keys())
            for posID in posids:
                colidname = "_".join([gene, str(int(posID))])
                allAAs = [str(i) for i in mutposition[posID]]
                allAAs = list(set(allAAs))
                outstr1 = f"All amino acids detected at {posID} for gene {gene}: {allAAs}"
                refAA = mutPosdf[(mutPosdf["GENE"] == gene) &
                                 (mutPosdf["POS"] == posID)]["REF"]
                refAA = "".join(refAA.values)
                altAA = mutPosdf[(mutPosdf["GENE"] == gene) &
                                 (mutPosdf["POS"] == posID)]["ALT"]

                for AAdetected in allAAs:
                    if str(AAdetected) == refAA:
                        outstr2 = f"no variant for gene {gene} at position {posID}"
                        dataval = refAA
                    elif AAdetected == "":
                        outstr2 = f"No read detected for gene {gene} at position {posID}: {refAA} -> None"
                        dataval = None
                    elif AAdetected in altAA:
                        dataval = AAdetected
                        outstr2 = f"varaint detected for gene {gene} at position {posID}: {refAA} -> {AAdetected}."

                    else:
                        outstr2 = f" Novel varaint detected for gene {gene} at position {posID}: {refAA} | {altAA} -> {AAdetected}"
                        dataval = AAdetected
                    emptydf.loc[nsample, colidname] = dataval
                    output.append("\n".join([outstr1, outstr2]))
    return emptydf, output


def writedf(emptydf, output):
    fout = f"{args.outPrefix}" + "_df.csv"
    emptydf.to_csv(fout, index=False)
    print(emptydf.head())
    print(f"dataframe created: {fout}")
    logFile = f"{args.outPrefix}" + ".log"
    with open(logFile, 'w', encoding='utf-8', newline="\n") as logfile:
        for out in output:
            logfile.write(out + "\n")


def main():
    runKMA(readPrefix=args.readPrefix,
           dbprefix=args.dbprefix, outPrefix=args.outPrefix)
    read_matrix(matrix_file=f"{args.outPrefix}.mat",
                outfile=f"{args.outPrefix}_cleaned.tsv")
    mutsdict = getMutations(
        dataFile=f"{args.outPrefix}_cleaned.tsv", genePos=args.genePosFile)
    dataDF, sampleOut = mutsdict2df(
        mutsdict=mutsdict, genePos=args.genePosFile, sampleName=args.readPrefix)

    writedf(emptydf=dataDF, output=sampleOut)


if __name__ == main():
    main()
