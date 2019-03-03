import sys
import math

def getEntropy(labels, totalInstance):
    entropy = 0.0
    for ele in labels:
        prob = labels[ele] / totalInstance
        entropy -= prob * (math.log(prob, 2.0))
    return entropy

def getMajorVote(labels):
    majorVote = 0.0
    for ele in labels:
        if labels[ele] > majorVote:
            majorVote = labels[ele]
    return majorVote

def main(in_file, out_file):
    labels = {}
    totalInstance = 0
    error = 0.0
    output = ""
    with open(in_file, "rt") as inFile:
        texts = inFile.read()
        instances = (texts.splitlines())[1:]
        for ele in instances:
            totalInstance += 1
            attributes = ele.split(",")
            label = attributes[-1]
            if label not in labels:
                labels[label] = 1.0
            else:
                labels[label] += 1.0
    entropy = getEntropy(labels, totalInstance)
    majorVote = getMajorVote(labels)
    error = 1 - majorVote / totalInstance
    output = "entropy: " + str(entropy) + "\n" + "error: " + str(error)

    with open(out_file, "wt") as outFile:
        outFile.write(output)

if __name__ == '__main__':
    inF = sys.argv[1]
    outF = sys.argv[2]
    main(inF, outF)



