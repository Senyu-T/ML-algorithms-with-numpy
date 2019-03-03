import sys
import math

# Implementation based on binary assumption
class tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        # the index of the attribute it split
        self.val = None
        self.labels = None
        self.lPath = None
        self.rPath = None
        self.pred = None

def main(in_train, in_test, max_depth, out_train, out_test, out_met):
    with open(in_train, "rt") as trainIn:
        trainTexts = trainIn.read()
        lines = trainTexts.splitlines()
        attributes = lines[0].split(",")
        # -1 for excluding final label
        attCount = len(attributes) - 1
        classes = attributes[-1]
        incidences = lines[1:]
        datas = []
        for ele in incidences:
            datas.append(ele.split(","))
        # read the data
        labels = {}
        count = 0
        for ele in datas:
            count += 1
            label = ele[-1]
            if label not in labels:
                labels[label] = 1
            else:
                labels[label] += 1
        classes = list(labels.keys())
        # build the very first tree
        root = buildRoot(datas, classes)
        print(getStrLabels(labels))
        # begin learning
        dt = splitTree(max_depth, max_depth, datas, attributes, classes)
        # yield output
        errTrain, res = predict(dt, datas)
        outputTrain = ""
        for predVal in res:
            outputTrain = outputTrain + predVal + '\n'

    with open(in_test, "rt") as testIn:
        testTexts = testIn.read()
        lines = testTexts.splitlines()
        incidences = lines[1:]
        testDatas = []
        for ele in incidences:
            testDatas.append(ele.split(","))
        errTest, res =predict(dt, testDatas)
        outputTest = ""
        for predVal in res:
            outputTest = outputTest + predVal + '\n'

    with open(out_train, "wt") as trainOut:
        trainOut.write(outputTrain)
    with open(out_test, "wt") as testOut:
        testOut.write(outputTest)
    with open(out_met, "wt") as metOut:
        met = "error(train): " + str(errTrain) + "\n" + \
              "error(test): " + str(errTest) + "\n"
        metOut.write(met)

'''
splictTree: given current depth and max depth, decide if continue splitting
            or not, then generate the left and right subtrees if needed.
REQUIRES: current_d <= max_d && current_d >= 0
'''
def splitTree(current_d, max_d, data, attributes, classes):
    depth = max_d - current_d + 1
    labels, insCount = getLabel(data, classes)
    maxMI, bestAtt = getMInfo(data, classes)
    # depth >= maxDepth, mutual info <= 0, stop split
    # can't handle max_d = 0. save for later
    if current_d <= 0 or maxMI <= 0.0:
        end = tree()
        end.pred = getMajorVote(labels)
        return end

    t = tree()
    t.val = bestAtt
    strAtt = attributes[bestAtt]
    attDict, countDict = getSubData(bestAtt, data, classes)
    # Assuming every attributes are binary
    paths = list(attDict.keys())
    lPath = paths[0]
    rPath = paths[1]
    t.lPath = lPath
    t.rPath = rPath
    lLable = attDict[lPath]
    rLable = attDict[rPath]
    strLeft = '| ' * depth + strAtt + ' = ' + lPath + \
            ': ' + getStrLabels(lLable)
    strRight = '| ' * depth + strAtt + ' = ' + rPath + \
            ': ' + getStrLabels(rLable)
    lData = []
    rData = []
    for instances in data:
        if instances[bestAtt] == lPath:
            lData.append(instances)
        else:
            rData.append(instances)
    print(strLeft)
    t.left = splitTree(current_d - 1, max_d, lData, attributes, classes)
    print(strRight)
    t.right = splitTree(current_d - 1, max_d, rData, attributes, classes)
    return t


'''
buildRoot: given the 2D data, return the root node
           which consist only by the label count
'''
def buildRoot(data, classes):
    labels, insCount = getLabel(data, classes)
    root = tree()
    root.insCount = insCount
    root.labels = labels
    return root


'''
predict: given a tree and instance datas, return the predicted
         labels of the instances, along with the error rate
'''
def predict(tree, data):
    result = []
    count = 0
    error = 0.0
    attCount = len(data[0]) - 1
    for instance in data:
        count += 1
        origTree = tree
        while origTree.lPath != None:
            attr = origTree.val
            if instance[attr] == origTree.lPath:
                origTree = origTree.left
            else:
                origTree = origTree.right
        result.append(origTree.pred)
    for i in range(count):
        trueLabel = data[i][attCount]
        predLabel = result[i]
        if trueLabel != predLabel:
            error += 1.0
    return error / count, result

# helper functions
# labels is the dict that stores count of the instances of each class
# insCount is the total number of instances
def getEntropy(labels, insCount):
    entropy = 0.0
    for ele in labels:
        prob = labels[ele] / (float)(insCount)
        if prob != 0:
            entropy -= prob * (math.log(prob, 2.0))
    return entropy

# return the major vote given a label dictionary
def getMajorVote(labels):
    majorVote = 0
    majorClass = ""
    for ele in labels:
        if labels[ele] > majorVote:
            majorVote = labels[ele]
            majorClass = ele
    return majorClass

# return the label dictionary and total ins count given the 2d data
def getLabel(data, classes):
    labels = {}
    labels[classes[0]] = 0
    labels[classes[1]] = 0
    count = 0
    for ele in data:
        count += 1
        label = ele[-1]
        labels[label] += 1
    return (labels, count)

# compute the mutual info, return the best attribute to split
# node is a tree, data is the info 2D list
def getMInfo(data, classes):
    label, insCount = getLabel(data, classes)
    hY = getEntropy(label, insCount)
    maxMI = -100.0
    bestAttr = -1
    attCount = len(data[0]) - 1
    for i in range(attCount):
        attDict, countDict = getSubData(i, data, classes)
        gain = hY
        for atts in attDict:
            countX = countDict[atts]
            probX = (float)(countX) / (float)(insCount)
            gain -= probX * getEntropy(attDict[atts], countX)
        if gain > maxMI:
            maxMI = gain
            bestAttr = i
    return maxMI, bestAttr

# return info given the attribute
def getSubData(attributeIndex, data, classes):
    attDict = {}
    countDict = {}
    attCount = len(data[0]) - 1
    for instance in data:
        thisAtt = instance[attributeIndex]
        thisLab = instance[attCount]
        if thisAtt not in attDict:
            attDict[thisAtt] = {}
            attDict[thisAtt][classes[0]] = 0
            attDict[thisAtt][classes[1]] = 0
            attDict[thisAtt][thisLab] = 1
            countDict[thisAtt] = 1
        else:
            countDict[thisAtt] += 1
            if thisLab not in attDict[thisAtt]:
                attDict[thisAtt][thisLab] = 1
            else:
                attDict[thisAtt][thisLab] += 1
    return attDict, countDict

# return the str counting labels
# labels is the dict that stores count of the instances of each class
def getStrLabels(labels):
    res = "["
    for ele in labels:
        res = res + str(labels[ele]) + " " + ele + " /"
    res = res[:-2]
    res += "]"
    return res


if __name__ == '__main__':
    in_train = sys.argv[1]
    in_test = sys.argv[2]
    max_depth = int(sys.argv[3])
    out_train = sys.argv[4]
    out_test = sys.argv[5]
    out_met = sys.argv[6]
    main(in_train, in_test, max_depth, out_train, out_test, out_met)
