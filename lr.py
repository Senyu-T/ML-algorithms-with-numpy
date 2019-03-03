import numpy as np
import sys

def read_dict(dic_file):
    voc = {}
    with open(dic_file, "rt") as dic:
        in_string = dic.read()
        lines = in_string.splitlines()
        for line in lines:
            l = line.split()
            voc[l[0]] = l[1]
    return voc

def sgd_onestep(theta, l_rate, K, x_i, y_i):
    mul = np.exp(theta.dot(x_i))
    scalar = y_i - mul / (1 + mul)
    theta = theta + l_rate * scalar *x_i
    return theta

def get_xyk(in_file, K):
    X = []
    y = []
    with open(in_file, "rt") as f_in:
        in_string = f_in.read()
        instances = in_string.splitlines()
        for ins in instances:
            words = ins.split("\t")
            y.append((int)(words[0]))
            x = np.zeros(K + 1)
            x[0] = 1
            for word in words[1::]:
                key = (int)((word.split(":"))[0])
                x[key + 1] = 1
            X.append(x)
    N = len(X)
    return X, y, N

def learn(X, y, K, N, num_epoche):
    theta = np.zeros(K + 1)
    l_rate = 0.1
    epoche = 0
    while epoche < num_epoche:
        for i in range(N):
            theta = sgd_onestep(theta, l_rate, K, X[i], y[i])
        epoche += 1
    return theta

def predict(theta, X, y, N, l_out):
    error = 0.0
    with open(l_out, "wt") as lab_out:
        for i in range(len(X)):
            sigmoid = np.exp(theta.dot(X[i]))
            p = sigmoid / (1 + sigmoid)
            if p >= 0.5:
                lab_out.write("1\n")
                if y[i] != 1:
                    error += 1
            else:
                lab_out.write("0\n")
                if y[i] != 0:
                    error += 1
    return (float)(error/N)

if __name__ == "__main__":
    f_tr = sys.argv[1]
    f_va = sys.argv[2]
    f_te = sys.argv[3]
    dic = read_dict(sys.argv[4])
    K = len(dic)
    o_tr = sys.argv[5]
    o_te = sys.argv[6]
    o_met = sys.argv[7]
    num_epoch = (int)(sys.argv[8])
    test_X, test_Y, test_N = get_xyk(f_te, K)
    train_X, train_Y, train_N = get_xyk(f_tr, K)
    theta = learn(train_X, train_Y, K, train_N, num_epoch)
    tr_err = predict(theta, train_X, train_Y, train_N, o_tr)
    te_err = predict(theta, test_X, test_Y, test_N, o_te)
    with open(o_met, "wt") as m:
        m.write("error(train): " + (str)(tr_err) + "\n")
        m.write("error(test): " + (str)(te_err) + "\n")
