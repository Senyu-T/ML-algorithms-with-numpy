import numpy as np
import sys

def forward(x_i, y_i, alpha, beta):
    a = alpha.dot(x_i)
    z = np.append([1], sigmoid(a))
    b = beta.dot(z)
    y_h = np.exp(b)/sum(np.exp(b))
    y_h = softmax(b)
    J = -y_i.dot(np.log(y_h))
    return y_h, z, J

def backward(x_i, y_i, y_h, z, beta):
    gyh = -y_i/y_h
    yh = np.array([y_h])
    gb = gyh.dot(np.diag(y_h) - yh.T.dot(yh))
    gb = np.array([gb])
    z = np.array([z])
    gbeta = gb.T.dot(z)
    gz = beta.T.dot(gb.T)
    ga = gz.T * z * (1 - z)
    ga = ga.T
    x_i = np.array([x_i])
    galpha = ga.dot(x_i)[1:, :]
    return galpha, gbeta

def predict(y_h):
    currentMax = 0
    maxIndex = 0
    for i in range(len(y_h)):
        if (y_h[i] > currentMax):
            currentMax = y_h[i]
            maxIndex = i
    return maxIndex

def sgd_onestep(x_i, y_i, l_rate, alpha, beta):
    y_h, z, j = forward(x_i, y_i, alpha, beta)
    galpha, gbeta = backward(x_i, y_i, y_h, z, beta)
    alpha = alpha - l_rate * galpha
    beta = beta - l_rate * gbeta
    return alpha, beta

def learn(trainF, testF, flag, D, num_epoche, l_rate):
    # N: number of instances
    # K: number of label classes
    # M: number of features
    # D: number of hidden layer units
    N, X, Y, M, K = parse(trainF)
    Nt, Xt, Yt, Mt, Kt = parse(testF)
    # assert(K == Kt and M == Mt)
    alpha, beta = initialize_para(M, K, D, flag)
    # the output string for metric_out
    m_out = ""

    for i in range(num_epoche):
        # object functions for train and test
        J = 0.0
        Jt = 0.0
        # labels for train and test
        L = []
        Lt = []
        # errors
        error = 0
        T_error = 0
        # learning process
        for j in range(N):
            alpha, beta = sgd_onestep(X[j], Y[j], l_rate, alpha, beta)
        # predicting process
        for j in range(N):
            y_h, z, o = forward(X[j], Y[j], alpha, beta)
            J += o
            # the last iteration
            if (i == num_epoche - 1):
                resTr = predict(y_h)
                L.append(resTr)
                if (Y[j][resTr] != 1):
                    error += 1

        for j in range(Nt):
            y_ht, zt, jt = forward(Xt[j], Yt[j], alpha, beta)
            Jt += jt
            if (i == num_epoche - 1):
                resTe = predict(y_ht)
                Lt.append(resTe)
                if (Yt[j][resTe] != 1):
                    T_error += 1
        J = J / N
        error = error / ((float)(N))
        Jt = Jt / Nt
        T_error = T_error /((float)(Nt))
        # building the output string
        m_out += "epoch=" 
        m_out += str(i)
        m_out += " crossentropy(train): "
        m_out += str(J)
        m_out += "\n"
        m_out += "epoch=" 
        m_out += str(i)
        m_out += " crossentropy(test): "
        m_out += str(Jt)
        m_out += "\n"
    m_out += "error(trian): "
    m_out += str(error)
    m_out += "\n"
    m_out += "error(test): "
    m_out += str(T_error)
    m_out += "\n"
    return  L, Lt, m_out

def initialize_para(M, K, D, flag):
    # random
    if (flag == 1):
        alpha = 0.2 * np.random.rand(D, M + 1) - 0.1
        beta = 0.2 * np.random.rand(K, D + 1) - 0.1
        # bias terms initially zero
        for i in range(D):
            alpha[i][0] = 0
        for j in range(K):
            beta[j][0] = 0
    # zeros
    else:
        alpha = np.zeros((D, M + 1))
        beta = np.zeros((K, D + 1))
    return alpha, beta

def parse(file):
    with open(file, "rt") as f:
        lines = f.read().splitlines()
        N = len(lines)
        Y = []
        X = []
        for i in range(N):
            line = lines[i]
            ele = line.split(",")
            # magic number, K = 10 by assumption
            y_i = np.zeros(10)
            y_i[int(ele[0])] = 1
            M = len(ele) -  1 #should be 128
            x_i = np.zeros(M + 1)
            # bias term
            x_i[0] = 1
            for j in range(1, len(ele)):
                x_i[j] = int(ele[j])
            X.append(x_i)
            Y.append(y_i)
        # K: number of label classes
        # assuming test, train have same K, M
        # K = len(set(y))
        K = 10
    return N, X, Y, M, K

def sigmoid(vec):
    return 1 / (1 + np.exp(-vec))

def softmax(vec):
    return np.exp(vec) / sum(np.exp(vec))

if __name__ == '__main__':
    trainIn = sys.argv[1]
    testIn = sys.argv[2]
    trainOut = sys.argv[3]
    testOut = sys.argv[4]
    metOut = sys.argv[5]
    numEpo = int(sys.argv[6])
    hiddenU = int(sys.argv[7])
    initFlag = int(sys.argv[8])
    learRate = float(sys.argv[9])
    Lt, Lv, m_out = learn(trainIn, testIn, initFlag, hiddenU, numEpo, learRate)
    t_s = ""
    for ele in Lt:
        t_s += str(ele)
        t_s += "\n"
    v_s = ""
    for ele in Lv:
        v_s += str(ele)
        v_s += "\n"

    with open(trainOut, "wt") as a:
        a.write(t_s)
    with open(testOut, "wt") as b:
        b.write(v_s)
    with open(metOut, "wt") as c:
        c.write(m_out)
