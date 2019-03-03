import sys, random
import environment

def buildQ(states, actions):
    Q = dict() 
    for state in states:
        thisState = dict()
        for action in actions:
            thisState[action] = 0.0
        Q[state] = thisState
    return Q

def getVal(E, Q, state):
    actions = E.actions
    maxVal = Q[state][actions[0]]
    for action in actions:
        if Q[state][action] > maxVal:
            maxVal = Q[state][action]
    return maxVal

def getAct(E, Q, state):
    actions = E.actions
    bestAct = actions[0]
    maxVal = Q[state][bestAct]
    for action in actions:
        if Q[state][action] > maxVal:
            maxVal = Q[state][action]
            bestAct = action
    return bestAct

def getAction(E, Q, state, epi):
    coin = random.random()
    if coin < epi:
        return random.randint(0, 3)
    return getAct(E, Q, state)

def update(E, Q, state, action, nxtState, reward, dis, l_r):
    Q[state][action] = (1 - l_r) * Q[state][action] + (dis\
            * getVal(E, Q, nxtState) + reward) * l_r 

def QIter(E, iterations, maxELen, l_r, dis, epi):
    Q = buildQ(E.states, E.actions)
    for i in range(iterations):
        E.reset()
        eLen = 0
        while (eLen < maxELen):
            act = getAction(E, Q, E.pos, epi)
            tmpPos = E.pos
            nxtS, r, terFlag = E.step(act)
            update(E, Q, tmpPos, act, nxtS, r, dis, l_r)
            eLen += 1
            if terFlag == 1: break
    return Q

def outA(Q, state):
    actions = Q[state]
    a = 0
    curMaxVal = actions[0]
    for act in actions:
        if Q[state][act] > curMaxVal:
            curMaxVal = Q[state][act]
            a = act
    return a

def output(Q, vf, qf, pf, E):
    with open(vf, "wt") as v_f:
        for state in E.states:
            a = outA(Q, state)
            toWrite = str(state[0]) + ' ' + str(state[1]) + ' '\
                      + str(Q[state][a]) + '\n'
            v_f.write(toWrite)
    with open(qf, "wt") as q_f:
        for state in Q:
            for action in Q[state]:
                toWrite = str(state[0]) + ' ' + str(state[1]) + ' '\
                      + str(action) + ' ' + str(Q[state][action]) + '\n'
                q_f.write(toWrite)
    with open(pf, "wt") as p_f:
        for state in E.states:
            a = outA(Q, state)
            toWrite = str(state[0]) + ' ' + str(state[1]) + ' '\
                    + str(float(a)) + '\n'
            p_f.write(toWrite)

if __name__ == "__main__":
    m_in = sys.argv[1]
    v_out = sys.argv[2]
    q_out = sys.argv[3]
    p_out = sys.argv[4]
    n_iter = (int)(sys.argv[5])
    max_e = (int)(sys.argv[6])
    l_r = (float)(sys.argv[7])
    dis = (float)(sys.argv[8])
    epi = (float)(sys.argv[9])
    E = environment.Environment(m_in)
    Q = QIter(E, n_iter + 1, max_e, l_r, dis, epi) 
    output(Q, v_out, q_out, p_out, E)
