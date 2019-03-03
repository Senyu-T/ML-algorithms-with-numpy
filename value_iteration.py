import sys

# obstacle as -1, available state as 0, goal as 1
def getStates(maze_in):
    with open(maze_in, "rt") as mazeF:
        lines = (mazeF.read()).splitlines()
        states = []
        maze = []
        row = 0
        for line in lines:
            thisRow = []
            col = 0
            for char in line:
                if char == '*':
                    thisRow.append(-1)
                elif char == '.':
                    thisRow.append(0)
                    states.append((row, col))
                else:
                    thisRow.append(1)
                    states.append((row, col))
                col += 1
            maze.append(thisRow)
            row += 1
    start = (row - 1, 0)
    goal = (0, col - 1)
    bound = (row - 1, col - 1)
    return maze, start, goal, bound, states

def takeAction(maze, bound, state, action):
    cRow, cCol = state
    assert(maze[cRow][cCol] != -1)
    # move west
    if action == 0:
        if cCol == 0 or maze[cRow][cCol - 1] == -1: 
            return state
        return (cRow, cCol - 1)
    # move north
    if action == 1:
        if cRow == 0 or maze[cRow - 1][cCol] == -1:
            return state
        return (cRow - 1, cCol)
    # move east
    if action == 2:
        if cCol == bound[1] or maze[cRow][cCol + 1] == -1:
            return state
        return (cRow, cCol + 1)
    # move south
    if action == 3:
        if cRow == bound[0] or maze[cRow + 1][cCol] == -1:
            return state
        return (cRow + 1, cCol)

def isTerminal(state, goal):
    if state == goal:
        return True
    return False

def buildV(states):
    V = dict()
    for state in states:
        V[state] = 0
    return V

def buildQ(states, actions):
    Q = dict() 
    for state in states:
        thisState = dict()
        for action in actions:
            thisState[action] = 0.0
        Q[state] = thisState
    return Q

def getQVal(Value, Q, state, action, gamma, maze, bound):
    reward = -1
    nxtState = takeAction(maze, bound, state, action)
    QVal = Value[nxtState] * gamma + reward
    Q[state][action] = QVal 
    return QVal 

# requires: !isTerminal(state)
def getAction(Value, Q, state, actions, gamma, maze, bound):
    curMaxAct = actions[0]
    curMaxVal = getQVal(Value, Q, state, curMaxAct, gamma, maze, bound)
    for action in actions:
        curVal = getQVal(Value, Q, state, action, gamma, maze, bound)
        if curMaxVal <= curVal:
            curMaxAct = action
            curMaxVal = curVal
    return curMaxAct, curMaxVal

def valueIter(iterations, states, maze, gamma, bound, start, goal):
    actions = [0, 1, 2, 3]
    V = buildV(states)
    Q = buildQ(states, actions)
    for i in range(iterations):
        thisValue = buildV(states)
        for state in states:
            if not isTerminal(state, goal):
                bestAction, thisQ = getAction(V, Q, state, \
                        actions, gamma, maze, bound)
                thisValue[state] = thisQ
        V = thisValue
    return V, Q

# the last arg is for sequential output for policy
def output(Value, Q, vf, qf, pf, states):
    with open(vf, "wt") as v_f:
        for state in Value:
            toWrite = str(state[0]) + ' ' + str(state[1]) + ' '\
                      + str(Value[state]) + '\n'
            v_f.write(toWrite)
    with open(qf, "wt") as q_f:
        for state in Q:
            for action in Q[state]:
                toWrite = str(state[0]) + ' ' + str(state[1]) + ' '\
                      + str(action) + ' ' + str(Q[state][action]) + '\n'
                q_f.write(toWrite)
    with open(pf, "wt") as p_f:
        for state in states:
            actions = Q[state]
            a = 0
            curMaxVal = actions[0]
            for act in actions:
                if Q[state][act] > curMaxVal:
                    curMaxVal = Q[state][act]
                    a = act
            toWrite = str(state[0]) + ' ' + str(state[1]) + ' '\
                    + str(float(a)) + '\n'
            p_f.write(toWrite)

if __name__ == "__main__":
    m_in = sys.argv[1]
    v_f = sys.argv[2]
    q_v_f = sys.argv[3]
    p_f = sys.argv[4]
    num_e = (int)(sys.argv[5])
    discount = (float)(sys.argv[6])
    maze, start, goal, bound, states = getStates(m_in)
    Value, Q = valueIter(num_e + 1, states, maze, discount, bound, start, goal)
    output(Value, Q, v_f, q_v_f, p_f, states)
