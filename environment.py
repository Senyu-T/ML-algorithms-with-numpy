import sys
class Environment(object):
    # basically the same as in previous task
    def __init__(self, fileName):
        with open(fileName, "rt") as mazeF:
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
        self.maze = maze
        self.actions = [0, 1, 2, 3]
        self.states = states
        self.start = (row - 1, 0)
        self.goal = (0, col - 1)
        self.bound = (row - 1, col - 1)
        # current position
        self.pos = (row - 1, 0)

    def step(self, a):
        cRow, cCol = self.pos
        #assert(self.maze[cRow][cCol] != -1)
        if self.pos == self.goal:
            return self.pos, 0, 1
        # move west
        if a == 0:
            if cCol == 0 or self.maze[cRow][cCol - 1] == -1:
                return (self.pos, -1, 0)
            nxtState = (cRow, cCol - 1)
        # move north
        if a == 1:
            if cRow == 0 or self.maze[cRow - 1][cCol] == -1:
                return (self.pos, -1, 0)
            nxtState = (cRow - 1, cCol)
        # move east
        if a == 2:
            if cCol == self.bound[1] or self.maze[cRow][cCol + 1] == -1:
                return (self.pos, -1, 0)
            nxtState = (cRow, cCol + 1)
        # move south
        if a == 3:
            if cRow == self.bound[0] or self.maze[cRow + 1][cCol] == -1:
                return (self.pos, -1, 0)
            nxtState = (cRow + 1, cCol)
        self.pos = nxtState
        if nxtState == self.goal:
            return (nxtState, -1, 1)
        return (nxtState, -1, 0)

    def reset(self):
        self.pos = self.start

if __name__ == '__main__':
    m_in = sys.argv[1]
    out = sys.argv[2]
    act_sf = sys.argv[3]
    E = Environment(m_in)
    with open(act_sf, "rt") as acts:
        actions = acts.read()
        actions = (actions[:-1]).split(" ")
    with open(out, "wt") as of:
        for act in actions:
            a = (int)(act)
            nxt, r, isT = E.step(a)
            toWrite = str(nxt[0]) + " " + str(nxt[1]) + " " \
                    + str(r) + " " + str(isT) + "\n"
            of.write(toWrite)
