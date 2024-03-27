def nextMove(n, r, c, grid):
    bot = [r, c]
    princess = []

    for position, i in enumerate(grid):
        if "p" in i:
            princess = [position, i.index("p")]

    if bot[0] < princess[0]:
        return "DOWN"

    elif bot[0] > princess[0]:
        return "UP"

    elif bot[0] == princess[0]:
        if bot[1] < princess[1]:
            return "RIGHT"

        elif bot[1] > princess[1]:
            return "LEFT"



n = int(input())
r, c = [int(i) for i in input().strip().split()]
grid = []
for i in range(0, n):
    grid.append(input())

print(nextMove(n, r, c, grid))