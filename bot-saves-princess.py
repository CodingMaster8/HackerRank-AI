
# Princess Peach is trapped in one of the four corners of a square grid.
# You are in the center of the grid and can move one step at a time in
# any of the four directions. Can you rescue the princess?

# Link: https://www.hackerrank.com/challenges/saveprincess

def displayPathtoPrincess(n, grid):
    # print all the moves here
    bot = []
    princess = []
    not_found = True

    for position, i in enumerate(grid):
        if "p" in i:
            princess = [i.index("p"), position]
        if "m" in i:
            bot = [i.index("m"), position]

    while (not_found):
        if bot[0] < princess[0]:
            bot[0] = bot[0] + 1
            print("RIGHT")

        elif bot[0] > princess[0]:
            bot[0] = bot[0] - 1
            print("LEFT")

        if bot[1] < princess[1]:
            bot[1] = bot[1] + 1
            print("DOWN")

        elif bot[1] > princess[1]:
            bot[1] = bot[1] - 1
            print("UP")

        if bot[0] == princess[0] and bot[1] == princess[1]:
            not_found = False


m = int(input())
grid = []
for i in range(0, m):
    grid.append(input().strip())

displayPathtoPrincess(m, grid)