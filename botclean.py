#Bot designed to clean the nearest dirt


import math


def next_dirt(posr, posc, dirties):
    nearest_dirt = []

    # get euclidean distance of each dirt
    for i in range(len(dirties)):
        distance = math.sqrt(((dirties[i][0] - posr) ** 2) + ((dirties[i][1] - posc) ** 2))
        nearest_dirt.append(distance)

    sort = [x for (y, x) in sorted(zip(nearest_dirt, dirties))]
    return (sort)


def next_move(posr, posc, board):
    dirties = []
    bot = [posr, posc]

    # Get all the dirties positions
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == "d":
                dirties.append([i, j])

    next = next_dirt(posr, posc, dirties)

    # like the princess method approach
    if next[0][1] < posc:
        print("LEFT")
    elif next[0][1] > posc:
        print("RIGHT")
    elif next[0][0] < posr:
        print("UP")
    elif next[0][0] > posr:
        print("DOWN")
    else:
        print("CLEAN")


# Set the data
if __name__ == "__main__":
    pos = [int(i) for i in input().strip().split()]
    board = [[j for j in input().strip()] for i in range(5)]
    next_move(pos[0], pos[1], board)