##xh = input("enter hr")
##try:
##    xv = int(input("enter rate"))
##    #xv = xv + 0
##    chk = xv - 40
##    if chk == 0:
##        print("yes")
##    else:
##        print("no")
##except:
##    print("this is not an integer")
##    

def findIndex(x, m):
    for item in x:
        if m in item:
            return (x.index(item), item.index(m))

def displayPathtoPrincess(grid):
##    grid = []
##    for x in range(n):
##        grid2.append(grid[(x*n):(x+1)*n])
    goal = findIndex(grid, 'p')
    location = findIndex(grid, 'm')
    moves = []
    while goal[0] != location[0]:
        if goal[0] < location[0]:
            location = (location[0]-1, location[1])
            moves.append('up')
        elif goal[0] > location[0]:
            location = (location[0]+1, location[1])
            moves.append('down')
    while goal[1] != location[1]:
        if goal[1] < location[1]:
            location = (location[0], location[1]-1)
            moves.append('left')
        elif goal[1] > location[1]:
            location = (location[0], location[1]+1)
            moves.append('right')
    return moves

grid = [['_', '_', '_', '-'],
        ['_', 'm', '_', '-'],
        ['p', '_', '_', '-']]

##grid = ['p', '-', '-', '-', '-',
##        '-', '-', '-', '-', '-',
##        '_', '-', '-', '-', '-',
##        '-', '-', '-', '-', '-',
##        '-', '-', '-',  'm', '-']

n = 5

print(displayPathtoPrincess(grid))

##def findIndex(x, m):
##    lis = []
##    for item in range(len(x)):
##        #print(item)
##        for char in range(len(x[item])):
##            if x[item][char] == m:
##                lis.append((item, char))
##    return lis
##
##e = [['-','-','d'],
##     ['-','d','d'],
##     ['-','-','d']]
##print(findIndex(e, 'd'))

##def findIndex(x, m):
##    lis = []
##    for item in range(len(x)):
##        for char in range(len(x[item])):
##            if x[item][char] == m:
##                lis.append((item, char))
##    return lis
##
##def nearest(dirtyList, cleanList, location):
##    for x in range(len(dirtyList)):
##        if dirtyList[x] in cleanList:
##            dirtyList.remove(dirtyList[x])
##    row_list = []
##    col_list = []
##    row_win = (0,0)
##    row_check = 10
##    col_win = (0,0)
##    col_check = 10
##    print(location)
##    for pos in dirtyList:
##        if pos[0] == location[0]:
##            row_list.append(pos)
##        elif pos[1] == location[1]:
##            col_list.append(pos)
##    for x in row_list:
##        if abs(location[1]-x[1]) < row_check:
##            row_check = abs(location[1]-x[1])
##            row_win = x
##    for x in col_list:
##        if abs(location[0]-x[0]) < col_check:
##            col_check = abs(location[0]-x[0])
##            col_win = x
##    print(row_list)
##    print(col_list)
##    print(dirtyList)
##    print(row_win)
##    print(col_win)
##    if abs(location[1] - row_win[1]) < abs(location[0] - col_win[0]):
##        if row_win[1] < location[1]:
##            print('LEFT')
##            if (location[0]-1, location[1]) in dirtyList:
##                cleanList.append((location[0]-1, location[1]))
##            location = (location[0]-1, location[1])
##        elif row_win[1] > location[1]:
##            print('RIGHT')
##            if (location[0]+1, location[1]) in dirtyList:
##                cleanList.append((location[0]+1, location[1]))
##            location = (location[0]+1, location[1])
##    elif abs(location[0] - col_win[0]) < abs(location[1] - row_win[1]):
##        if col_win[0] < location[0]:
##            print('UP')
##            if (location[0], location[1]-1) in dirtyList:
##                cleanList.append((location[0], location[1]-1))
##            location = (location[0], location[1]-1)
##        elif col_win[0] > location[0]:
##            print('DOWN')
##            if (location[0], location[1]+1) in dirtyList:
##                cleanList.append((location[0], location[1]+1))
##            location = (location[0], location[1]+1)
##    elif abs(location[0] - col_win[0]) == abs(location[1] - row_win[1]):
##        if row_win[1] < location[1]:
##            print('LEFT')
##            if (location[0]-1, location[1]) in dirtyList:
##                cleanList.append((location[0]-1, location[1]))
##            location = (location[0]-1, location[1])
##        elif row_win[1] > location[1]:
##            print('RIGHT')
##            if (location[0]+1, location[1]) in dirtyList:
##                cleanList.append((location[0]+1, location[1]))
##            location = (location[0]+1, location[1])
##        
##            
##        
##def next_move(posr, posc, board):
##    dirty_list = findIndex(board, 'd')
##    clean_list = []
##    position = (posr, posc)
##    pos_clone = position
##    nearest(dirty_list, position)
##
##
##board = [['b', '-', '-', 'd', '-'],
##         ['d', '-', '-', 'd', 'd'],
##         ['-', 'd', '-', 'd', '-'],
##         ['d', '-', '-', '-', '-'],
##         ['-', '-', '-', '-', '-']]
##
##posr, posc = findIndex(board, 'b')[0]
##
##print(next_move(posr, posc, board))
##
##
##
##
#############################################################
##################  STOCASTIC BOT CLEAN  ####################
#############################################################
##def findIndex(x, m):
##    lis = []
##    for item in range(len(x)):
##        for char in range(len(x[item])):
##            if x[item][char] == m:
##                lis.append((item, char))
##    return lis
##
##
##def next_move(posx, posy, dimx, dimy, board):
##    bot = (posx, posy)
##    dirtyList = findIndex(board, 'd')
##    dirt = (dimx, dimy)
##    thresh = dimx * dimy
##    for pos in dirtyList:
##        steps = abs(bot[0]-pos[0]) + abs(bot[1]-pos[1])
##        if steps < thresh:
##            thresh = steps
##            dirt = pos  
##    
##    if bot[0] == dirt[0]:
##        if bot[1] < dirt[1]:
##            print('RIGHT')
##        elif bot[1] == dirt[1]:
##            print('CLEAN')
##        else:
##            print('LEFT')
##    
##    elif bot[1] == dirt[1]:
##        if bot[0] < dirt[0]:
##            print('DOWN')
##        elif bot[0] == dirt[0]:
##            print('CLEAN')
##        else:
##            print('UP')
##            
##    elif abs(bot[0] - dirt[0]) > abs(bot[1] - dirt[1]):
##        if bot [1] < dirt[1]:
##            print('RIGHT')
##        else:
##            print('LEFT')
##            
##    elif abs(bot[0] - dirt[0]) == abs(bot[1] - dirt[1]):
##        if bot [1] < dirt[1]:
##            print('RIGHT')
##        else:
##            print('LEFT')
##    
##    elif abs(bot[0] - dirt[0]) < abs(bot[1] - dirt[1]):
##        if bot [0] < dirt[0]:
##            print('DOWN')
##        else:
##            print('UP')
##    
##    print("")
##
##if __name__ == "__main__":
##    pos = [int(i) for i in input().strip().split()]
##    dim = [int(i) for i in input().strip().split()]
##    board = [[j for j in input().strip()] for i in range(dim[0])]
##    next_move(pos[0], pos[1], dim[0], dim[1], board)
    

#############################################################
#############   BOT CLEAN PARTIALLY OBSERVABLE   ############
#############################################################

##floor = [['b','-','o','o','o'],
##         ['-','d','o','o','o'],
##         ['o','o','o','o','o'],
##         ['o','o','o','o','o'],
##         ['o','o','o','o','o']]
##
##def findIndex(x, m):
##    for item in x:
##        if m in item:
##            return (x.index(item), item.index(m))
##
##def next_move(posx, posy, board):
##    bot = (posx, posy)
##    dirt = findIndex(floor, 'd')
##    
##    if bot[0] == dirt[0]:
##        if bot[1] < dirt[1]:
##            print('RIGHT')
##        elif bot[1] == dirt[1]:
##            print('CLEAN')
##        else:
##            print('LEFT')
##    
##    elif bot[1] == dirt[1]:
##        if bot[0] < dirt[0]:
##            print('DOWN')
##        elif bot[0] == dirt[0]:
##            print('CLEAN')
##        else:
##            print('UP')
##            
##    elif abs(bot[0] - dirt[0]) > abs(bot[1] - dirt[1]):
##        if bot [1] < dirt[1]:
##            print('RIGHT')
##        else:
##            print('LEFT')
##            
##    elif abs(bot[0] - dirt[0]) == abs(bot[1] - dirt[1]):
##        if bot [1] < dirt[1]:
##            print('RIGHT')
##        else:
##            print('LEFT')
##    
##    elif abs(bot[0] - dirt[0]) < abs(bot[1] - dirt[1]):
##        if bot [0] < dirt[0]:
##            print('DOWN')
##        else:
##            print('UP')
##
##
##posx, posy = findIndex(floor, 'b')
##next_move(posx, posy, floor)

##import math
##def quadEqn(a,b,c):
##    x1 = -b + math.sqrt(((b**2)-4*a*c)/2*a)
##    x2 = -b - math.sqrt(((b**2)-4*a*c)/2*a)
##    return (x1, x2)
##
##print(quadEqn(1,-2,-3))


## pseudocode for bot saves princess ###
# step 1: find the bot
# iterate over the list and check if the required letter is in the list
# during iteration
# return a tuple containing the x, y coordinates of the desired letter
# where the x represents the row and the y rep the column
#

h = [[1,2,3],
     [2,3,4],
     [4,5,6],
     [5,6,7]]

import RPI.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
for x in range(3):
    GPIO.output(7,True)
    time.sleep(1)
    GPIO.output(7, False)
    GPIO.output(11,True)
    time.sleep(1)
    GPIO.output(11,False)
    GPIO.output(13,True)
    time.sleep(1)
    GPIO.output(13,Fals e)
    GPIO.output(15,True)
GPIO.cleanup()

import curses

#Get the , turn off echoing of keyboard to screen,
#turn on instant(no waiting) key response, and use special values
#for cursor keys
screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)

try:
    while True:
        char = screen.getch()
        if char == ord('q'):
            break
        elif char == curses.KEY_UP:
            print('up')
        elif char == curses.KEY_DOWN:
            print('down')
        elif char == curses.KEY_RIGHT:
            print('right')
        elif char == curses.KEY_LEFT:
            print('left')
        elif char == 10:
            print('stop')
            #all gpio false

finally:
    curses.nocbreak(); screen.keypad(0); curses.echo()
    curses.endwin()
    gpio.cleanup()



