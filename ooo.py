##x = 2
##
##while :
##    print('yes')

##
##f = 'apple'
##print(f[2])


##num = 2
##while num <= 10:
##    print(num)
##    num += 2
##    print('Goodbye!')

##num = 10
##print('Hello')
##while num >= 2:
##    print(num)
##    num -= 2

##for iteration in range(5):
##    count = 0
##    while True:
##        for letter in "hello, world":
##            count += 1
##        print("iteration " + str(iteration) + "; count is: " + str(count))

##
###collecting varA and varB from user
###this is not necessary in the code though
##varA = int(input("enter an int: "))
##varB = int(input("enter an int: "))
##
###checks to ensure the bigger and smaller values
###are handled correctly
##if varA < varB:
##    varC = varA
##    varD = varB
##else:
##    varC = varB
##    varD = varA
##
###this block attempts to take the lower variable to
###the nearest multiple of 5
##while varC % 5 != 0:
##    varC += 1
##
###this block performs the check and prints
##while varC <= varD:
##    if varC <= varD:
##        print(varC)
##        varC += 5


# bigList is the list of lists
#
# char is the string character that the coordinate
# is to be found
def charCoor(bigList, char):
    #this line performs an iteration over the big list
    #the next for-loop performs an operation over the
    #characters in the smallists gotten from the big list iteration
    for smallList in range(len(bigList)):
        for cha in range(len(bigList[smallList])):
            if bigList[smallList][cha] == char:
                return(smallList, cha)

listA = [['1', '2', '3'],
         ['4', '5', '6'],
         ['7', '8', '9']]

print(charCoor(listA, '7'))



            
