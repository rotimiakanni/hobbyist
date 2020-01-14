import random

alpha = ["a", "b", "c", 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
         'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

while True:
    alpha2 = random.choice(alpha)
    user = input("Enter a word that starts with " + alpha2)
    if (alpha2 == user[0]):
        print("CORRECT")
    else:
        print("WRONG")
