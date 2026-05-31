
bag = ["Red"] * 5 + ["Blue"] * 3 + ["Green"] * 2


red = 0
blue = 0
green = 0


for i in range(500):

    choice = int(input("Pick a number from 0-9 (0 represents [1] and 9 represents [10]): "))

    picked = bag[choice]

    print("Picked:", picked)

    if picked == "Red":
        red += 1

    elif picked == "Blue":
        blue += 1

    else:
        green += 1


print("Red Probability:", red / 500 * 100, "%")
print("Blue Probability:", blue / 500 * 100, "%")
print("Green Probability:", green / 500 * 100, "%")