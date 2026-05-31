deck_of_suits_in_a_cards = ["Hearts", "Diamonds", "Clubs", "Spades"]
deck_of_cards = ["Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King"]

cards = []
hearts_found = 0
for x in deck_of_suits_in_a_cards:
    for y in deck_of_cards:
        if x == "Hearts":
            hearts_found += 1
        array_of_cards = y + " of " + x
        cards.append(array_of_cards)
print("Total number of Hearts found in cards: ", hearts_found)

total_cards = len(cards)
print("Total number of cards in the deck: ", total_cards)



probability = hearts_found / total_cards * 100
print("Probabilty of getting 12 hearts in a deck cards is: ", probability, "%", " 0.25 Or 13/52" )

for i in range(total_cards):
    print(i, ":", cards[i])

choice = int(input("Enter a number from A(1)-52: "))



chosen_card = cards[choice]

if "Hearts" in chosen_card:
    print("You choose a cards of Hearts")
else:
    print("Not a heart")