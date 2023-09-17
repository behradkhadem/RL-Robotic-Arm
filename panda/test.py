import random

class LeitnerSystem:

    def __init__(self, items):
        self.items = items
        self.levels = 5
        self.current_level = 1

    def recommend(self):
        if self.current_level == 5:
            return random.choice(self.items)
        else:
            return self.items[self.current_level - 1]

    def study(self, item):
        if self.current_level == 5:
            self.current_level = 1
        else:
            self.current_level += 1

def main():
    items = ["item1", "item2", "item3", "item4", "item5"]
    ls = LeitnerSystem(items)
    print(ls.recommend())
    ls.study("item1")
    print(ls.recommend())

if __name__ == "__main__":
    main()