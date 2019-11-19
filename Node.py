class Node:
    def __init__(self):
        self.false_child = None
        self.true_child = None
        self.question = None
        self.ratios = []

    def decide(self, row):
        if self.question:
            res = self.question.eval(row)
            if res:
                return self.true_child
            else:
                return self.false_child
