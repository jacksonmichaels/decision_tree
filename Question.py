import pandas as pd
import operator


class Question:
    def __init__(self, op, col, target):
        self.operator = op
        self.col = col
        self.target = target

    def eval(self, row):
        return self.operator.__call__(row[self.col], self.target)

    def __repr__(self):
        return "Is {} {} {}".format(self.col, self.operator, self.target)
