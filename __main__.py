import operator
from Question import Question
from Node import Node
import pandas as pd
import numpy as np

class Tree:
    def __init__(self):
        self.root = None
        self.table = None
        self.trained = False
        self.target = ""
        self.max_depth = 20

    def train(self, table, target):
        self.target = target
        self.table = table
        self.root = self.generate_tree(self.table, 0)

    def generate_tree(self, table, depth):
        node = Node()
        node.table = table
        best_q, best_gain = self.get_best_question(table)
        if self.get_entropy(table) == 0 or depth == self.max_depth or best_gain == 0:
            node.ratios = self.get_fractions(table)
            return node
        t, f = self.partition(table, best_q)
        node.question = best_q

        print("Best Question:", best_q)

        node.true_child = self.generate_tree(t, depth + 1)
        node.false_child = self.generate_tree(f, depth + 1)
        return node

    def get_best_question(self, table):
        questions = self.get_all_questions(table)
        ranked_questions = [(q, self.get_information_gain(table, q)) for q in questions]
        ranked_questions.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
        return ranked_questions[0]

    def get_information_gain(self, table, question):
        table_len = len(table)
        dEnt = self.get_entropy(table)
        t, f = self.partition(table, question)
        tEnt = self.get_entropy(t)
        fEnt = self.get_entropy(f)
        gain = dEnt - ((len(t) / table_len) * tEnt + (len(f) / table_len) * fEnt)
        return gain

    def get_entropy(self, table):
        fractions = self.get_fractions(table)
        fractions *= np.log2(fractions)
        return fractions.sum() * -1

    def get_all_questions(self, table):
        questions = []
        for col in table.columns:
            if col != self.target:
                for val in table[col].unique():
                    if np.issubdtype(table[col].dtype, np.number):
                        if (val != table[col].min() and val != table[col].max()):
                            questions.append(Question(operator.lt, col, val))
                    else:
                        questions.append(Question(operator.eq, col, val))
        return questions

    def get_fractions(self, table):
        idx = pd.Index(table[self.target])
        ret_val = idx.value_counts(normalize=True)
        return ret_val

    def partition(self, table, question):
        true_table = [question.eval(row) for index, row in table.iterrows()]
        false_table = np.invert(true_table)
        return table[true_table], table[false_table]

    def predict(self, row):
        node = self.root
        while node.question is not None:
            node = node.decide(row)
        return node.ratios.idxmax()

    def predict_table(self, tab):
        targets = [self.predict(row) for index, row in tab.iterrows()]
        tab[self.target] = targets
        return tab


def clean(tab):
    drop_cols = ["Name", "Ticket", "PassengerId", "Cabin"]
    tab = tab.drop(drop_cols, axis=1)
    tab = tab.fillna(0)
    tab = tab.astype({'Fare': 'int32', 'Age': 'int32'})
    return tab


df_train = pd.read_csv("data/train.csv")
print("read data")

df_testing = pd.read_csv("data/test.csv")
print("read testing data")

df_train = clean(df_train)
df_testing = clean(df_testing)

print("dropped cols")
#X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)

tree = Tree()
print("made tree object")
tree.train(df_train, "Survived")
print("trained tree")

y_pred = tree.predict_table(df_testing)
print(y_pred)
print("predicted")

'''
t,f = tree.partition(q)

print(t)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f)
'''
