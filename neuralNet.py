#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn import datasets

class NeuralNet:
    def __init__(self):
        self.standard_data = None
        self.normal_data = None
        self.iris_data = datasets.load_iris()
        self.column_names = self.iris_data.feature_names + ['class']
        self.iris_raw = pd.DataFrame(data=np.c_[self.iris_data['data'], self.iris_data['target']], columns=self.column_names)

    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):

        x_raw = self.iris_raw.drop('class', axis=1)
        y_raw = self.iris_raw['class']

        label_encoder = LabelEncoder()
        y_trans = label_encoder.fit_transform(y_raw)
        x_standard = StandardScaler().fit_transform(x_raw)
        x_normal = MinMaxScaler().fit_transform(x_raw)

        self.standard_data = pd.DataFrame(x_standard, columns=self.column_names[:-1])
        self.standard_data['class'] = y_trans

        self.normal_data = pd.DataFrame(x_normal, columns=self.column_names[:-1])
        self.normal_data['class'] = y_trans

        return 0

    def train_evaluate(self):
        columns = len(self.standard_data.columns)
        X = self.standard_data.iloc[:, 0:(columns - 1)]
        y = self.standard_data.iloc[:, (columns-1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [500, 700] # also known as epochs
        num_hidden_layers = [2, 3]
        results = []
        plt.figure(figsize=(14, 10))

        for activation in activations:
            for rate in learning_rate:
                for max_iter in max_iterations:
                    for layers in num_hidden_layers:
                        hidden_layer_sizes = tuple([10] * layers)
                        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                              learning_rate_init=rate, max_iter=max_iter, random_state=42)
                        model.fit(X_train, y_train)

                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)

                        train_acc = accuracy_score(y_train, y_train_pred)
                        test_acc = accuracy_score(y_test, y_test_pred)
                        train_loss = log_loss(y_train, model.predict_proba(X_train))
                        test_loss = log_loss(y_test, model.predict_proba(X_test))

                        results.append({
                            'activation': activation,
                            'learning_rate': rate,
                            'max_iterations': max_iter,
                            'num_hidden_layers': layers,
                            'train_acc': train_acc,
                            'test_acc': test_acc,
                            'train_loss': train_loss,
                            'test_loss': test_loss
                        })

                        plt.plot(model.loss_curve_, label=f'{activation}, rate={rate}, iter={max_iter}, layers={layers}')

        plt.title('Loss curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        results = pd.DataFrame(results)
        print(results)
        markdown_table = results.to_markdown(index=False)
        #print(markdown_table)

        return results

if __name__ == "__main__":
    neural_network = NeuralNet()
    neural_network.preprocess()
    neural_network.train_evaluate()
