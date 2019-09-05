import math
import random
import string
import csv
# import static as static
from tabulate import tabulate
import numpy as np
from ImageMaker import ImageMaker
from test_iris import *
import unittest

random.seed(0)


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)


# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - math.tanh(y) ** 2

#Main class for neural network
class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh + 1  # +1 for bias node
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # sum of inputs
        self.sh = [1.0] * self.nh
        self.so = [1.0] * self.no

        # normalization parameters
        self.max = [0.0] * (self.ni - 1)
        self.min = [10.0] * (self.ni - 1)

        # create weights
        self.wi = makeMatrix(self.nh, self.ni)
        self.wo = makeMatrix(self.no, self.nh)

        # difrences of weights (for drawing)
        self.ri = makeMatrix(self.nh, self.ni)
        self.ro = makeMatrix(self.no, self.nh)
        # set them to random values
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[j][i] = rand(-1.0, 1.0)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[k][j] = rand(-1.0, 1.0)

        # self.wi[0][0] = 11.5349
        # self.wi[1][0] = 96.8871
        # self.wi[2][0] = -0.6602
        # self.wi[3][0] = 58.5381
        # self.wi[4][0] = -0.5916
        # self.wi[0][1] = 2.3221
        # self.wi[1][1] = -52.9829
        # self.wi[2][1] = 3.9816
        # self.wi[3][1] = -37.0987
        # self.wi[4][1] = -47.8942
        # self.wi[0][2] = 13.7675
        # self.wi[1][2] = 37.0570
        # self.wi[2][2] = -5.6558
        # self.wi[3][2] = 47.6512
        # self.wi[4][2] = 77.1399
        # self.wi[0][3] = 3.5308
        # self.wi[1][3] = -40.7752
        # self.wi[2][3] = -10.4035
        # self.wi[3][3] = -39.7127
        # self.wi[4][3] = 158.3520
        # self.wi[0][4] = -13.3867
        # self.wi[1][4] = -42.9766
        # self.wi[2][4] = -6.8914
        # self.wi[3][4] = -31.6713
        # self.wi[4][4] = -92.5038
        # self.wo[0][0] = 0.0001
        # self.wo[0][1] = -3.1652
        # self.wo[0][2] = -0.5
        # self.wo[0][3] = 3.1652
        # self.wo[0][4] = 0.5
        # self.wo[0][5] = 2
        # self.max[0] = 7.9
        # self.max[1] = 4.4
        # self.max[2] = 6.9
        # self.max[3] = 2.5
        # self.min[0] = 4.3
        # self.min[1] = 2.0
        # self.min[2] = 1.0
        # self.min[3] = 0.1

    # calculating new outputs
    def update(self, inputs):
        inputs = self.normalize(inputs)
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh - 1):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[j][i]
            self.ah[j] = sigmoid(sum)
            self.sh[j] = sum

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + (self.ah[j] * self.wo[k][j])
            self.ao[k] = sum
        if self.ao[0] > 1000:
            print(self.ao[0])
        return self.ao[:]

    #changing weights and calculating error
    def backPropagate(self, targets, N):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[k][j]
            hidden_deltas[j] = dsigmoid(self.sh[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[k][j] = self.wo[k][j] + N * change
                self.ro[k][j] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[j][i] = self.wi[j][i] + N * change
                self.ri[j][i] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    # sorting data depends on predicted values
    def test(self, patterns):
        mb = makeMatrix(3, 3, 0.0)
        for p in patterns:
            # print(p[0], '->', self.update(p[0]), '==', round(self.update(p[0])[0]), '->', p[1])
            self.update(p[0])
            if (p[1][0] == 1) & (round(self.ao[0]) == 1):
                mb[0][0] = mb[0][0] + 1
            elif (p[1][0] == 1) & (round(self.ao[0]) == 2):
                mb[1][0] = mb[1][0] + 1
            elif (p[1][0] == 1) & (round(self.ao[0]) == 3):
                mb[2][0] = mb[2][0] + 1
            elif (p[1][0] == 2) & (round(self.ao[0]) == 1):
                mb[0][1] = mb[0][1] + 1
            elif (p[1][0] == 2) & (round(self.ao[0]) == 2):
                mb[1][1] = mb[1][1] + 1
            elif (p[1][0] == 2) & (round(self.ao[0]) == 3):
                mb[2][1] = mb[2][1] + 1
            elif (p[1][0] == 3) & (round(self.ao[0]) == 1):
                mb[0][2] = mb[0][2] + 1
            elif (p[1][0] == 3) & (round(self.ao[0]) == 2):
                mb[1][2] = mb[1][2] + 1
            elif (p[1][0] == 3) & (round(self.ao[0]) == 3):
                mb[2][2] = mb[2][2] + 1
        self.printmb(mb)

    # shows hom many true and false predicted flowers we have
    def printmb(self, mb):
        print(tabulate([['', 'Setosa', mb[0][0], mb[0][1], mb[0][2]],
                        ['Predicted', 'Versicolor', mb[1][0], mb[1][1], mb[1][2]],
                        ['Class', 'Virginica', mb[2][0], mb[2][1], mb[2][2]]],
                       headers=['\nSetosa', 'Actual Class\nVersicolor', '\nVirginica'],
                       tablefmt="plain"))

    # function for training managment
    def train(self, patterns, iterations=400, N=0.1, draw = False):
        # N: learning rate
        im = ImageMaker(self.nh - 1)
        name = 0
        for p in patterns:
            for i in range(self.ni - 1):
                if float(p[0][i]) > self.max[i]:
                    self.max[i] = float(p[0][i])
                if float(p[0][i]) < self.min[i]:
                    self.min[i] = float(p[0][i])
        for i in range(iterations):
            error = 0.0
            k = 0
            for p in patterns:
                k = k + 1
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N)
                if draw:
                    im.makeImage(p[0], p[1][0], self.ao[0], self.ri, self.ro, k)
                    print(k)
            if i % 10 == 0:
                print('error %-.5f' % error)
                # im.makeImage(p[0], p[1][0], self.ao[0], self.ri, self.ro, name)
                name = name + 1
                if error < 0.00001: break

    # normalizing input values
    def normalize(self, inputs, ru=1, rd=-1):
        ninputs = [0.0] * (self.ni - 1)
        for i in range(self.ni - 1):
            ninputs[i] = ((ru - rd) * (float(inputs[i]) - self.min[i]) / (self.max[i] - self.min[i])) + rd
        return ninputs


def main(border=50, iterations=1000, N=0.05, nn=1, draw = False):
    #Some tests
    #tests()

    #open file with text data
    openFile('iris.txt')

    # create a network with four input, five hidden, and one output nodes
    n = NN(4, nn, 1)

    # dla 50 instancji kwiata
    border = round(50 * border / 100)

    # teach and verify arrays
    teachVerify(setosa, versicolor, virginica, border)

    # treain network
    n.train(teach, iterations, N, draw)

    # testing on teaching and verifying datas
    n.test(teach)
    n.test(verify)

# tests function
def tests():
    ti = TestIris()
    ti.main()


# global arrays
arr = []
teach = []
verify = []
setosa = []
versicolor = []
virginica = []

# file parsing function
def openFile(file='iris.txt'):
    with open(file) as csv_file:
        # with using csv reader
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            arr.append([])  # array for one flower
            arr[-1].append([])  # array for input data
            arr[-1].append([])  # array for output data

            arr[-1][0].append(row[0])
            arr[-1][0].append(row[1])
            arr[-1][0].append(row[2])
            arr[-1][0].append(row[3])
            if (row[4] == 'Iris-setosa'):
                row[4] = 1.0
            elif (row[4] == 'Iris-versicolor'):
                row[4] = 2.0
            elif (row[4] == 'Iris-virginica'):
                row[4] = 3.0
            else:
                print('Something went wrong')
            arr[-1][1].append(row[4])

        # dividing data into specific plants arrays
        k = 0
        for r in arr:
            k = k + 1
            if k <= 50:
                setosa.append(r)
            elif k <= 100:
                versicolor.append(r)
            else:
                virginica.append(r)

#creating teaching and veryfing data
def teachVerify(setosa, versicolor, virginica, border=50):
    rows = np.random.permutation(50)
    for i in range(border):
        teach.append(setosa[rows[i]])
    for i in range(border, 50):
        verify.append(setosa[rows[i]])
    for i in range(border):
        teach.append(versicolor[rows[i]])
    for i in range(border, 50):
        verify.append(versicolor[rows[i]])
    for i in range(border):
        teach.append(virginica[rows[i]])
    for i in range(border, 50):
        verify.append(virginica[rows[i]])


if __name__ == '__main__':
    main(100, 1, 0.1, 5, False)