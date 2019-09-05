import math
import bred
import csv
import unittest
import numpy as np
from unittest import TestCase


class TestIris(TestCase):
    # Check creating of matrix
    def test_makeMatrix(self, i=7, j=3, value=3.0):
        matrix = bred.makeMatrix(i, j, value)
        self.assertTrue(matrix[6][2] == 3.0)

    # Check tanh function
    def test_sigmoid(self, x=5):
        k = bred.sigmoid(x)
        formula = math.tanh(x)
        self.assertTrue(k == formula)

    #Check derivative function
    def test_dsigmoid(self, y=5):
        k = bred.dsigmoid(y)
        formula = 1.0 - math.tanh(y) ** 2
        self.assertTrue(k == formula)

    def test_CheckSomething(self, border=50):
        test_arr = []

        with open('iris.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                test_arr.append([])      # array for one flower
                test_arr[-1].append([])  # array for input data
                test_arr[-1].append([])  # array for output data

                test_arr[-1][0].append(row[0])
                test_arr[-1][0].append(row[1])
                test_arr[-1][0].append(row[2])
                test_arr[-1][0].append(row[3])
                if (row[4] == 'Iris-setosa'):
                    row[4] = 1.0
                elif (row[4] == 'Iris-versicolor'):
                    row[4] = 2.0
                elif (row[4] == 'Iris-virginica'):
                    row[4] = 3.0
                else:
                    print('Something went wrong')
                test_arr[-1][1].append(row[4])
                setosa = []
                versicolor = []
                virginica = []
                k = 0
                for r in test_arr:
                    k = k + 1
                    if k <= 50:
                        setosa.append(r)
                    elif k <= 100:
                        versicolor.append(r)
                    else:
                        virginica.append(r)

        border = round(50*border/100)

        bred.teachVerify(setosa, versicolor, virginica, border)

        lenTeach = len(bred.teach)
        lenVerify = len(bred.verify)

        self.assertTrue((test_arr[101] == virginica[1]) & (setosa[49][-1][0] == 1) & \
               (lenTeach == 3*border) & (lenVerify == 3*(50 - border)))

    def main(self):
        unittest.main()


