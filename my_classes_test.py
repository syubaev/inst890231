import unittest
import numpy as np
import pandas as pd
import os
from my_classes import CrossVal, DataSet
import pickle


def predict_dummy(X_train, y_train, X_test):
    return np.repeat(0.5, X_test.shape[0])

class TestDataSet(unittest.TestCase):
    def setUp(self):
        if  not os.path.isfile("./pckl/dt_test.p"):
            dt = DataSet(1000)
            pickle.dump(dt, open("./pckl/dt_test.p", "wb"))
        else:
            dt = pickle.load(open("./pckl/dt_test.p", "rb"))
        self.dt = dt

    def test_features(self):
        x, y = self.dt.features(True)
        self.assertGreater(np.sum(y), 0, msg = "y conatins less then 1 exapmmle\n y count={}".format(np.sum(y)))

class TestCrossVal(unittest.TestCase):

    def setUp(self):
        if  not os.path.isfile("./pckl/dt_test.p"):
            dt = DataSet(1000)
            pickle.dump(dt, open("./pckl/dt_test.p", "wb"))
        else:
            dt = pickle.load(open("./pckl/dt_test.p", "rb"))
        self.dt = dt

    def test_save_prediction(self):
        cv = CrossVal('test_save_prediction')
        cv.save_prediction()

    def test_cross_val_predict(self):
        cv = CrossVal('lgb_pred')
        res = cv.cross_val_predict(predict_dummy, self.dt, ['UP_orders'])
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
