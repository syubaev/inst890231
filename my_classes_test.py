import unittest
import numpy as np
import pandas as pd
import os
import time
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

    def test_user_id_are_same(self):
        dt1 = DataSet(100)
        dt2 = DataSet(100)
        dt3 = DataSet(100)

        self.assertEqual(set(dt1.users.index), set(dt2.users.index))
        self.assertEqual(set(dt2.users.index), set(dt3.users.index))
        self.assertEqual(set(dt1.users.index), set(dt3.users.index))

    def test_user_product_two_real(self):
        dt = DataSet(4000)

        st_time = time.time()
        a = dt.set_user_x_products()
        print(time.time() - st_time)

        st_time = time.time()
        b = dt.set_user_x_products_jit()
        print(time.time() - st_time)

        self.assertTrue(a.shape == b.shape)
        self.assertTrue(np.all(a == b))

    def test_user_product_numba(self):
        priors = self.dt.priors
        priors['user_product'] = priors.product_id + priors.user_id.astype(np.int64) * 100000

        print('compute userXproduct f - this is long...')
        st_time = time.time()
        d = dict()
        for row in priors.itertuples():
            z = row.user_product
            if z not in d:
                d[z] = (1,
                        (row.order_number, row.order_id),
                        row.add_to_cart_order)
            else:
                d[z] = (d[z][0] + 1,
                        max(d[z][1], (row.order_number, row.order_id)),
                        d[z][2] + row.add_to_cart_order)
        print(time.time() - st_time, 'sec')
        userXproduct = pd.DataFrame.from_dict(d, orient='index')
        userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
        userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
        userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
        userXproduct.sort_index(inplace=True)
        # user_product, order_number, order_id, add_to_cart_order


        unq = np.unique(priors['user_product'])
        maping_userProduct_ind = dict(zip(unq, np.arange(len(unq))))

        st_time = time.time()
        priors['user_product_ind'] = priors.user_product.map(maping_userProduct_ind)
        input_array = np.asarray(priors[['user_product_ind', 'order_number', 'order_id', 'add_to_cart_order']])
        out_array = np.zeros((len(unq), 4))

        for i in np.arange(len(input_array)):
            user_prod_ind = input_array[i, 0]
            order_number = input_array[i, 1]
            order_id = input_array[i, 2]
            add_to_cart_order = input_array[i, 3]
            # modify out
            out_array[user_prod_ind, 0] += 1
            prev_order_number = out_array[user_prod_ind, 1]
            if order_number > prev_order_number:
                out_array[user_prod_ind, 1] = order_number
                out_array[user_prod_ind, 2] = order_id
            out_array[user_prod_ind, 3] += add_to_cart_order
        print(time.time() - st_time, 'sec')

        out_array

        userXproduct2 = pd.DataFrame.from_dict(out_array)
        userXproduct2.columns = ['nb_orders', 'order_numb', 'last_order_id', 'sum_pos_in_cart']
        userXproduct2['user_product'] = unq
        userXproduct2.set_index('user_product', inplace=True)
        userXproduct2.drop('order_numb', axis=1, inplace=True)
        userXproduct2.nb_orders = userXproduct2.nb_orders.astype(np.int16)

        userXproduct2.head()
        userXproduct.head()

        self.assertTrue(userXproduct.shape == userXproduct2.shape)
        self.assertTrue(set(userXproduct.index) == set(userXproduct2.index))
        self.assertTrue(np.all(userXproduct == userXproduct2))

    def test_init_from_nb_of_sample(self):
        sample_nb = 93
        dt = DataSet(sample_nb)
        self.assertEqual(dt.users.shape[0], sample_nb)
        self.assertEqual( len(dt.orders.user_id.unique()), sample_nb)

    def test_init_from_array(self):
        users_id = [144288, 145552, 152713, 153941, 158231]

        dt = DataSet(ARR_ORDERS_ID = users_id)
        self.assertTrue(set(dt.orders.user_id) == set(users_id))
        self.assertTrue(set(dt.users.index) == set(users_id))
        self.assertTrue(set(dt.priors.user_id) == set(users_id))

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
