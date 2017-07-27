import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


class DataSet:
    """Class to maintenance data loading and features calculation
    orders, priors, train, products, selected_orders,
    users, userXproduct
    """

    def __init__(self, SAMPLE_SIZE_USERS=None):
        """
        :param SAMPLE_SIZE_USERS: - number of user for random sampling. Default is None, for all users.
        """
        self.orders, self.priors, self.train, self.products = self.load_data_frames(SAMPLE_SIZE_USERS)
        self.selected_orders = self.set_selected_orders()
        self.users = self.set_user()
        self.userXproduct = self.set_user_x_products()
        self.products = self.set_products_reorder()

        assert len(set(self.orders[self.orders.eval_set == 'train'].user_id) - set(self.orders.user_id)) == 0

    def load_data_frames(self, SAMPLE_SIZE_USERS=None):
        IDIR = '../data/'
        print('loading orders')
        # order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
        orders = pd.read_csv(IDIR + 'orders.csv', dtype={
            'order_id': np.int32,
            'user_id': np.int32,
            'eval_set': 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32})

        print('loading prior')
        # order_id, product_id, add_to_cart_order, reordered
        priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

        print('loading train')
        # order_id, product_id, add_to_cart_order, reordered
        train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

        print('loading products')
        products = pd.read_csv(IDIR + 'products.csv', dtype={
            'product_id': np.uint16,
            'order_id': np.int32,
            'aisle_id': np.uint8,
            'department_id': np.uint8},
                               usecols=['product_id', 'aisle_id', 'department_id'])

        print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
        print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
        print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

        np.random.seed(42)
        users_train = orders[orders.eval_set == 'train'].user_id.values

        if SAMPLE_SIZE_USERS is None:
            user_sample = users_train
        else:
            user_sample = np.random.choice(users_train, size=SAMPLE_SIZE_USERS, replace=False)

        # SAMPLE FOR ORDERS
        orders = orders[orders.user_id.isin(user_sample)]

        print("orders new shape", orders.shape)

        # pd.merge(left, right, how='inner', on=None
        priors = pd.merge(priors, orders, how='inner', on='order_id')
        train = pd.merge(train, orders, how='inner', on='order_id')

        assert len(set(orders.order_id) - set(priors.order_id) - set(train.order_id)) == 0
        if SAMPLE_SIZE_USERS is not None:
            assert len(set(orders.user_id)) == SAMPLE_SIZE_USERS

        print('priors {}'.format(priors.shape))
        print('orders {}'.format(orders.shape))
        print('train {}'.format(train.shape))

        orders.set_index('order_id', inplace=True, drop=False)
        return orders, priors, train, products

    def set_products_reorder(self):
        """
        :return: products с доп фичами:
            orders - кол-во заказов продукта в прошлом
            reorders - кол-во раз когда продукт перезаказывали
            reorder_rate - отношение перезаказов к заказам продуката
        """
        # кол-во заказов по продуктам
        priors = self.priors
        products = self.products
        prods = pd.DataFrame()

        prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
        # кол-во перезаказов по продуктам
        prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
        prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
        products = products.join(prods, on='product_id')
        products.set_index('product_id', drop=False, inplace=True)
        return products

    def set_user(self):
        """

        :return: создает df users с фичами:
            average_days_between_orders - среднее кол-во дней между заказами
            nb_orders - кол-во заказов
            total_items : int - общее кол-во позиций
            all_products : set - сет продкутов купелый юзером
            total_distinct_items : int - кол-во уникальных продуктов
            average_basket : int - средний размер корзины
        """
        orders, priors = self.orders, self.priors
        print('computing user f')
        usr = pd.DataFrame()
        usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(
            np.float32)
        usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

        users = pd.DataFrame()
        users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
        users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
        users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

        users = users.join(usr)
        users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
        print('user f', users.shape)
        return users

    def set_user_x_products(self):
        """
        :return: add userXproduct
        Вычисляется все комбинации продукт-пользователь;
        При этом считаются только те пары, которые пользователь покупал.
        В словарь d записано для каждой пары юзер-продукт:
            user_product : int - id из пары юзер-продукт
            nb_orders : int - кол-во заказов продукта,
            last_order_id : int - ид_последнего заказа,
            sum_pos_in_cart : int - сумма порядковых номеров добавления продукта в корзину
        """
        priors = self.priors
        priors['user_product'] = priors.product_id + priors.user_id * 100000

        print('compute userXproduct f - this is long...')

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

        print('to dataframe (less memory)')
        userXproduct = pd.DataFrame.from_dict(d, orient='index')
        userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
        userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
        userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
        userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)
        print('user X product f', len(userXproduct))

        return userXproduct

    def set_selected_orders(self):
        return self.orders[self.orders.eval_set == 'train'][['order_id', 'user_id']]

    #def features_create(self):

    def features(self, labels_given=False):
        """

        :param labels_given:
        :return: df : pd.DataFrame, labels : np.array
        df - содержит в себе
        """
        users, train, orders, products, userXproduct, selected_orders = \
            self.users, self.train, self.orders, self.products, self.userXproduct, self.selected_orders
        train.set_index(['order_id', 'product_id'], drop=False, inplace=True)
        order_list = []
        product_list = []
        labels = []
        i = 0
        # selected_orders columns
        # 'order_id', 'user_id', 'eval_set', 'order_number', 'order_dow',
        # 'order_hour_of_day', 'days_since_prior_order'
        for row in selected_orders.itertuples():
            i += 1
            if i % 10000 == 0:
                print('order row', i)
            order_id = row.order_id
            user_id = row.user_id
            user_products = users.all_products[user_id]  # SET из product_id которые заказывл юзер
            product_list += user_products  # Добавляем продукты пользователя в список продуктов
            order_list += [order_id] * len(
                user_products)  # Добавляем ид_пользователя n раз в список заказов, где n - кол-во продуктов
            if labels_given:
                # Если ид_заказа и продукт есть в трейне, то True иначе False -> 1, 0
                labels += [(order_id, product) in train.index for product in user_products]

        df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.int32)
        labels = np.array(labels, dtype=np.int8)
        del order_list, product_list


        print('user related features')
        assert orders.index.name == 'order_id'
        df['user_id'] = df.order_id.map(orders.user_id)

        assert users.index.name == 'user_id'
        df['user_total_orders'] = df.user_id.map(users.nb_orders)
        df['user_total_items'] = df.user_id.map(users.total_items)
        df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
        df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
        df['user_average_basket'] = df.user_id.map(users.average_basket)

        print('order related features')
        # df['dow'] = df.order_id.map(orders.order_dow)
        df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
        df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
        df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders

        print('product related features')

        assert products.index.name == 'product_id'
        df['aisle_id'] = df.product_id.map(products.aisle_id)
        df['department_id'] = df.product_id.map(products.department_id)
        df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
        df['product_reorders'] = df.product_id.map(products.reorders)
        df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

        print('user_X_product related features')
        df['z'] = df.user_id * 100000 + df.product_id
        # df.drop(['user_id'], axis=1, inplace=True)
        df['UP_orders'] = df.z.map(userXproduct.nb_orders)
        assert np.sum(pd.isnull(df['UP_orders'])) == 0

        df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
        df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
        df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
        df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
        df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
        df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day -
                                          df.UP_last_order_id.map(orders.order_hour_of_day)
                                          ). \
            map(lambda x: min(x, 24 - x)).astype(np.int8)

        df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)

        print(df.memory_usage())
        return df, labels


class CrossVal:
    def __init__(self, file_prediction, nb_folds=5):
        self.kf = KFold(nb_folds, shuffle=True, random_state=42)
        self.res = pd.DataFrame(columns=['fold', 'f1'])
        self.file_prediction = file_prediction
        self.set_prediction()

    def set_prediction(self):
        self.df_prediction = pd.DataFrame(columns=['order_id', 'product_id', 'y_pred', 'y_test'])

    def save_prediction(self):
        self.df_prediction.to_csv('../tmp/' + self.file_prediction + '.csv', index=False)

    def cross_val_predict(self, f_prediction, dt: DataSet, f_to_use: list):
        """
        Функция разбивает выбранные заказы на трейн и тест и обучает модель.
        :param dt:DataSet
        :param f_prediction: - фун-ция от (X_train, y_train, X_test) возращает массив вероятность
        :return:
        """
        self.set_prediction()
        selected_orders = dt.selected_orders
        X, y = dt.features(labels_given=True)
        X.loc[:, 'labels'] = y
        assert np.sum(y) > 0
        TRESHOLD = 0.22
        X.set_index('order_id', inplace=True, drop=False)
        for i_fold, (train_index, test_index) in enumerate(self.kf.split(selected_orders.order_id)):
            train_orders = selected_orders.order_id.iloc[train_index]
            test_orders = selected_orders.order_id.iloc[test_index]

            X_train, X_test = X.loc[train_orders], X.loc[test_orders]
            y_train, y_test = X_train['labels'], X_test['labels']

            assert np.sum(y_train) > 0

            print("fold_numb {} .train {} test {} train shape".format(i_fold, np.sum(y_train), np.sum(y_test), len(y_train) ))
            X_train = X_train.drop('labels', axis=1)
            X_test = X_test.drop('labels', axis=1)

            y_pred_proba = f_prediction(X_train[f_to_use], y_train, X_test[f_to_use])



            f1 = f1_score(y_test,  (

                y_pred_proba > TRESHOLD).astype(np.int16))

            X_test['y_test'] = y_test
            X_test['y_pred'] = y_pred_proba

            self.res.loc[i_fold, :] = [i_fold, f1]
            self.df_prediction = pd.concat([self.df_prediction,
                                            X_test[['order_id', 'product_id', 'y_pred', 'y_test']]
                                            ])

        self.save_prediction()
        return self.res

