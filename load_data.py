import numpy as np
import pandas as pd

def load_data_frames(SAMPLE_SIZE_USERS = None):
    IDIR = '../data/'
    print('loading orders')
    #order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
    orders = pd.read_csv(IDIR + 'orders.csv', dtype={
            'order_id': np.int32,
            'user_id': np.int32,
            'eval_set': 'category',
            'order_number': np.int16,
            'order_dow': np.int8,
            'order_hour_of_day': np.int8,
            'days_since_prior_order': np.float32})


    print('loading prior')
    #order_id, product_id, add_to_cart_order, reordered
    priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
                'order_id': np.int32,
                'product_id': np.uint16,
                'add_to_cart_order': np.int16,
                'reordered': np.int8})

    print('loading train')
    #order_id, product_id, add_to_cart_order, reordered
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

    return orders, priors, train, products
    ###


