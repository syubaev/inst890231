
import numpy as np
import pandas as pd
from load_data import load_data_frames
import lightgbm as lgb
from sklearn.metrics import f1_score

from sklearn.model_selection import KFold
N_SAMPLE = 10000
orders, priors, train, products = load_data_frames(N_SAMPLE)

orders.set_index('order_id', inplace=True, drop=False)

print('computing product f')
prods = pd.DataFrame()

# кол-во заказов по продуктам
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
# кол-во перезаказов по продуктам
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products.join(prods, on='product_id')
products.set_index('product_id', drop=False, inplace=True)
del prods

### user features


print('computing user f')
usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

users = users.join(usr)
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
print('user f', users.shape)

### userXproduct features
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
del d
userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)
print('user X product f', len(userXproduct))

del priors

### build list of candidate products to reorder, with features ###

def features(selected_orders, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0
    # selected_orders columns
    # 'order_id', 'user_id', 'eval_set', 'order_number', 'order_dow',
    # 'order_hour_of_day', 'days_since_prior_order'
    for row in selected_orders.itertuples():
        i += 1
        if i % 10000 == 0: print('order row', i)
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
    df['user_id'] = df.order_id.map(orders.user_id)
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
    df['aisle_id'] = df.product_id.map(products.aisle_id)
    df['department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

    print('user_X_product related features')
    df['z'] = df.user_id * 100000 + df.product_id
    df.drop(['user_id'], axis=1, inplace=True)
    df['UP_orders'] = df.z.map(userXproduct.nb_orders)
    df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = (df.z.map(userXproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
    df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day -
                                      df.UP_last_order_id.map(orders.order_hour_of_day)
                                      ). \
        map(lambda x: min(x, 24 - x)).astype(np.int8)
    # df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
    #                                              df.order_id.map(orders.order_dow)

    df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
    #print(df.dtypes)
    #print(df.memory_usage())
    return df, labels


### train / test orders ###
print('cross val:')
kf = KFold(5, shuffle=True, random_state=42)

train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

selected_orders = orders[orders.eval_set == 'train']

df_prediction = pd.DataFrame(columns=['order_id', 'product_id', 'y_pred', 'y_test'])
res = pd.DataFrame(columns=['fold', 'f1'])
for i_fold, (train_index, test_index) in enumerate( kf.split( selected_orders.index ) ):

    train_orders = selected_orders.iloc[train_index]
    test_orders = selected_orders.iloc[test_index]

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 96,
        'max_depth': 10,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': -1
    }
    ROUNDS = 100



    f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
                'user_average_days_between_orders', 'user_average_basket',
                'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
                'UP_delta_hour_vs_last']  # 'dow', 'UP_same_dow_as_last_order'

    df_train, labels = features(train_orders, labels_given=True)

    print('formating for lgb')
    d_train = lgb.Dataset(df_train[f_to_use],
                          label=labels,
                          categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'
    del df_train



    print('light GBM train :-)')
    bst = lgb.train(params, d_train, ROUNDS)
    # lgb.plot_importance(bst, figsize=(9,20))
    del d_train

    ### build candidates list for test ###

    df_test, y_test = features(test_orders, labels_given=True)

    print('light GBM predict')
    y_pred_prob = bst.predict(df_test[f_to_use])

    df_test['y_pred'] = y_pred_prob
    df_test['y_test'] = y_test
    df_prediction = pd.concat([df_prediction,
                               df_test[['order_id', 'product_id', 'y_pred', 'y_test']] ])

    TRESHOLD = 0.22
    f1 = f1_score(y_test, (y_pred_prob > TRESHOLD).astype(np.int16))
    print('\n\nF1',f1)
    res.loc[i_fold,:] = [i_fold, f1]

df_prediction.to_csv('../tmp/prediction_LIGHT_GBM.csv', index=False)
print(res)

#d = dict()
#for row in df_test.itertuples():
#    if row.pred > TRESHOLD:
#        try:
#            d[row.order_id] += ' ' + str(row.product_id)
#        except:
#            d[row.order_id] = str(row.product_id)
#
#for order in test_orders.order_id:
#    if order not in d:
#        d[order] = 'None'

