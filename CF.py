import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import time

from tqdm import tqdm

IDIR = '../data/'
SAMPLE_SIZE_USERS = 1000


pd.options.display.max_rows = 25
pd.options.display.max_columns = 25

def cross_val_item_item(X, Y_items):
    def pos_rate(x): return round(np.sum(x) / len(x), 3)

    clf = LogisticRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    list_threshold = [0.2, 0.3, 0.5]
    f1_cols = ['f1_' + str(thr) for thr in list_threshold]
    res = pd.DataFrame(columns=['i', 'fold', 'pos_rate_train', 'pos_rate_test', 'pos_rate'] + f1_cols)


    i_loc = 0
    # For each item in dependent part
    for i_item in tqdm(range(Y_items.shape[1])):
        y, = np.array(Y_items[:, i_item].todense().T)

        if pos_rate(y) > 0.05:
            for i_fold, (train_index, test_index) in enumerate( kf.split(range( len(y) )) ):

                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                clf.fit(X_train, y_train)
                y_pred_prob = clf.predict_proba(X_test)[:, 1]
                f1_list = [f1_score(y_test, (y_pred_prob > thr).astype(np.int16)).__round__(3)
                           for thr in list_threshold]
                res.loc[i_loc, :] = [i_item,
                                     i_fold,
                                     pos_rate(y_train),
                                     pos_rate(y_test),
                                     pos_rate(y)
                                    ] + f1_list
                i_loc += 1

    res[f1_cols] = res[f1_cols].astype(float)
    return res


st_time = time.time()
print('loading prior')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8},
            engine='c')

print('loading train')
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

print('loading orders')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

# print('loading products')
# products = pd.read_csv(IDIR + 'products.csv', dtype={
#         'product_id': np.uint16,
#         'aisle_id': np.uint8,
#         'department_id': np.uint8},
#         usecols=['product_id', 'product_name', 'aisle_id', 'department_id'])
# print('computing product f')
# products.set_index('product_id', drop=False, inplace=True)

print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

###

print('add order info to priors')
orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)


np.random.seed(42)

users_train = orders[orders.eval_set == 'train'].user_id.values

if SAMPLE_SIZE_USERS is None:
    user_sample = users_train
else:
    user_sample = np.random.choice(users_train, size=SAMPLE_SIZE_USERS, replace=False)
np.random.shuffle(user_sample)

# SAMPLE FOR ORDERS
orders = orders[orders.user_id.isin(user_sample)]
print("orders new shape", orders.shape)

priors = priors.join(orders, on='order_id', rsuffix='_', how='inner')
priors.drop('order_id_', inplace=True, axis=1)
print("priors new shape", priors.shape)

train = train.join(orders, on='order_id', rsuffix='_', how='inner')
train.drop('order_id_', inplace=True, axis=1)
print("train new shape", train.shape)

# USER PRIOR N LAST ORDERS
LAST_N_PRIORS = 4

print('Calc products merged str. LAST_N_PRIORS=', LAST_N_PRIORS)
users_prior = pd.DataFrame()
if LAST_N_PRIORS is not None:

    orders_numb_top = orders[['user_id', 'order_number', 'order_id']].\
        sort_values(['user_id', 'order_number'], ascending=[1, 0]).\
        groupby('user_id').head(LAST_N_PRIORS)['order_id'].\
        values

    priors_filtered = priors[priors.order_id.isin(orders_numb_top)]
    users_prior['all_products'] = priors_filtered.groupby('user_id')['product_id'].apply(set)
else:
    users_prior['all_products'] = priors.groupby('user_id')['product_id'].apply(set)

users_prior = users_prior.reindex(user_sample)

users_train = pd.DataFrame()
users_train['all_products'] = train.groupby('user_id')['product_id'].apply(set)
users_train = users_train.reindex(user_sample)


# C : Items referred to as independent variables
# D : Items referred to as dependent variables
user_products_C = users_prior.all_products.apply(
    lambda x: " ".join([str(prod_id) + '_C' for prod_id in x]))
user_products_D = users_train.all_products.apply(
    lambda x: " ".join([str(prod_id) + '_D' for prod_id in x]))

print('Calc CountVectorizer')
X_C = CountVectorizer().fit_transform(user_products_C)
cv = CountVectorizer()
X_D = cv.fit_transform(user_products_D)

X_D = (X_D > 0).astype(np.int16)
X_C = (X_C > 0).astype(np.int16)
print('CountVectorizer done, X_D=', X_D.shape, 'X_C=', X_C.shape)
# Для каждого активного пользователя посчитать

N_COMPONENTS = 200
N_TAKE_TOP = 15
# TRY TruncatedSVD

svd = TruncatedSVD(n_components=N_COMPONENTS, n_iter=7, random_state=42)

X_C_svd = svd.fit_transform(X_C)
print('Done SVD calc; n_components', N_COMPONENTS)
#plt.plot(np.cumsum(svd.explained_variance_ratio_)); plt.show()
#plt.plot(svd.explained_variance_ratio_)
X_C_svd = X_C_svd[:, :N_TAKE_TOP]




res = cross_val_item_item(X_C_svd, X_D)

print('Done calcs, sec', round(time.time() - st_time, 3))

print(
    res.groupby('i').agg(
        {'f1_0.2': ['mean', 'std']})
)
#res.set_index('i', drop=False, inplace=True)

#
# res.loc[354]
#res[res['f1_0.2'] > 0.4]


#list(cv.vocabulary_.keys())[1221]

#pd.merge(orders, orders_numb_top, on=['user_id', 'order_number'], how='inner')