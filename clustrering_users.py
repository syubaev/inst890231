import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from nltk.cluster.gaac import GAAClusterer


IDIR = '../data/'
SAMPLE_SIZE_USERS = None
N_COMPONENTS = 200
N_TAKE_TOP = 15
N_CLUSTERS = 10

pd.options.display.max_rows = 25
pd.options.display.max_columns = 25

# region Load data frames
print('loading prior')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

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
# endregion


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
print("Filter orders by only users from sample. New shape", orders.shape)


def join_orders(df):
    df = df.join(orders, on='order_id', rsuffix='_', how='inner')
    df = df.drop('order_id_', axis=1)
    print("New shape", df.shape)
    return df

print("Filter priors and train")
priors = join_orders(priors)
train = join_orders(train)


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


# TRY TruncatedSVD

svd = TruncatedSVD(n_components=N_COMPONENTS, n_iter=7, random_state=42)

X_C_svd = svd.fit_transform(X_C)
X_C_svd = X_C_svd[:, :N_TAKE_TOP]

#clst = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='cosine', linkage='average')
#clst_labels = clst.fit_predict(X_C_svd[:40000, :])


# test the GAAC clusterer with 4 clusters
clusterer = GAAClusterer(4, normalise=False)
clusters = clusterer.cluster(X_C_svd[:20000, :], True)
