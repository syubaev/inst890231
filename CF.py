import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

IDIR = '../data/'

pd.options.display.max_rows = 25
pd.options.display.max_columns = 25

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

###

print('computing product f')
products.set_index('product_id', drop=False, inplace=True)


print('add order info to priors')
orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)

print('computing user f')

#users = pd.DataFrame()
#users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)

#print('user f', users.shape)

# products = 49688, user = 206209
SAMPLE_SIZE_USERS = 10000
USERS_SPLIT = 0.3
THRESHOLD = 0.35
USER_SPLIT_IND = int(USERS_SPLIT*SAMPLE_SIZE_USERS)
np.random.seed(42)

users_train = orders[orders.eval_set == 'train'].user_id.values
user_sample = np.random.choice(users_train, size=SAMPLE_SIZE_USERS, replace=False)
np.random.shuffle(user_sample)
user_sample_A = user_sample[:USER_SPLIT_IND]
user_sample_B = user_sample[USER_SPLIT_IND:]

# SAMPLE FOR ORDERS
orders = orders[orders.user_id.isin(user_sample)]
print("orders new shape", orders.shape)

priors = priors.join(orders, on='order_id', rsuffix='_', how='inner')
priors.drop('order_id_', inplace=True, axis=1)
print("priors new shape", priors.shape)

train = train.join(orders, on='order_id', rsuffix='_', how='inner')
train.drop('order_id_', inplace=True, axis=1)
print("train new shape", train.shape)

users_prior = pd.DataFrame()
users_prior['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users_prior = users_prior.reindex(user_sample)
users_train = pd.DataFrame()
users_train['all_products'] = train.groupby('user_id')['product_id'].apply(set)
users_train = users_train.reindex(user_sample)
# LAST N prior orders

# C : Items referred to as independent variables
# D : Items referred to as dependent variables
user_products_C = users_prior.all_products.apply(
    lambda x: " ".join([str(prod_id) + '_C' for prod_id in x]))
user_products_D = users_train.all_products.apply(
    lambda x: " ".join([str(prod_id) + '_D' for prod_id in x]))


X_C = CountVectorizer().fit_transform(user_products_C)
cv = CountVectorizer()
X_D = cv.fit_transform(user_products_D)

X_D = (X_D > 0).astype(np.int16)
X_C = (X_C > 0).astype(np.int16)

# Для каждого активного пользователя посчитать


# TRY TruncatedSVD
pca = PCA(n_components=207, random_state=42)
X_C_pca = pca.fit_transform(X_C.todense())
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.plot(pca.explained_variance_ratio_)
X_C_pca = X_C_pca[:, :10]

from sklearn.metrics import f1_score

res = pd.DataFrame(columns=['i','roc', 'f1'])
i_loc = 0
clf = LogisticRegression()
for i in range(X_D.shape[1]):
    A, = np.array(X_D[:, i].todense().T)
    if np.sum(A) > 50:
        X_C_pca_A, X_C_pca_B = X_C_pca[:USER_SPLIT_IND, :], X_C_pca[USER_SPLIT_IND:, :]
        y_A, y_B = A[:USER_SPLIT_IND], A[USER_SPLIT_IND:]
        clf.fit(X_C_pca_A, y_A)
        y_pred_prob = clf.predict_proba(X_C_pca_B)[:, 1]
        y_pred = (y_pred_prob > THRESHOLD).astype(np.int16)
        res.loc[i_loc,:] = [i,
                            roc_auc_score(y_B, y_pred_prob),
                            f1_score(y_B, y_pred)
                            ]
        i_loc += 1
        #print('col', i, 'sum of items', np.sum(A), 'roc', roc_auc_score(y_B, y_pred))

A, = np.array(X_D[:, 16886].todense().T)

y_A, y_B = A[:USER_SPLIT_IND], A[USER_SPLIT_IND:]
