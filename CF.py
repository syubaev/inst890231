import numpy as np
import pandas as pd
from load_data import load_data_frames

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
import time

# clustering
from sklearn.cluster import AgglomerativeClustering
from nltk.cluster.gaac import GAAClusterer

from tqdm import tqdm

N_CLUSTERS = 10
N_SAMPLE = 10000
N_COMPONENTS = 200
N_TAKE_TOP = 10


svd = TruncatedSVD(n_components=N_COMPONENTS, n_iter=7, random_state=42)
clusterer = AgglomerativeClustering(n_clusters=N_CLUSTERS, affinity='cosine', linkage='average')

cv = CountVectorizer()
tfidf = TfidfTransformer()
pipe = Pipeline(steps=[('cv', cv), ('tf', tfidf)])

# USER PRIOR N LAST ORDERS
LAST_N_PRIORS = 3

pd.options.display.max_rows = 25
pd.options.display.max_columns = 25

def cross_val_item_item(X, Y_items, order_id_list, vocabulary):
    def pos_rate(x): return round(np.sum(x) / len(x), 3)

    clf = LogisticRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    list_threshold = [0.2, 0.3, 0.5]
    f1_cols = ['f1_' + str(thr) for thr in list_threshold]
    res = pd.DataFrame(columns=['i', 'fold', 'pos_rate_train', 'pos_rate_test', 'pos_rate'] + f1_cols)
    df_prediction = pd.DataFrame(columns=['order_id', 'product_id', 'y_pred', 'y_test'])

    i_loc = 0
    # For each item in dependent part
    for i_item in range(Y_items.shape[1]):
        y, = np.array(Y_items[:, i_item].todense().T)
        i_product = vocabulary[i_item]
        try:
            if pos_rate(y) > 0.05:
                for i_fold, (train_index, test_index) in enumerate( kf.split( y )):

                    X_train, X_test = X[train_index, :], X[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    order_id_test = order_id_list[y_test]

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
                    df_test = pd.DataFrame({'order_id': order_id_test,
                                            'product_id': [i_product] * len(y_test),
                                            'y_pred': y_pred_prob,
                                            'y_test': y_test})

                    df_prediction = pd.concat([df_prediction,
                                               df_test[['order_id', 'product_id', 'y_pred', 'y_test']]])
                    i_loc += 1
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")


    res[f1_cols] = res[f1_cols].astype(float)
    return res, df_prediction


st_time = time.time()
orders, priors, train, products = load_data_frames(N_SAMPLE)


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


users_train = pd.DataFrame()
users_train['all_products'] = train.groupby('user_id')['product_id'].apply(set)

assert np.array_equal(users_prior.index, users_train.index)
assert np.all(orders[orders.eval_set == 'train'].groupby('user_id')['order_id'].count() == 1) # Проверка что 1 пользователю соотвествует 1 заказ

user_id_list = users_prior.index
tmp = orders[orders.eval_set == 'train'][['order_id', 'user_id']]
tmp.set_index('user_id', inplace=True)
order_id_list = tmp.loc[user_id_list].order_id.values
del tmp

# C : Items referred to as independent variables
# D : Items referred to as dependent variables
user_products_C = users_prior.all_products.apply(
    lambda x: " ".join([str(prod_id) + '_C' for prod_id in x]))
user_products_D = users_train.all_products.apply(
    lambda x: " ".join([str(prod_id) for prod_id in x]))

print('Calc CountVectorizer')
X_C = pipe.fit_transform(user_products_C)
X_D = pipe.fit_transform(user_products_D)

X_D = (X_D > 0).astype(np.int16)
X_C = (X_C > 0).astype(np.int16)
print('CountVectorizer done, X_D=', X_D.shape, 'X_C=', X_C.shape)
# Для каждого активного пользователя посчитать


X_C_svd = svd.fit_transform(X_C)
print('Done SVD calc; n_components', N_COMPONENTS)
#plt.plot(np.cumsum(svd.explained_variance_ratio_)); plt.show()
#plt.plot(svd.explained_variance_ratio_)
X_C_svd = X_C_svd[:, :N_TAKE_TOP]



clst_labels = clusterer.fit_predict(X_C_svd)
print('Done clustering ', N_CLUSTERS)

df_prediction_global = None
for clust in range(clusterer.n_clusters):
     bind = clst_labels == clust
     order_list_in_cluster = order_id_list[bind]
     X = X_C_svd[bind, :]
     Y = X_D[bind, :]
     res, df_prediction_cluster = cross_val_item_item(X, Y, order_list_in_cluster, cv.get_feature_names())
     print('CLUSTER #', clust, 'NUMBER IN CLUSTER', np.sum(bind))
     print(
         res.groupby('i').agg(
             {'f1_0.2': ['mean', 'std']})
     )
     if df_prediction_global is None:
         df_prediction_global = df_prediction_cluster
     else:
         df_prediction_global = pd.concat([df_prediction_global, df_prediction_cluster])
#clusterer = GAAClusterer(4, normalise=False)
#clusters = clusterer.cluster(X_C_svd[:20000, :], True)


#res, df_prediction = cross_val_item_item(X_C_svd, X_D, order_id_list, cv.get_feature_names())

#df_prediction.to_csv('../tmp/prediction_LogReg_Item_Item.csv', index=False)

df_prediction_global.to_csv('../tmp/prediction_LogReg_Item_Item_cluster' +\
                            str(N_CLUSTERS) +'.csv', index=False)

print('Done calcs, sec', round(time.time() - st_time, 3))

print(
    res.groupby('i').agg(
        {'f1_0.2': ['mean', 'std']})
)
