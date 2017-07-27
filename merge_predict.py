from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os
import matplotlib.pyplot as plt


def precision(y_true, y_pred):
    Tp = np.sum((y_pred == 1) & (y_true == 1))
    Fp = np.sum((y_pred == 1) & (y_true == 0))
    x = Tp / (Tp + Fp)
    print('Tp {} Fp {} precision {:.3f}'.format(Tp, Fp, x))
    return x


def recall(y_true, y_pred):
    Tp = np.sum((y_pred == 1) & (y_true == 1))
    Fn = np.sum((y_pred == 0) & (y_true == 1))
    x = Tp / (Tp + Fn)
    print('Tp {} Fn {} recall {:.3f}'.format(Tp, Fn, x))
    return x



df_lgb = pd.read_csv('../tmp/prediction_LIGHT_GBM.csv', dtype={'order_id':np.int32, 'product_id':np.int32})
df_lgb.set_index(['order_id', 'product_id'], inplace=True)


df_log_reg = pd.read_csv('../tmp/prediction_LogReg_Item_Item_cluster10.csv', dtype={'order_id':np.int32, 'product_id':np.int32})
df_log_reg.set_index(['order_id', 'product_id'], inplace=True)

print(
    'LGB all', f1_score(df_lgb.y_test, df_lgb.y_pred > 0.22)
)

print(
    'log_reg all', f1_score(df_log_reg.y_test, df_log_reg.y_pred > 0.22)
)

print(precision(df_lgb.y_test, df_lgb.y_pred > 0.22))
print(recall(df_lgb.y_test, df_lgb.y_pred > 0.22))

print(precision(df_log_reg.y_test, df_log_reg.y_pred > 0.22))
print(recall(df_log_reg.y_test, df_log_reg.y_pred > 0.22))


t = pd.merge(df_lgb, df_log_reg, how='left', left_index=True, right_index=True, suffixes=['', '_reg'])

t1 = t[pd.notnull(t.y_pred_reg)][['y_pred', 'y_pred_reg', 'y_test']]
def pritnf1(t_):
    print('REG',
        f1_score(t_.y_test, (t_.y_pred_reg > 0.22).astype(int)),
          'LGB',
        f1_score(t_.y_test, (t_.y_pred > 0.22).astype(int))
          )


pritnf1(t1)

t2 = pd.merge(df_lgb, df_log_reg, how='inner', left_index=True, right_index=True, suffixes=['', '_reg'])
t2.shape, t1.shape

# Нужно понять какая ошибка растет и почему Presion or Recall or both?
# В выборке по Log_Reg мало примеров. Тежи примеры в LGB отскоренны на много лучше
x = []
y = []
for thr in np.arange(0.17, 0.5, 0.01):
    x.append(thr)
    y.append(f1_score(df_lgb.y_test, (df_lgb.y_pred > thr).astype(int)))

#scale_pos_weight, default=1.0, type=double#    weight of positive class in binary classification task

df_lgb = pd.read_csv('../tmp/prediction_LIGHT_GBM_scale_pos_weight.csv', dtype={'order_id':np.int32, 'product_id':np.int32})
df_lgb.set_index(['order_id', 'product_id'], inplace=True)
print(
    'LGB all', f1_score(df_lgb.y_test, df_lgb.y_pred > 0.22)
)
precision(df_lgb.y_test, df_lgb.y_pred > 0.22)
recall(df_lgb.y_test, df_lgb.y_pred > 0.22)

m = confusion_matrix(df_lgb.y_test, (df_lgb.y_pred > 0.22).astype(int))
m = m / np.sum(m)
sns.heatmap(m, annot=True)

print(len(df_lgb.y_test) / np.sum(df_lgb.y_test == 1))


clust_files = [f for f in os.listdir('../tmp/') if 'lgb_pred_by_clusters_' in f]
df_lgb_clust = None
for f in clust_files:
    file = '../tmp/' + f
    if df_lgb_clust is None:
        df_lgb_clust = pd.read_csv(file)
    else:
        df_lgb_clust = pd.concat([df_lgb_clust, pd.read_csv(file)])

df_lgb_clust.head()

print(
    'LGB clust', f1_score(df_lgb_clust.y_test, df_lgb_clust.y_pred > 0.22)
)

m = confusion_matrix(df_lgb_clust.y_test, (df_lgb_clust.y_pred > 0.22).astype(int))
m = m / np.sum(m)
sns.heatmap(m, annot=True)
