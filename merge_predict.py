import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

df_lgb = pd.read_csv('../tmp/prediction_LIGHT_GBM.csv', dtype={'order_id':np.int32, 'product_id':np.int32})
df_lgb.set_index(['order_id', 'product_id'], inplace=True)

df_log_reg = pd.read_csv('../tmp/prediction_LogReg_Item_Item_cluster10.csv', dtype={'order_id':np.int32, 'product_id':np.int32})
df_log_reg.set_index(['order_id', 'product_id'], inplace=True)


t = pd.merge(df_lgb, df_log_reg, how='left', left_index=True, right_index=True, suffixes=['', '_reg'])


t1 = t[pd.notnull(t.y_pred_reg)][['y_pred', 'y_pred_reg', 'y_test']]
print('REG',
    f1_score(t1.y_test, (t1.y_pred_reg > 0.22).astype(int)),
      'LGB',
    f1_score(t1.y_test, (t1.y_pred > 0.22).astype(int))
      )