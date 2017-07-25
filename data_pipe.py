import lightgbm as lgb
from my_classes import DataSet, CrossVal

dt = DataSet(1000)
dt.products_reorder().\
    user_features().\
    user_x_products()
dt.features(True)

print(dt.products.shape)

f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
                    'user_average_days_between_orders', 'user_average_basket',
                    'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
                    'aisle_id', 'department_id', 'product_orders', 'product_reorders',
                    'product_reorder_rate', 'UP_orders', 'UP_orders_ratio',
                    'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
                    'UP_delta_hour_vs_last']


def lgb_predict(X_train, y_train, X_test):
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
        # 'scale_pos_weight': 5
    }
    ROUNDS = 100

    d_train = lgb.Dataset(X_train,
                          label=y_train,
                          categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'

    print('light GBM train :-)')
    bst = lgb.train(params, d_train, ROUNDS)

    print('light GBM predict')
    y_pred_prob = bst.predict(X_test)
    return y_pred_prob

cv = CrossVal('lgb_pred')
res = cv.cross_val_predict(lgb_predict, dt, f_to_use)
