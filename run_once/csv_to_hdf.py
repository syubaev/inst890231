import pandas as pd
import os
import numpy as np

IDIR = '../data/'
print('loading orders')
# order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
    'order_id': np.int32,
    'user_id': np.int32,
#    'eval_set': 'category',
    'order_number': np.int16,
    'order_dow': np.int8,
    'order_hour_of_day': np.int8,
    'days_since_prior_order': np.float32})
orders.to_hdf(IDIR + 'orders.hdf', 'orders')
orders2 = pd.read_hdf(IDIR + 'orders.hdf', 'orders')

print('loading prior')
# order_id, product_id, add_to_cart_order, reordered
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
    'order_id': np.int32,
    'product_id': np.uint16,
    'add_to_cart_order': np.int16,
    'reordered': np.int8})
priors.to_hdf(IDIR + 'priors.hdf', 'priors')
priors2 = pd.read_hdf(IDIR + 'priors.hdf', 'priors')

print('loading train')
# order_id, product_id, add_to_cart_order, reordered
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
    'order_id': np.int32,
    'product_id': np.uint16,
    'add_to_cart_order': np.int16,
    'reordered': np.int8})
train.to_hdf(IDIR + 'train.hdf', 'train')

print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
    'product_id': np.uint16,
    'order_id': np.int32,
    'aisle_id': np.uint8,
    'department_id': np.uint8},
                       usecols=['product_id', 'aisle_id', 'department_id'])
products.to_hdf(IDIR + 'products.hdf', 'products')