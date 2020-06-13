from django.shortcuts import get_object_or_404, render
from django.http import HttpResponse, HttpResponseRedirect, Http404, JsonResponse
from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework.renderers import JSONRenderer
from rest_framework import serializers
from django.template import loader
import json
from .models import User, Stock

import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.regularizers import L1L2
from keras import backend as K
from sklearn.externals import joblib  # save the model
import keras.losses


# Using HttpResponse function
def user_details(request):
    # - show revers order
    latest_users = User.objects.order_by('-name')[:5]

    template = loader.get_template('stock/index.html')
    context = {
        'latest_users': latest_users,
    }
    return HttpResponse(template.render(context, request))


# def user_details(request):
#     # - show revers order
#     latest_users = User.objects.order_by('-name')[:5]
#     context = {
#         'latest_users': latest_users,
#     }
#     return render(request, 'stock/index.html', context)

def user_get_by_id(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    response = "User id : %s"
    return HttpResponse(response % user.name)

    # def user_get_by_id(request, user_id):
    #     try:
    #         user = User.objects.get(pk=user_id)
    #         response = "User id : %s"
    #     except User.DoesNotExist:
    #         raise Http404("User Does not exist")
    #     return HttpResponse(response % user.name)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def create_model(request):
    value1 = create_creative_stock_model();
    value2 = create_directfn_stock_model();
    value3 = create_ctc_stock_model();
    # ----
    return HttpResponse(value2 + " " + value1 + " " + value3);


# used to create model
def create_creative_stock_model():
    dataset = pd.read_csv(r'C:\Users\shanaka\Desktop\final semester\projects V2\stockserver\formatted_all.csv')
    dataset = dataset[['DATE', 'OP', 'CLS']]
    days = []

    for i in range(1549):
        s = str(dataset.loc[i, 'DATE'])
        # you could also import date instead of datetime and use that.
        date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        days.append(date.weekday())
        dataset.loc[i, 'DATE'] = date.date()

    dataset['DAY'] = days
    dataset = dataset[['DATE', 'DAY', 'CLS']]

    # values = dataset.iloc[:, 1:3].values;
    values = dataset[['DAY', 'CLS']]
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    values[['DAY']] = scaler.fit_transform(values[['DAY']])
    values[['CLS']] = scaler2.fit_transform(values[['CLS']])

    reframed = series_to_supervised(values, 4, 1)

    # split into train and test sets
    values = reframed.values
    train = values[:1000, :]
    val = values[1000:1349, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    val_X, val_y = val[:, :-1], val[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]),
                   bias_regularizer=L1L2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.5))

    model.add(LSTM(units=64))
    model.add(Dropout(0.5))

    model.add(Dense(units=16, init='uniform', activation='relu'))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)
    history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_data=(val_X, val_y), verbose=2,
                        shuffle=False)

    joblib.dump(model, 'stock-creative-pre.joblib')
    return 'done creating creative stock model';


# used to create model
def create_ctc_stock_model():
    dataset = pd.read_csv(r'C:\Users\shanaka\Desktop\final semester\projects V2\stockserver\formatted_128444-CTC.csv')
    dataset = dataset[['DATE', 'OP', 'CLS']]
    days = []

    for i in range(1635):
        s = str(dataset.loc[i, 'DATE'])
        # you could also import date instead of datetime and use that.
        date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        days.append(date.weekday())
        dataset.loc[i, 'DATE'] = date.date()

    dataset['DAY'] = days
    dataset = dataset[['DATE', 'DAY', 'CLS']]

    # values = dataset.iloc[:, 1:3].values;
    values = dataset[['DAY', 'CLS']]
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    values[['DAY']] = scaler.fit_transform(values[['DAY']])
    values[['CLS']] = scaler2.fit_transform(values[['CLS']])

    reframed = series_to_supervised(values, 4, 1)

    # split into train and test sets
    values = reframed.values
    train = values[:1000, :]
    val = values[1000:1425, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    val_X, val_y = val[:, :-1], val[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]),
                   bias_regularizer=L1L2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.5))

    model.add(LSTM(units=64))
    model.add(Dropout(0.5))

    model.add(Dense(units=16, init='uniform', activation='relu'))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)
    history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_data=(val_X, val_y), verbose=2,
                        shuffle=False)

    joblib.dump(model, 'stock-ctc-pre.joblib')
    return 'done creating CTC stock model';


# used to create model
def create_directfn_stock_model():
    dataset = pd.read_csv(r'C:\Users\shanaka\Desktop\final semester\projects V2\stockserver\formatted_128714-JKH.csv')
    dataset = dataset[['DATE', 'OP', 'CLS']]
    days = []

    for i in range(1773):
        s = str(dataset.loc[i, 'DATE'])
        # you could also import date instead of datetime and use that.
        date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        days.append(date.weekday())
        dataset.loc[i, 'DATE'] = date.date()

    dataset['DAY'] = days
    dataset = dataset[['DATE', 'DAY', 'CLS']]

    # values = dataset.iloc[:, 1:3].values;
    values = dataset[['DAY', 'CLS']]
    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    values[['DAY']] = scaler.fit_transform(values[['DAY']])
    values[['CLS']] = scaler2.fit_transform(values[['CLS']])

    reframed = series_to_supervised(values, 4, 1)

    # split into train and test sets
    values = reframed.values
    train = values[:1400, :]
    val = values[1400:1525, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    val_X, val_y = val[:, :-1], val[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]),
                   bias_regularizer=L1L2(l1=0.001, l2=0.001)))
    model.add(Dropout(0.5))

    model.add(LSTM(units=64))
    model.add(Dropout(0.5))

    model.add(Dense(units=16, init='uniform', activation='relu'))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error)
    history = model.fit(train_X, train_y, epochs=100, batch_size=50, validation_data=(val_X, val_y), verbose=2,
                        shuffle=False)

    joblib.dump(model, 'stock-directfn-pre.joblib')
    return 'done creating directfn stock model';


def run_model(request, data):
    creative_value = run_creative_model(data);
    directfn_value = run_directfn_model(data);
    ctc_value = run_ctc_model(data);

    send_value_creative = {'id': 2, 'predict': creative_value};
    send_value_directfn = {'id': 3, 'predict': directfn_value};
    send_value_ctc = {'id': 4, 'predict': ctc_value};
    send_array = [send_value_creative, send_value_directfn, send_value_ctc]
    # ----
    # posts_serialized = json.dumps(send_value_array, indent=2)
    return JsonResponse(send_array, safe=False);


# used to get creative stock data prediction value
def run_creative_model(data):
    data_set = pd.read_csv(r'C:\Users\shanaka\Desktop\final semester\projects V2\stockserver\formatted_all.csv')
    data_set = data_set[['DATE', 'OP', 'CLS']]
    days = []

    for i in range(1549):
        s = str(data_set.loc[i, 'DATE'])
        # you could also import date instead of datetime and use that.
        date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        days.append(date.weekday())
        data_set.loc[i, 'DATE'] = date.date()

    data_set['DAY'] = days
    data_set = data_set[['DATE', 'DAY', 'CLS']]
    values = data_set[['DAY', 'CLS']]
    values = values.astype('float32')
    # stock_values = data_set.iloc[:, 1:3].values;
    stock_values = data_set[['DAY', 'CLS']]
    stock_values = stock_values.astype('float32')
    data_set.iloc[1349 + data: 1349 + data + 1, :].values;
    # normalize features
    # values = scaler.fit_transform(stock_values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    values[['DAY']] = scaler.fit_transform(stock_values[['DAY']])
    values[['CLS']] = scaler2.fit_transform(stock_values[['CLS']])

    reframed = series_to_supervised(values, 4, 1)
    values = reframed.values

    model = joblib.load('stock-creative-pre.joblib')
    input_values = values[1349 + data: 1349 + data + 1, :]
    input_values_X, input_values_y = input_values[:, :-1], input_values[:, -1]
    input_values_X = input_values_X.reshape((input_values_X.shape[0], 1, input_values_X.shape[1]))
    predictions = model.predict(input_values_X, batch_size=1)
    predicted_stock_price = scaler2.inverse_transform(predictions)
    Send_value = predicted_stock_price[0][0];
    return str(Send_value);


# used to get creative stock data prediction value
def run_directfn_model(data):
    data_set = pd.read_csv(r'C:\Users\shanaka\Desktop\final semester\projects V2\stockserver\formatted_128714-JKH.csv')
    data_set = data_set[['DATE', 'OP', 'CLS']]
    days = []

    for i in range(1773):
        s = str(data_set.loc[i, 'DATE'])
        # you could also import date instead of datetime and use that.
        date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        days.append(date.weekday())
        data_set.loc[i, 'DATE'] = date.date()

    data_set['DAY'] = days
    data_set = data_set[['DATE', 'DAY', 'CLS']]
    values = data_set[['DAY', 'CLS']]
    values = values.astype('float32')
    # stock_values = data_set.iloc[:, 1:3].values;
    stock_values = data_set[['DAY', 'CLS']]
    stock_values = stock_values.astype('float32')
    data_set.iloc[1525 + data: 1525 + data + 1, :].values;
    # normalize features
    # values = scaler.fit_transform(stock_values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    values[['DAY']] = scaler.fit_transform(stock_values[['DAY']])
    values[['CLS']] = scaler2.fit_transform(stock_values[['CLS']])

    reframed = series_to_supervised(values, 4, 1)
    values = reframed.values

    model = joblib.load('stock-directfn-pre.joblib')
    input_values = values[1525 + data: 1525 + data + 1, :]
    input_values_X, input_values_y = input_values[:, :-1], input_values[:, -1]
    input_values_X = input_values_X.reshape((input_values_X.shape[0], 1, input_values_X.shape[1]))
    predictions = model.predict(input_values_X, batch_size=1)
    predicted_stock_price = scaler2.inverse_transform(predictions)
    Send_value = predicted_stock_price[0][0];
    return str(Send_value);


def run_ctc_model(data):
    data_set = pd.read_csv(r'C:\Users\shanaka\Desktop\final semester\projects V2\stockserver\formatted_128444-CTC.csv')
    data_set = data_set[['DATE', 'OP', 'CLS']]
    days = []

    for i in range(1635):
        s = str(data_set.loc[i, 'DATE'])
        # you could also import date instead of datetime and use that.
        date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        days.append(date.weekday())
        data_set.loc[i, 'DATE'] = date.date()

    data_set['DAY'] = days
    data_set = data_set[['DATE', 'DAY', 'CLS']]
    values = data_set[['DAY', 'CLS']]
    values = values.astype('float32')
    # stock_values = data_set.iloc[:, 1:3].values;
    stock_values = data_set[['DAY', 'CLS']]
    stock_values = stock_values.astype('float32')
    data_set.iloc[1425 + data: 1425 + data + 1, :].values;
    # normalize features
    # values = scaler.fit_transform(stock_values)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    values[['DAY']] = scaler.fit_transform(stock_values[['DAY']])
    values[['CLS']] = scaler2.fit_transform(stock_values[['CLS']])

    reframed = series_to_supervised(values, 4, 1)
    values = reframed.values

    model = joblib.load('stock-ctc-pre.joblib')
    input_values = values[1425 + data: 1425 + data + 1, :]
    input_values_X, input_values_y = input_values[:, :-1], input_values[:, -1]
    input_values_X = input_values_X.reshape((input_values_X.shape[0], 1, input_values_X.shape[1]))
    predictions = model.predict(input_values_X, batch_size=1)
    predicted_stock_price = scaler2.inverse_transform(predictions)
    Send_value = predicted_stock_price[0][0];
    return str(Send_value);


# def get_pre_model():
#     model = joblib.load('stock-pre.joblib')
#     return HttpResponse('model value created')
#
#
# def send_stock(request, value):
#     if (value != 10):
#         return HttpResponse('sell')
#     else:
#         return HttpResponse('sell')


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
