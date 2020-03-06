from collections import Counter
from math import pi

import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import (ColumnDataSource, DataTable, NumberFormatter,
                          RangeTool, StringFormatter, TableColumn,)
from bokeh.palettes import Spectral11
from bokeh.plotting import figure
from bokeh.sampledata.autompg2 import autompg2 as mpg
from bokeh.sampledata.stocks import AAPL
from bokeh.transform import cumsum

# Timeseries
import math
import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D


n_sample = 1


# Load the data  
frame = pd.read_csv('dataframe1.csv')
# formated data
data = np.hstack((frame.values.astype('float32')[:, :-2], frame.values.astype('float32')[:, [-2]]))

# normalize data 
scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(data)

iput, oput = scaled[:,:-1], scaled[:,[-1]]

# data split
train_x, test_x, train_y, test_y = train_test_split(iput, oput, test_size=0.5, shuffle=True)

# reshape data for model
train_x = train_x.reshape(n_sample, len(train_x), iput.shape[1])
test_x = test_x.reshape(n_sample, len(test_x), iput.shape[1])

train_y = train_y.reshape(n_sample, 31912)
test_y = test_y.reshape(n_sample, 31912)

gc.collect()

n_in, n_out, n_feature = train_x.shape[1], train_y.shape[1], train_x.shape[2]

# training model
model = Sequential()
model.add(Conv1D(
        filters=2,
        kernel_size=7, 
        activation='relu', 
        input_shape=(n_in, n_feature)
    )
)
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) #0.2 return better
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(n_out))
model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae'])


#model.save_weights(str('depth_weight1.212')+'.h5')
#model.load_weights('depth_weight1.253.h5')

mdl = model.fit(
        train_x, train_y, batch_size=32, validation_data=(test_x, test_y),
        epochs=30, shuffle=False, verbose=2)

pyplot.figure(plot_height=110, tools="", toolbar_location=None, #name="line",
           x_axis_type="datetime")
pyplot.plot(mdl.history['loss'], label='Train')
pyplot.plot(mdl.history['val_loss'], label='Test')
pyplot.legend()
pyplot.show()

layout = column(sizing_mode="scale_width", name="line")

curdoc().add_root(layout)

# Donut chart



curdoc().add_root()

# Bar chart



curdoc().add_root()

# Table


curdoc().add_root()

# Setup

curdoc().title = "Bokeh Dashboard"
curdoc().template_variables['stats_names'] = ['users', 'new_users', 'time', 'sessions', 'sales']
curdoc().template_variables['stats'] = {
    'users'     : {'icon': 'user',        'value': 11200, 'change':  4   , 'label': 'Total Users'},
    'new_users' : {'icon': 'user',        'value': 350,   'change':  1.2 , 'label': 'New Users'},
    'time'      : {'icon': 'clock-o',     'value': 5.6,   'change': -2.3 , 'label': 'Total Time'},
    'sessions'  : {'icon': 'user',        'value': 27300, 'change':  0.5 , 'label': 'Total Sessions'},
    'sales'     : {'icon': 'dollar-sign', 'value': 8700,  'change': -0.2 , 'label': 'Average Sales'},
}
