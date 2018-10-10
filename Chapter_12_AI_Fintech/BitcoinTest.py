# Import datetime to operate on dates in the data set
import datetime

# time will help us to get current system date and time
import time

# Matplotlib for creating plots
import matplotlib.pyplot as plt

# Intialize the figure size so that we get large plots
# to visualize well
plt.rcParams['figure.figsize'] = [15, 10]

# We will use pandas to load and clean the data set
import pandas as pd

# Keras will be used to create LSTM network
# we will be using tensorflow backend in Keras 
from keras.layers import Activation, Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential

# MPL tool kit will help us to create special plots
# we will see there use in the end
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

# Finally numpy for algebric calculations 
import numpy as np

# Let's define URL for pull the data from the website
URL = ["https://coinmarketcap.com/currencies/bitcoin/"+
       "historical-data/?start=20130428&end="+
       time.strftime("%Y%m%d")]

# Here we will pull the data from web
bitcoin_market_info = pd.read_html(URL[0])[0]

# Convert the date string to the correct date format
bitcoin_market_info = \
bitcoin_market_info.assign(Date=
                           pd.to_datetime(bitcoin_market_info['Date']))

# When Volume is equal to '-' convert it to 0
# otherwise we will get NaN in the dataset
bitcoin_market_info.loc[bitcoin_market_info['Volume']=="-",'Volume']=0

# Convert to int
bitcoin_market_info['Volume'] = \
bitcoin_market_info['Volume'].astype('int64')

# Sometime coinmarketcap starting returning asterisks in the column names
# so we need to remove those asterisks if present
bitcoin_market_info.columns = \
bitcoin_market_info.columns.str.replace("*", "")

# Look at the first few rows
print(bitcoin_market_info.head())

# We will put 'bt_' to all the columns 
# in case you work with other currencies too
bitcoin_market_info.columns =\
[bitcoin_market_info.columns[0]]+\
['bt_'+i for i in bitcoin_market_info.columns[1:]]

# Let's plot the data set
fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = 
                               {'height_ratios':[3, 1]})
 
# Create labels for x and y axes 
ax1.set_ylabel('Closing Price ($)',fontsize=12)
ax2.set_ylabel('Volume ($ bn)',fontsize=12)
ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
ax2.set_yticklabels(range(10))
 
# Dates as the labels of x-axis 
# We will plot the data for evry 7 month prices.
ax1.set_xticks([datetime.date(i,j,1) 
                for i in range(2013,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) 
                for i in range(2013,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  
                     for i in range(2013,2019) for j in [1,7]])
ax1.plot(bitcoin_market_info['Date'].astype(datetime.datetime),
         bitcoin_market_info['bt_Open'])
ax2.bar(bitcoin_market_info['Date'].astype(datetime.datetime).values, 
        bitcoin_market_info['bt_Volume'].values)
fig.tight_layout()
plt.show()

# Let's store our data frame in another variable
market_info = bitcoin_market_info
    
# Let's get the data from 2016 onwards    
market_info = market_info[market_info['Date']>='2016-01-01']

# Here you can define other coins too
# for example Etherium
coins = 'bt_'
kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']
                                        -x[coins+'Open'])
          /x[coins+'Open']}
market_info = market_info.assign(**kwargs)

# Let's print the data
market_info.head()

# For creating a ML model we will need to have training and test set
# So that we can eavaluate the performance of our model
# We will choose prices before June 2017 as training data
# and prices after that as test data.

# We can not randomely split the data set for training and testing
# as this is a time series data set. time position is very important
split_date = '2017-06-01'

# Let's plot training and testing data together
ax1 = plt.axes()
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax1.set_xticklabels('')
 
# Training data
ax1.plot(market_info[market_info['Date'] < split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] < split_date]['bt_Close'], 
         color='#B08FC7', label='Training')
 
# Testing data
ax1.plot(market_info[market_info['Date'] >= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date'] >= split_date]['bt_Close'], 
         color='#8FBAC8', label='Test')
 
ax1.set_xticklabels('')
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
 
# Plot the data
plt.tight_layout()
ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0., prop={'size': 14})
plt.show()

# We will start with the simplest model 
# trivial lag model: P_t = P_(t-1)
# We will try to visualize the effect of model
# on price prediction

# First we will create date axis for plot
ax1 = plt.axes()
ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax1.set_xticklabels('')
 
# We will plot actual data first 
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= split_date]['bt_Close'].values, label='Actual')
 
# Now the predicted data
ax1.plot(market_info[market_info['Date']>= split_date]['Date'].astype(datetime.datetime),
          market_info[market_info['Date']>= datetime.datetime.strptime(split_date, '%Y-%m-%d') - 
                      datetime.timedelta(days=1)]['bt_Close'][1:].values, label='Predicted')
 
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})
ax1.set_title('Simple Lag Model (Test Set)')
 
fig.tight_layout()
plt.show()

# Let's plot the histogram of price history 
# to check whether the data is following normal distribution
ax1 = plt.axes()
ax1.hist(market_info[market_info['Date']< \
                     split_date]['bt_day_diff'].values, bins=100)
ax1.set_title('Bitcoin Daily Price Changes')
plt.show()

# Let's plot the result of random walk model
np.random.seed(50)

# Extract Mean and Standard deviation of price history 
bt_r_walk_mean, bt_r_walk_sd = \
np.mean(market_info[market_info['Date']< \
                    split_date]['bt_day_diff'].values), \
        np.std(market_info[market_info['Date']< \
                           split_date]['bt_day_diff'].values)

# Let's create a normal distribution using Mean and Standard deviation
bt_random_steps = \
np.random.normal(bt_r_walk_mean, bt_r_walk_sd, 
                (max(market_info['Date']).to_pydatetime() - \
            datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)

# Create axis
ax1 = plt.axes()
ax1.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')\
                       for i in range(12)])
 
# Plot Actual and Predicted data
ax1.plot(market_info[market_info['Date']>= \
                     split_date]['Date'].astype(datetime.datetime),
        market_info[market_info['Date']>= \
                 split_date]['bt_Close'].values, label='Actual')
 
ax1.plot(market_info[market_info['Date']>= \
                     split_date]['Date'].astype(datetime.datetime),
      market_info[(market_info['Date']+ \
                   datetime.timedelta(days=1))>= \
                  split_date]['bt_Close'].values[1:] * 
     (1+bt_random_steps), label='Predicted')
 
ax1.set_title('Single Point Random Walk (Test Set)')
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
 
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, 
           borderaxespad=0., prop={'size': 14})
plt.tight_layout()
plt.show()

# Let's create a list for coin price
bt_random_walk = []

# Here we will create random walk model
for n_step, bt_step in enumerate(bt_random_steps):
    if n_step==0:
        bt_random_walk.append(market_info[market_info['Date']< \
                                          split_date]['bt_Close'].values[0] \
                              * (bt_step+1))        
    else:
        bt_random_walk.append(bt_random_walk[n_step-1] * (bt_step+1))

# Create axis 
ax1 = plt.axes()                
ax1.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')\
                       for i in range(12)])
 
# Plot Actual and Predicted data
ax1.plot(market_info[market_info['Date']>= \
                     split_date]['Date'].astype(datetime.datetime),
         market_info[market_info['Date']>= \
                     split_date]['bt_Close'].values, label='Actual')
ax1.plot(market_info[market_info['Date']>= \
                     split_date]['Date'].astype(datetime.datetime),
         bt_random_walk[::-1], label='Predicted')
 
ax1.set_title('Full Interval Random Walk')
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, \
           borderaxespad=0., prop={'size': 14})
plt.tight_layout()
plt.show()

# Here we will edit the data frame contents
# we will add 'close_off_high' and 'volatility' columns here
coins = 'bt_'
kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- \
                x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
            coins+'volatility': lambda x: (x[coins+'High']-\
                 x[coins+'Low'])/(x[coins+'Open'])}
market_info = market_info.assign(**kwargs)

# As original data frame consist date in descending order
# we need to reverse the order
model_data = market_info[['Date']+\
            [coins+metric for metric in \
             ['Close','Volume','close_off_high','volatility']]]

# Here the sorting take place
model_data = model_data.sort_values(by='Date')
print(model_data.head())

# Now we don't need the date columns anymore
# we will drop it and create a training and test data set
training_set, test_set = model_data[model_data['Date']<split_date]\
, model_data[model_data['Date']>=split_date]

training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

# We need to create samples for input to the LSTM network
# we will create a set of 10 instances as a sample for LSTM
window_len = 10
coin = 'bt_'
norm_cols = [coin+metric for metric in ['Close','Volume']]

# Let's create LSTM training data here
LSTM_training_inputs = []

# We will create multiple samples of defined size (10 in our case)
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_training_inputs.append(temp_set)

# Model output is next price normalised to 10th previous closing price
LSTM_training_outputs = (training_set['bt_Close'][window_len:].values\
                         /training_set['bt_Close'][:-window_len].values)-1

# One sample is look like this
print(LSTM_training_inputs[0])

# To check the model performance we need to create
# a Test set similar as we have created the training set.
LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)

# Similarly for test outputs
LSTM_test_outputs = (test_set['bt_Close'][window_len:].values\
                     /test_set['bt_Close'][:-window_len].values)-1


# I find it easier to work with numpy arrays rather than pandas dataframes
# especially as we now only have numerical data
LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

# Now Its time to create our LSTM model in here
# We will write a method for this
def build_model(inputs, output_size, neurons, 
                activ_func="linear",
                dropout=0.25,weights_path=None):
    
    # Start with creating a Sequential model object
    model = Sequential()
    
    # Create a LSTM layer with defined size
    model.add(LSTM(neurons, input_shape=(inputs.shape[1],
                                          inputs.shape[2])))
    
    # Add some dropout to prevent overfitting of the model
    model.add(Dropout(dropout))
    
    # Add a Dense layer with linear activation
    model.add(Dense(units=output_size))    
    model.add(Activation(activ_func))
    
    # If you have already train weights it can load them
    if weights_path is not None:
        model.load_weights(weights_path)
    
    # Return the compiled model for training    
    return model

# random seed for reproducibility
np.random.seed(202)

# Initialise model architecture
bt_model = build_model(LSTM_training_inputs, 
                       output_size=1, 
                       neurons = 1024)
 
# We will train our model using SGD with ADAM optimizer
loss="mae"
optimizer="adam"
bt_model.compile(loss=loss, optimizer=optimizer)
 
# Train model on data
bt_history = bt_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=100, batch_size=32, verbose=2, 
                            shuffle=True)
 
loss_history = bt_history.history
loss_history = loss_history['loss']
epochs = bt_history.epoch
plt.plot(epochs,loss_history)
plt.show()
 
# After training save the model on disk
bt_model.save('BitcoinPrediction.h5')

# Let's Visualize the model performance by plotting the actual and predicted values
# We will plot the values over a small window 
# It will be quite interesting and new for you guys


# We will the plot and grab axis
fig, ax1 = plt.subplots(1,1)

# Create axis for the plot
ax1.set_xticks([datetime.date(i,j,1)\
                 for i in range(2013,2019) for j in [1,5,9]])
ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y') \
                      for i in range(2013,2019) for j in [1,5,9]])

# Let's plot actual data
ax1.plot(model_data[model_data['Date']< \
        split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['bt_Close'][window_len:], label='Actual')

# Plot predicted data
ax1.plot(model_data[model_data['Date']<\
         split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(bt_model.predict(LSTM_training_inputs))+1) *\
           training_set['bt_Close'].values[:-window_len])[0], 
         label='Predicted')

# Set titles for the plot
ax1.set_title('Training Set: Single Timepoint Prediction')

# Set label for y-axis
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)

# Plot mean absolute error for defined window size
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose\
                        (bt_model.predict(LSTM_training_inputs))+1)-
            (training_set['bt_Close'].values[window_len:])/\
            (training_set['bt_Close'].values[:-window_len]))), 
             xy=(0.75, 0.9),  xycoords='axes fraction',
            xytext=(0.75, 0.9), textcoords='axes fraction')
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, 
           borderaxespad=0., prop={'size': 14})

# Here is the interesting part
# We will create a Zoomed window for a small section of the price history
# to check how well our model fit on the training data
# zoom-factor: 2.52, location: centre
axins = zoomed_inset_axes(ax1, 2.52, loc=10, bbox_to_anchor=(400, 307)) 
axins.set_xticks([datetime.date(i,j,1)\
                   for i in range(2013,2019) for j in [1,5,9]])

# Plot Actual data in the zoomed window
axins.plot(model_data[model_data['Date']< \
            split_date]['Date'][window_len:].astype(datetime.datetime),
         training_set['bt_Close'][window_len:], label='Actual')

# Plot Predicted data in the zoomed window
axins.plot(model_data[model_data['Date']< \
        split_date]['Date'][window_len:].astype(datetime.datetime),
         ((np.transpose(bt_model.predict(LSTM_training_inputs))+1) *\
           training_set['bt_Close'].values[:-window_len])[0], 
         label='Predicted')

# Set axis values 
axins.set_xlim([datetime.date(2017, 2, 15),
                 datetime.date(2017, 5, 1)])
axins.set_ylim([920, 1400])
axins.set_xticklabels('')
mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()

# Plot results on test data
fig, ax1 = plt.subplots(1,1)

# Set axis properties
ax1.set_xticks([datetime.date(2017,i+1,1) for i in range(12)])
ax1.set_xticklabels([datetime.date(2017,i+1,1).strftime('%b %d %Y')\
                       for i in range(12)])

# Create plot of actual values for defined time period
ax1.plot(model_data[model_data['Date']>= \
                    split_date]['Date'][10:].astype(datetime.datetime),
         test_set['bt_Close'][window_len:], label='Actual')

# Load and test the model on test data and create plot
ax1.plot(model_data[model_data['Date']>=\
                     split_date]['Date'][10:].astype(datetime.datetime),
         ((np.transpose(bt_model.predict(LSTM_test_inputs))+1) *\
           test_set['bt_Close'].values[:-window_len])[0], 
         label='Predicted')

# Calculate mean absolute error and plot it
ax1.annotate('MAE: %.4f'%np.mean(np.abs((np.transpose(bt_model.predict(LSTM_test_inputs))\
+1)-(test_set['bt_Close'].values[window_len:])/\
                                        (test_set['bt_Close'].values[:-window_len]))),
             xy=(0.75, 0.9),  xycoords='axes fraction',
             xytext=(0.75, 0.9), textcoords='axes fraction')

ax1.set_title('Test Set: Single Timepoint Prediction',fontsize=13)
ax1.set_ylabel('Bitcoin Price ($)',fontsize=12)
ax1.legend(bbox_to_anchor=(0.1, 1), loc=2, borderaxespad=0., prop={'size': 14})

# Plot the results
plt.show()