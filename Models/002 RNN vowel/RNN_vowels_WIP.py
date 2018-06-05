
from __future__ import print_function

from MyUtils.RNN_vowels_base import *



## Getting warnings from sklearn doing coder.inverse_transform(2)
## DeprecationWarning: The truth value of an empty array is ambiguous.
# Returning False, but in future this will result in an error. Use `array.size > 0` to check that an array is not empty
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#data_path = r"C:\Temp\TestPyCharm\Models\002 RNN vowel"
models_path = r"C:\Users\rkirk\Documents\GIT\Python\TestPyCharm\Models\002 RNN vowel"
data_path = r"C:\Users\rkirk\Documents\GIT\Python\TestPyCharm\Data\Text Data"
alphabet = 'abcdefghijklmnopqrstuvwxyz'

# Global coder that is used by default if None is set for a model (e.g. if it has been reloaded)
_coder = None


# Define models : Looking ahead & Random noise ##########################

data_info0  = data_with({'n_chars':3000,  'max_len':10,  'flags':0,    })
data_info1  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':1 })
data_info1b = data_with({'n_chars':20000, 'max_len':10,  'max_ahead':1 })
data_info1c = data_with({'n_chars':20000, 'max_len':10,  'max_ahead':1, 'randomise':(0.5, 1)})
data_info2  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 })
data_info2a = data_with({'n_chars':20000, 'max_len':10,  'max_ahead':2 })
data_info2b = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':3, 'randomise':(0.5, 1)})
data_info2c = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':2, 'randomise':(0.5, 1)})
data_info2d = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':2, 'randomise':(0.5, 1)})
data_info2e = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':2 })
data_info3  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':3 })
data_info4  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':4 })
data_info5  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 , 'distance':True })
data_info5a = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':4 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})
data_info5b = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':4 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})
data_info5c = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})
data_info5d = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})
data_info5e = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':4 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})


model8_0_info  = model_with({'name':"8_0_next_char_old",        'data_info':data_info0 })
model8_1_info  = model_with({'name':"8_1_next_char_1",          'data_info':data_info1,    'loss':loss_2_stage_K})
model8_1b_info = model_with({'name':"8_1b_next_char_1",         'data_info':data_info1b})
model8_1c_info = model_with({'name':"8_1c_next_char_1_rnd",     'data_info':data_info1c})
model8_2_info  = model_with({'name':"8_2_next_char_2",          'data_info':data_info2 })
model8_2a_info = model_with({'name':"8_2a_next_char_2_20k",     'data_info':data_info2a})
model8_2b_info = model_with({'name':"8_2a_next_char_3_rand20k", 'data_info':data_info2b})
model8_2c_info = model_with({'name':"8_2a_next_char_2_rand20k", 'data_info':data_info2c})
model8_2d_info = model_with({'name':"8_2d_next_char_1_rnd_40do",'data_info':data_info2d, 'dropout':0.4})
model8_2d6_info= model_with({'name':"8_2d_next_char_1_rnd_60do",'data_info':data_info2d, 'dropout':0.6})
model8_2e_info = model_with({'name':"clean data",               'data_info':data_info2e, 'dropout':0.4})
model8_3_info  = model_with({'name':"8_3_next_char_3",          'data_info':data_info3 })
model8_4_info  = model_with({'name':"8_4_next_char_4",          'data_info':data_info4 })

model8_5_info  = model_with({'name':"8_5_distance_to",          'data_info':data_info5 , 'dropout':0.4, 'loss':keras.losses.mse})
model8_5a_info = model_with({'name':"8_5a_distance_to_4",       'data_info':data_info5a, 'dropout':0.4, 'loss':keras.losses.mse})
# Too simplistic to just model distance as 1/d
# E.g. if predicting what follows 't' if will be averaging cases where 'e' is next (y=1), or position 2 (y=0.5) or not at all (y=0.0)
# - in cases where X doesn't occur, the penalty is v.high for predicting say 1, when the truth is 0
# - e.g. for "<Text>" the T is never predicted as 90% of the time it never occurs
# - Better to have a 2 stage prediction:
#   - Predict 1/0 for if X occurs at all (with a binary-cross-entropy loss)
#   - In cases where thruth is 0, no further loss
#   - In cases where thruth is 1, mse loss based on prediction of where.
model8_5b_info = model_with({'name':"8_5b_2stage_distance_to_4",        'data_info':data_info5b, 'dropout':0.4, 'loss':loss_2_stage_K})
model8_5c_info = model_with({'name':"8_5c_2stage_distance_to_4",        'data_info':data_info5c, 'dropout':0.4, 'loss':loss_2_stage_K})
model8_5d_info = model_with({'name':"8_5d_2stage_distance_to_4",        'data_info':data_info5d, 'dropout':0.4, 'loss':loss_2_stage_K})
model8_5e_info = model_with({'name':"8_5e_2stage_distance_to_4",        'data_info':data_info5e, 'dropout':0.4, 'loss':loss_2_stage_K})







#### Run build process for each model #####

#  build_model(model8_0_info, create=True)
#  build_model(model8_1_info, create=True, epochs = 10)
#  build_model(model8_1b_info, create=True)     # Look ahead of 1 - no noise
#  build_model(model8_1c_info, create=True)     # Look ahead of 1 - with noise
#  build_model(model8_2_info, create=True)
#  build_model(model8_2a_info, create=True)
#  build_model(model8_2b_info, create=True)
#  build_model(model8_2c_info, create=True)     # Look ahead of 2 - with noise of 1
#  build_model(model8_2d_info, create=True)     # Look ahead of 2 - with noise & dout
#  build_model(model8_2d6_info, create=True)     # Look ahead of 2 - with noise & dout
#  build_model(model8_3_info, create=True)
#  build_model(model8_4_info, create=True)
#  build_model(model8_5_info, create=False, fit=True, epochs=100)
#
#  build_model(model8_5a_info, create=True)
#  build_model(model8_5a_info, create=False, epochs=25)
#  build_model(model8_5b_info, create=True, epochs=200)
#  build_model(model8_5c_info, create=False, epochs=100)
#  build_model(model8_5d_info, create=True, epochs=200)
#  build_model(model8_5e_info, create=True, epochs=200)






##### Reload previous models
# model_load(model8_1_info)
# model_load(model8_2_info)
# model_load(model8_2a_info)
# model_load(model8_2c_info)
# model_load(model8_4_info)
# model_load(model8_5_info)
# model_load(model8_5a_info)
# model_load(model8_5c_info)







##### View charts for saved models  ####
# Run the model_info above to define model

def prediction_chart(model_info):
    model_load(model_info) # Set 'model'
    load_data(model_info)  # Set 'coder'
    plot_next_letter = lambda ax, text: predict_next_letter(model_info, text, ax, printing=True)
    myUi.ChartUpdater(plot_Fn = plot_next_letter)



# Reload previous models
# q..u..e..s..t..i..o..n

prediction_chart(model8_1_info)
prediction_chart(model8_2_info)
prediction_chart(model8_2a_info)
prediction_chart(model8_2c_info)
prediction_chart(model8_4_info)
prediction_chart(model8_5_info)
prediction_chart(model8_5a_info)
prediction_chart(model8_5c_info)
# => Much more mixed predictions: e.g. after r is not just vowels
# => Much more mixed predictions: e.g. q is followed by a,e,i as well as u


# When using randomised data:
# - becomes hard to see real patterns through the noise


# model8_4_info works well with max_ahead=4, but not using distances, just 1/0

# Can't get actual distance values to work well
# - Perhaps need to predict tuple (prob, dist) and have a better 2-stage loss function



# Use with model5
# Try: "p r e s e r v e"
# The try from "s" -> i..n..g..l..e
# t..o..o..t..h
# m..o..l..a..r
# t..w..e..n..t..y
raw_data, coder = get_raw_data(n_chars=3000)
model5 = model_load("Keep\model_5_2b_next_letter_40Dropout_v165")
model5 = model_load("Keep\model_5_next_letter_final")
model_info = (model5, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter(model_info, text, ax, printing=True, coder=coder)
myUi.ChartUpdater(plot_Fn = plot_next_letter)










########### Create summary counts for each model #############

def model_summary(model_info):
    model_load(model_info) # Set 'model'
    # Load clean data - will need different format of X with 2 or 3 dims depending on model
    X, y, text, _coder = load_data(model_info)
    pred_counts(model_info,  X, y, n_top=4, n_find=4, results='s')


model_summary(model8_1_info)
model_summary(model8_2e_info)
model_summary(model8_4_info)
model_summary(model8_5_info)
model_summary(model8_5a_info)
model_summary(model8_5b_info)
model_summary(model8_5d_info)
model_summary(model8_0_info)
model_summary(model8_1_info)
model_summary(model8_1b_info)
model_summary(model8_1c_info)
model_summary(model8_2_info)
model_summary(model8_2a_info)
model_summary(model8_2b_info)
model_summary(model8_2c_info)
model_summary(model8_2d_info)
model_summary(model8_2d6_info)
model_summary(model8_3_info)
model_summary(model8_4_info)   # Works quite well


pred = predict(model8_5_info, 't', flag=None, max_len=10, printing=True)


# Print summaries saved above
print(model8_1_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.456943  0.034954  0.024786  0.015253
# 1       0.069908  0.089291  0.035272  0.021290
# => High prediction 45% for position 0, but poor after that

print(model8_2_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.265836  0.172378  0.023884  0.010644
# 1       0.134735  0.106698  0.039720  0.014798
# => Good prediction 26%, 16% for positions 0 and 1, but poor after that

print(model8_3_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.221060  0.138816  0.069646  0.011172
# 1       0.115046  0.092465  0.066793  0.019729
# => Good prediction 26%, 16% for positions 0 and 1, and better 6.9% for 2




# No noise - look ahead of 1
# => Really focusses on next value and gets it well
print(model8_1b_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.490836  0.026707  0.017543  0.013877
# 1       0.080017  0.073000  0.032468  0.022884
# 2       0.037390  0.046292  0.027702  0.019271


# With noise - look ahead of 1
# - still manages to make a decent prediction overall
# Low frequency relationships (e.g. q...) are lost
# But common ones like "th..e" still picked up.
print(model8_1c_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.447218  0.066692  0.020492  0.013827
# 1       0.056922  0.079690  0.031545  0.023804
# 2       0.034112  0.060440  0.027736  0.019622

# Prediction below of clean data
# n_Find         0         1         2         3
# n_Top
# 0       0.313852  0.058677  0.047393  0.026926
# 1       0.120778  0.053696  0.036576  0.024280
# 2       0.072763  0.045837  0.030661  0.022879



# Model - look ahead of 2
# => Attention spread over 2 values, so not so good at pos=0
print(model8_2a_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.299817  0.158680  0.018591  0.008641
# 1       0.115737  0.120712  0.035350  0.018329
# 2       0.031684  0.030636  0.046347  0.020162


# Look ahead of 2 with random noise
# - Still good at finding position 0, despite noise
# - with noise, often the related item will now be at position 1
print(model8_2c_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.271573  0.070701  0.045065  0.032318
# 1       0.113728  0.054021  0.038904  0.026300
# 2       0.088708  0.045918  0.036203  0.025494


print(model8_2d_info['summary'])
# Tested on clean data. having built with drop out
# - is slightly better than no dropout
# - chart of next is much cleaner
# n_Find         0         1         2         3
# n_Top
# 0       0.310582  0.037420  0.028083  0.020483
# 1       0.156702  0.047119  0.027432  0.021714
# 2       0.124855  0.036769  0.031630  0.016358


# Not much different with 60% dropout - worse if anything
print(model8_2d6_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.309816  0.040591  0.031479  0.022228
# 1       0.148695  0.042455  0.030581  0.019605
# 2       0.116595  0.049772  0.026715  0.019122










##################   Earlier Models #############################################



############## Build Models - individual specification, without model_info ###################

max_len = 10
h = []


# First simple model
# - need to train for 100+ epochs, even though accuracy doesn't change much
model_name="vowel1a2"
raw_data, coder = get_raw_data(n_chars=3000)
X,y = get_xy_data(raw_data, coder, max_len=max_len)
(X_train,  y_train), (X_test,y_test) = MyNn.splitData(X, y, [80,20])
model1 = create_model_A(hidden_units=30, input_shape = (max_len, 1), model_name=model_name)
h += model_fit(model1, X, y, epochs=2, batch_size=1)
plt.plot(h)


# More units, more input
model_name="vowel1b"
raw_data, coder = get_raw_data(n_chars=6000)
X,y = get_xy_data(raw_data, coder, max_len=max_len)
model1b = create_model_A(hidden_units=50, input_shape = (max_len, 1))
h += model_fit(model1b, X, y, epochs=2, batch_size=1, model_name=model_name)




# Doesn't work so well with 2 layers
# - Perhaps due to prediction from dummy timestates being treated as genuine
model_name="vowel2"
raw_data, coder = get_raw_data(n_chars=6000)
X,y = get_xy_data(raw_data, coder, max_len=max_len)
model2 = create_model_B(hidden_units=[30,20], input_shape = (max_len, 1), model_name=model_name)
h += model_fit(model2, X, y, epochs=100, batch_size=1)




# Use embedding
# - works well for first few places, then predicts 0
model_name="vowel3"
raw_data, coder = get_raw_data(n_chars=6000)
num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2)
model3 = create_model_C(hidden_units=30, max_len=max_len, num_categories=num_cats, embedding_size=7, model_name=model_name)
h += model_fit(model3, X, y, epochs=100, batch_size=1, model_name=model_name)




# Use embedding and predict vowel & space
# - works well
model_name="4.1_vowel_space"
h = []
raw_data, coder = get_raw_data(n_chars=6000)
num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=2)
model4 = create_model_D(hidden_units=50, max_len=max_len, num_categories=num_cats, embedding_size=20, num_flags=2, model_name=model_name)
h += model_fit(model4, X, y, epochs=2, batch_size=5)
plt.plot(h)



# Use embedding and predict vowel & space
# - Works well
h=[]
model_name="4b_vowel_space"
raw_data, coder = get_raw_data(n_chars=20000)
num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=2)
num_flags = y.shape[2] - 1
model4b = create_model_D(hidden_units=100, max_len=max_len, num_categories=num_cats, embedding_size=30, num_flags=num_flags, model_name=model_name)
h += model_fit(model4b, X, y, epochs=1000, batch_size=3)
model4b.evaluate(X, y, batch_size=50)
plt.plot(h)





# Use embedding to predict next letter
# - Model 5_ used only 3000 obs, 5.1_ used 20000
# - model 5_ shows interesting pattern with k:
#       - k at the start followed by n
#       - ok is v.different

h=[]
model_name, dropout =("5_3a_next_letter_NoDropout", 0.0)
model_name, dropout =("5_3b_next_letter_40Dropout", 40.0)
model_name, dropout =("5_2b_next_letter_40Dropout", 40.0) # 200000
raw_data, coder = get_raw_data(n_chars=20000)
raw_data, coder = get_raw_data(n_chars=200000, filename="training words.txt")

num_cats = len(coder.classes_)+1
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
num_flags = y.shape[2] - 1
model5 = create_model_D(hidden_units=100, max_len=max_len, num_categories=num_cats, embedding_size=50, num_flags=num_flags, model_name=model_name, mask_zero=True, dropout=dropout)
h += model_fit(model5, X, y, epochs=200, batch_size=5)
model5.evaluate(X, y, batch_size=5)
model_evaluate(model5, X, y, stateful=False, printing=False, mask_zero=True)
model_evaluate(model5, X, y, stateful=False, printing=True, mask_zero=True, n_obs=3)

plt.plot(h)





# Stateful
h=[]
model_name="6_Stateful_next_letter"
raw_data, coder = get_raw_data(n_chars=3000)
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=3, flags=0)
num_flags = y.shape[2] - 1
model6 = create_model_E(hidden_units=100, num_flags=num_flags, model_name=model_name)
#tr_loss, tr_acc = model_fit_stateful(model6, X, y, epochs=1000)
h += model_fit_stateful(model6, X, y, epochs=1000)
plt.plot(h)


model_evaluate(model6, X, y, stateful=True, printing=False, mask_zero=True)
model_evaluate(model6, X, y, stateful=True, printing=True, coder=coder, n_obs=3)
model_evaluate(model6, X, y, stateful=True, printing=False, coder=coder)

# Overstates accuracy when include the padded values
model_evaluate(model6, X, y, stateful=True, printing=False, mask_zero=False)





# Stateful with Embedding
# model 7   - based on   3000 obs and  50 epochs, 10 embeddings
# model 7.1 - based on  20000 obs and 250 epochs, 10 embeddings
# model 7.2 - based on 200000 obs and 250 epochs, 10 embeddings
# Much slower using stateful: 1 epoch with 200k obs takes 5 mins, compared to 30secs for model5
h=[]
model_name="7_2_Stateful_emb_next_letter"
raw_data, coder = get_raw_data(n_chars=20000)
raw_data, coder = get_raw_data(n_chars=200000, filename="training words.txt")
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
num_flags = y.shape[2] - 1
model7 = create_model_F(hidden_units=100, num_flags=num_flags, embedding_size=10, model_name=model_name)
h += model_fit_stateful(model7, X, y, epochs=5)
plt.plot(h)

model_evaluate(model7, X, y, stateful=True, printing=False)
model_evaluate(model7, X, y, stateful=True, printing=True, coder=coder, n_obs=3)

























#############  Interactively see most likely next letters - earlier models #####################


# Set model info to Model, x_dims, normalise (2 if using embeddings)
model_info = (model1, 3, False)
model_info = (model1b, 3, False)
model_info = (model3, 2, False)
model_info = (model4, 2, False)
model_info = (model4b, 2, False)
model_info = (model6, 3, False)
model_info = (model7, 2, False)



# Note: you need a coder
raw_data, coder = get_raw_data(n_chars=3000)

# predict_next_letter(model_info, 'th')

# Use with model5
# Try: "p r e s e r v e"
# The try from "s" -> i..n..g..l..e
# t..o..o..t..h
# m..o..l..a..r
# t..w..e..n..t..y
model5 = model_load("Keep\model_5_next_letter_final")
model_info = (model5, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter(model_info, text, ax, printing=True, coder=coder)
myUi.ChartUpdater(plot_Fn = plot_next_letter)

# With model5.2
# Try: p..r..e..s..i..d..e..n..t
# Try: r : only followed by vowels
# can do r..n..a caused by international
# - xecutions



model6 = model_load("Keep\model_6_Stateful_next_letter_final")
model_info = (model6, 3, False)
plot_next_letter = lambda ax, text: predict_next_letter_stateful(model_info, text, ax, coder=coder)
myUi.ChartUpdater(plot_Fn = plot_next_letter)



model7  = model_load("Keep\model_7_Stateful_emb_next_letter_final")
model_info = (model7, 2, False)
plot_next_letter = lambda ax, text: predict_next_letter_stateful(model_info, text, ax, printing=True, coder=coder)
myUi.ChartUpdater(plot_Fn = plot_next_letter)






# Flag None = category index
# Flag 1 = prob(vowel) following
# Flag 2 = prob(space) following
# N.b. For model5, with all letters, flag 1 is ' ' and 2 is 'a'
flag = 1
flag = 2
flag = None


# Prob of next letter
model5_info = (model5, 2, False)
r=predict_each(model5_info, "abcdefghijklmnopqrstuvwxyz", '', None)


model41 = model_load("Keep\model_4.1_vowel_space_final")
model41_info = (model41, 2, False)
# Probability of vowel next
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", '', 1)
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", 't', 1)
# Prob of space
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", '', 2)
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", 't', 2)
r=predict_each(model41_info, "abcdefghijklmnopqrstuvwxyz", 'th', 2)



# Prob of v
model1b = model_load("Keep\model_vowel1b_final")
model1b_info = (model1b, 3, False)
r=predict_each(model1b_info, "abcdefghijklmnopqrstuvwxyz", '', 1)


model1a = model_load("Keep\model_vowel1a_final")
model1a_info = (model1a, 3, False)
r=predict_each(model1a_info, "abcdefghijklmnopqrstuvwxyz", '', 1)
r=predict_each(model1a_info, "abcdefghijklmnopqrstuvwxyz", '', 2)


# Predict prob of vowel (flag=1) after each letter in word
predict_positions(model41_info, "hello", 1, coder=coder)

predict_positions(model_info, "formatter", 2)
predict_positions(model_info, "evory")
predict_positions(model_info, "phoning")
predict_positions(model_info, "thrashing", flag)
predict_positions(model_info, "crashing")
predict_positions(model_info, "shing")
predict_positions(model_info, "bananaman")
predict_positions(model_info, "crashing", flag)
predict_positions(model_info, "england", flag)
predict_positions(model_info, "ingland", flag)
predict_positions(model_info, "ingland", flag)
predict_positions(model_info, "answer", flag)







################ Predicting & Evaluating - earlier models ##################

# Best model is model5_2


# Import one model
model1a = model_load("Keep\model_vowel1a_final")
model1b = model_load("Keep\model_vowel1b_final")
model3  = model_load("Keep\model_vowel3_final")
model4  = model_load("Keep\model_4.1_vowel_space_final")
model4b = model_load("Keep\model_4b_vowel_space_final")
model5  = model_load("Keep\model_5_next_letter_final")
model52  = model_load("Keep\model_5_2_next_letter_final")
model52b = model_load("Keep\model_5_2b_next_letter_40Dropout_v165")
model6  = model_load("Keep\model_6_Stateful_next_letter_final")
model7  = model_load("Keep\model_7_Stateful_emb_next_letter_final")
model71  = model_load("Keep\model_7_1_Stateful_emb_next_letter_final")
model72  = model_load("Keep\model_7_2_Stateful_emb_next_letter_v80")


model52.summary()
model6.summary()
model72.summary()



### Load in data for use in evaluation
x_dim = 2 # Use 3 if no embeddings in model (e.g. model 6)
max_len=10
raw_data, coder = get_raw_data(n_chars=20000)
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=x_dim, flags=0)


model_evaluate(model52, X, y, stateful=False)
# Model 'Keep\model_5_2_next_letter_final' : loss 1.4773, accuracy 0.6383

X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=3, flags=0)
model_evaluate(model6, X, y, stateful=True)
# Model 'Keep\model_6_Stateful_next_letter_final' : loss 3.1114, accuracy 0.4097

X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
model_evaluate(model72, X, y, stateful=True)
# Model 'Keep\model_7_2_Stateful_emb_next_letter_v80' : loss 2.7270, accuracy 0.4485


max_len=10
raw_data, coder = get_raw_data(n_chars=3000)
X,y = get_xy_data(raw_data, coder, max_len=max_len, normalise=False, x_dims=2, flags=0)
model5 = model_load("Keep\model_5_2b_next_letter_40Dropout_v165")
model5 = model_load("Keep\model_5_next_letter_final")
# See whether the Top predictions are Found, given diffent length Prefix of sequence
s, c, d = pred_counts(model5, X, y, n_top=3, n_obs=1, coder=coder, results = 'scd')
s, c    = pred_counts(model5, X, y, n_top=4, n_find=4, results='sc')
p       = pred_counts(model5, X, y, n_top=4, n_find=4, results='p')

# 33% of Top predictions were found in the next position, 4.6% at position 1
# 10% of 2nd predictions were found in the next position, 6.1% at position 1
# n_Find         0         1         2         3
# n_Top
# 0       0.332552  0.048922  0.030466  0.022791
# 1       0.104289  0.061050  0.038727  0.022967
# 2       0.068022  0.047340  0.029060  0.022322
# 3       0.048922  0.068549  0.032048  0.021971

# Accuracy increases based on the number of preceding characters
pct_by_pfx = p[p['n_Top']==0][['n_Pfx','Pct0']]
plt.bar(pct_by_pfx['n_Pfx'], pct_by_pfx['Pct0'])
plt.show


# The 3000  data was used in training & the No Dropout model does best on this
# The 20000 data was new & the Dropout model does better on this as it has learned more general features

# raw_data, coder = get_raw_data(n_chars=3000)     # raw_data, coder = get_raw_data(n_chars=20000)
# model_5_2b_next_letter_40Dropout_v165            # model_5_2b_next_letter_40Dropout_v165
# n_Find         0         1         2         3   # n_Find         0         1         2         3
# n_Top                                            # n_Top
# 0       0.387183  0.045280  0.026861  0.020721   # 0       0.414216  0.029513  0.024221  0.018111
# 1       0.098235  0.056408  0.034536  0.020721   # 1       0.114233  0.057553  0.035077  0.020184
# 2       0.076746  0.038373  0.033001  0.017268   # 2       0.065463  0.040532  0.031586  0.020839
# 3       0.053722  0.035303  0.026477  0.029163   # 3       0.044242  0.036059  0.027331  0.020839

# model_5_next_letter_final                        # model_5_next_letter_final
# n_Find         0         1         2         3   # n_Find         0         1         2         3
# n_Top                                            # n_Top
# 0       0.510204  0.011303  0.008791  0.010047   # 0       0.332552  0.048922  0.030466  0.022791
# 1       0.064992  0.086028  0.032967  0.016954   # 1       0.104289  0.061050  0.038727  0.022967
# 2       0.034537  0.053689  0.027943  0.011303   # 2       0.068022  0.047340  0.029060  0.022322
# 3       0.021350  0.066248  0.028885  0.014757   # 3       0.048922  0.068549  0.032048  0.021971
























