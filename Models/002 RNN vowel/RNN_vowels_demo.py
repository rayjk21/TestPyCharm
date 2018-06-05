
import MyUtils.RNN_vowels_base as m
# from MyUtils.RNN_vowels_base import *



def data_with(data_settings):
    data_info = {'n_chars': None, 'max_len': 10,
                    'x_dims': 2, 'y_dims': 2,
                    'flags': None, 'max_ahead': 1, 'distance':False,
                    'normalise': False, 'randomise': None
                    }
    data_info.update(data_settings)
    return data_info

def model_with(model_settings):
    model_info = {'create_fn': m.create_model_D_, 'name':"default_model",
                          'dropout':0.0, 'hidden_units':100, 'embedding_size':50,
                          'loss':m.keras.losses.categorical_crossentropy,
                          'mask_zero':True}

    model_info.update(model_settings)
    return model_info



data_info1  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':1 })
data_info2  = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 })
data_info2a = data_with({'n_chars':20000, 'max_len':10,  'max_ahead':2 })
data_info2b = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':3, 'randomise':(0.5, 1)})
data_info2c = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':2, 'randomise':(0.5, 1)})
data_info2d = data_with({'n_chars':20000, 'max_len':15,  'max_ahead':2, 'randomise':(0.5, 1)})
data_info5c = data_with({'n_chars':3000,  'max_len':10,  'max_ahead':2 , 'out_of_range':0.0, 'distance':True , 'allow':"</>", 'filename':"training words.html"})

model8_1_info  = model_with({'name':"8_1_next_char_1",          'data_info':data_info1,    'loss':m.loss_2_stage_K})
model8_2_info  = model_with({'name':"8_2_next_char_2",          'data_info':data_info2 })
model8_2a_info = model_with({'name':"8_2a_next_char_2_20k",     'data_info':data_info2a})
model8_2b_info = model_with({'name':"8_2a_next_char_3_rand20k", 'data_info':data_info2b})
model8_2c_info = model_with({'name':"8_2a_next_char_2_rand20k", 'data_info':data_info2c})


# Build models with different parameters
print(model8_2c_info)
print(model8_2c_info['data_info'])



# Raw Data
print(m.read_file(n=3000))



#################  Demo - build model #####################

model9_info = model_with({'name':"9_demo", 'data_info':data_info1,
                          'hidden_units': 50, 'dropout':0.2, 'embedding_size':50
                           })


m.build_model(model9_info, extract=True, create=False, fit=False)
m.build_model(model9_info, extract=True, create=True, fit=True, epochs=50)


################### Predict next letter ahead ########################


def prediction_chart(model_info):
    m.model_load(model_info) # Set 'model'
    m.load_data(model_info)  # Set 'coder'
    plot_next_letter = lambda ax, text: m.predict_next_letter(model_info, text, ax, printing=True)
    m.myUi.ChartUpdater(plot_Fn = plot_next_letter)

prediction_chart(model9_info)

# Based on predicting just 1 ahead
prediction_chart(model8_1_info)
# r => a/e/i/o/u
# m..o..l..a..r
# q..u..e..s..t..i..o..n

# Predict 2 letters ahead
prediction_chart(model8_2_info)
# r => other letters too
# q..u..e.. - can see next but one letter predicted

# With random noise inserted
prediction_chart(model8_2c_info)








########### Create summary counts for each model #############

def model_summary(model_info):
    m.model_load(model_info) # Set 'model'
    # Load clean data - will need different format of X with 2 or 3 dims depending on model
    X, y, text, _coder = m.load_data(model_info)
    m.pred_counts(model_info,  X, y, n_top=4, n_find=4, results='s')


model_summary(model8_1_info)
model_summary(model8_2_info)
model_summary(model8_2c_info)



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

# Look ahead of 2 with random noise
# - Still good at finding position 0, despite noise
# - with noise, often the related item will now be at position 1
print(model8_2c_info['summary'])
# n_Find         0         1         2         3
# n_Top
# 0       0.271573  0.070701  0.045065  0.032318
# 1       0.113728  0.054021  0.038904  0.026300
# 2       0.088708  0.045918  0.036203  0.025494




