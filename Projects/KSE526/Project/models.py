###################################################################################
# Title  : KSE526 project baseline
# Author : hs_min
# Date   : 2020.11.25
###################################################################################
#%%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, TimeDistributed, Softmax, Dot, Bidirectional, Layer, Conv1D, MaxPooling1D, Flatten, RepeatVector, LSTM, Attention, Concatenate, Dense
  
###################################################################
# Model
###################################################################
# https://www.tensorflow.org/tutorials/text/nmt_with_attention?hl=ko
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, values, query) : # 단, key와 value는 같음
        # query shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
class CNNBiLSTMATTN(Model):
    def __init__(self, config):
        super(CNNBiLSTMATTN, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units
        self.attn_units = config.attn_units

        self.conv1d1 = Conv1D(filters = self.filters,
                            kernel_size = self.kernel_size,
                            activation = self.activation,
                            padding = 'causal')

        self.conv1d2 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation,
                            padding = 'causal')
        
        self.lstm1 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state= False, 
                                            recurrent_initializer='glorot_uniform'))

        self.lstm2 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state=True, 
                                            recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate()
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn1 = Dense(50)

        self.conv1d3 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation,
                            padding = 'causal'
                            )

        self.aux_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.5, 
                                        return_sequences=False, return_state = False))

        self.aux_fcn1 = Dense(20)
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

    def call(self, inputs):
        x_, aux1, aux2 = inputs
        # CNN, 
        x = self.conv1d1(x_)
        x = self.conv1d2(x)

        lstm1 = self.lstm1(x)
        lsmt2, forward_h, forward_c, backward_h, backward_c = self.lstm2(lstm1)
        state_h = self.concat([forward_h, backward_h]) # 은닉 상태
        context_vector, attention_weights = self.attention(lsmt2, state_h)
        x = self.fcn1(context_vector)
        
        x_aux1 = self.conv1d3(aux1)
        aux_lstm = self.aux_lstm(x_aux1)
        x_aux1 = self.aux_fcn1(aux_lstm)

        x_aux2 = self.aux_fnc2(aux2)
        x_aux2 = self.aux_flatten(x_aux2)

        x = self.concat([x, x_aux1])
        x = self.concat([x, x_aux2])
        x = self.fcn3(x)
        x = self.fcn4(x)
    
        return x
   
