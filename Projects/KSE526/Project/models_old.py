###################################################################################
# Title  : KSE526 project baseline
# Author : hs_min
# Date   : 2020.11.25
###################################################################################
#%%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import RNN, GRU, BatchNormalization, Dropout, TimeDistributed, Softmax, Dot, Bidirectional, Layer, Conv1D, MaxPooling1D, Flatten, RepeatVector, LSTM, Attention, Concatenate, Dense
import tensorflow.keras.backend as K
#%%
###################################################################
# Loss
###################################################################
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred)))

def weighted_root_mean_squared_error(y_true, y_pred):#, w):
    w = 0.2
    mask = tf.cast(tf.less(y_pred, y_true), dtype=tf.float64)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true-y_pred))) + mask * w * (y_true-y_pred)

def last_time_step_rmse(y_true, y_pred):
    return root_mean_squared_error(y_true[:,-1], y_pred[:,-1])


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

# %%
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
                            activation = self.activation)

        self.conv1d2 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        
        self.mp1d = MaxPooling1D(pool_size = 2)        
        
        self.lstm1 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state= False, 
                                            recurrent_initializer='glorot_uniform'))
        # self.rv = RepeatVector(self.n_outputs)
        self.lstm2 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state=True, 
                                            recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate()
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn1 = Dense(50)#, activation='relu')

        self.conv1d3 = Conv1D(filters = self.filters, 
                            kernel_size = self.kernel_size, 
                            activation = self.activation)

        self.aux_lstm = Bidirectional(LSTM(self.lstm_units, dropout=0.5, 
                                        return_sequences=False, return_state = True))

        self.aux_attention = BahdanauAttention(self.lstm_units)
        self.aux_fcn1 = Dense(20)
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1d1(inputs[0])
        x = self.conv1d2(x)
        x = self.mp1d(x)
        # encoder_lstm = self.lstm1(x)
        # encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm1(x)
        # encoder_lstm, forward_h,  backward_h  = self.lstm1(inputs[0])
        # state_h = self.concat([forward_h, backward_h])
        # decoder_input = self.rv(state_h)
        decoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm2(x)
        # decoder_lstm, forward_h, backward_h = self.lstm2(x)
        state_h = self.concat([forward_h, backward_h]) # 은닉 상태
        # state_c = self.concat([forward_c, backward_c])
        context_vector, attention_weights = self.attention(decoder_lstm, state_h)
        x = self.fcn1(context_vector)
        # x = self.dropout(x)
        
        x_aux1 = self.conv1d3(inputs[1])
        aux_lstm, aux_forward_h, aux_forward_c, aux_backward_h, aux_backward_c = self.aux_lstm(x_aux1)
        # aux_state_h = self.concat([aux_forward_h, aux_backward_h]) # 은닉 상태
        # aux_context_vector, aux_attention_weights = self.aux_attention(aux_lstm, aux_state_h)
        x_aux1 = self.aux_fcn1(aux_lstm)

        x_aux2 = self.aux_fnc2(inputs[2])
        x_aux2 = self.aux_flatten(x_aux2)

        # x = self.concat([x, x_aux1, x_aux2])
        x = self.concat([x, x_aux1])
        x = self.concat([x, x_aux2])
        x = self.fcn3(x)
        x = self.fcn4(x)
        
        return x

class BiLSTMATTN(Model):
    def __init__(self, config):
        super(BiLSTMATTN, self).__init__()
        self.n_outputs = config.label_width
        self.filters = config.filters
        self.kernel_size = config.kernel_size
        self.activation = config.activation
        self.lstm_units = config.lstm_units
        self.attn_units = config.attn_units
        
        self.lstm1 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state= False, 
                                            recurrent_initializer='glorot_uniform'))
        # self.rv = RepeatVector(self.n_outputs)
        self.lstm2 = Bidirectional(LSTM(self.lstm_units, dropout=0.1, 
                                            return_sequences = True, return_state=True, 
                                            recurrent_initializer='glorot_uniform'))
        self.concat = Concatenate()
        self.attention = BahdanauAttention(self.lstm_units)
        self.fcn1 = Dense(50)#, activation='relu')

        self.aux_lstm = LSTM(self.lstm_units, dropout=0.2, return_sequences=False)
        self.aux_fcn1 = Dense(20)
        
        self.aux_fnc2 = TimeDistributed(Dense(20))
        self.aux_flatten = Flatten()

        self.fcn3 = Dense(10)
        self.fcn4 = Dense(self.n_outputs, activation='sigmoid')

    def call(self, inputs):
        encoder_lstm = self.lstm1(inputs[0])
        # encoder_lstm, forward_h, forward_c, backward_h, backward_c = self.encoder_lstm(inputs[0])
        # encoder_lstm, forward_h,  backward_h  = self.encoder_lstm(inputs[0])
        # state_h = self.concat([forward_h, backward_h])
        # decoder_input = self.rv(state_h)
        decoder_lstm, forward_h, forward_c, backward_h, backward_c = self.lstm2(encoder_lstm)
        # decoder_lstm, forward_h, backward_h = self.decoder_lstm(encoder_lstm)
        state_h = self.concat([forward_h, backward_h]) # 은닉 상태
        # state_c = self.concat([forward_c, backward_c])
        context_vector, attention_weights = self.attention(encoder_lstm, state_h)
        x = self.fcn1(context_vector)
        # x = self.dropout(x)
        
        x_aux1 = self.aux_lstm(inputs[1])
        x_aux1 = self.aux_fcn1(x_aux1)

        x_aux2 = self.aux_fnc2(inputs[2])
        x_aux2 = self.aux_flatten(x_aux2)

        x = self.concat([x, x_aux1, x_aux2])
        x = self.fcn3(x)
        x = self.fcn4(x)
        
        return x
