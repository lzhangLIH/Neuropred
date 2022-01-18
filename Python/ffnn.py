
import tensorflow as tf
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import time
from sksurv.metrics import concordance_index_censored
from sksurv.metrics_mod import cumulative_dynamic_auc
from sksurv.metrics import concordance_index_ipcw
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")


# to get reproducible results
seed = 123

tf.random.set_seed(seed)
np.random.seed(seed)
        

def custom_loss_breslow(events, times, predictions):  
    events = tf.reshape(events, (-1,))
    times = tf.reshape(times, (-1, ))
    predictions = tf.reshape(predictions, (-1, ))
    
    idx = tf.argsort(-times)
    events = tf.cast(tf.gather(events, idx), tf.float32)
    times = tf.gather(times, idx)
    predictions = tf.gather(predictions, idx)
    n_events = tf.maximum(1.0, tf.reduce_sum(events))
   
    hr = tf.math.exp(predictions)
    log_risk = tf.math.log(tf.cumsum(hr))
    
    unique_values, segment_ids = tf.unique(times)
    loss_s2_v = tf.math.segment_max(log_risk, segment_ids)
    loss_s2_count = tf.math.segment_sum(events, segment_ids)
    loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
    loss_s1 = tf.reduce_sum(tf.multiply(predictions, events))
    loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), n_events)

    return loss_breslow

num_var = 91
class TestModel(tf.keras.Model):
    train_step_signature = [
        tf.TensorSpec(shape=(None, num_var), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)  
    ]
        
    def __init__(self, num_blocks=1, hidden_size=32, rate=0.1, l2=1e-3, activation="relu", loss=custom_loss_breslow,
                 learning_rate=1e-4):
        super(TestModel, self).__init__()

        self._blocks = tf.keras.Sequential()     
        for i in range(num_blocks):
            self._blocks.add(tf.keras.layers.Dense(hidden_size, activation=None, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=l2)))
            self._blocks.add(tf.keras.layers.BatchNormalization())
            self._blocks.add(tf.keras.layers.Activation(activation))
            self._blocks.add(tf.keras.layers.Dropout(rate))
        
        self.final_layer = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=l2)) ####
        self.loss_fn = loss

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')

        self.train_ci = tf.keras.metrics.Mean(name='train_ci')
        self.val_ci = tf.keras.metrics.Mean(name='val_ci')
        self.reset(learning_rate)
        
    def reset(self, learning_rate = 1e-4):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_losses, self.val_losses = [], []
        self.train_cis, self.val_cis = [], []
        self.current_epoch = 0
        
    def call(self, x, training=None):
        out = self._blocks(x, training)
        out = self.final_layer(out)
         
        return out
    
    @tf.function(input_signature=train_step_signature)
    def train_step(self, x, e, t):
        with tf.GradientTape() as tape:        
            predictions = self(x, True)
            
            loss = self.loss_fn(e, t, predictions)
            loss = loss + tf.reduce_sum(self.losses)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) 
        
        self.train_loss(loss)
        return predictions
        
    @tf.function(input_signature=train_step_signature)
    def val_step(self, x, e, t):
        predictions = self(x, False)
        
        loss = self.loss_fn(e, t, predictions)
        loss = loss + tf.reduce_sum(self.losses)
           
        self.val_loss(loss)
        return predictions
        
    def evaluate(self, ds_test):
        predictions, events, times = [], [], []
        for i, batch in enumerate(ds_test):
            x, e, t = batch
            p = model(x, False)
                
            predictions = predictions + [z[0] for z in p.numpy()]
            events = events + [z[0] for z in e.numpy()]
            times = times + [z[0] for z in t.numpy()]
        
        predictions = np.array(predictions)
        events = np.array(events)
        times = np.array(times)
        
        # cindex = concordance_index_censored(events.astype(np.bool), times, np.exp(predictions))
        # return cindex
        return predictions
    
    def cum_auc(self, ds_test, y_train, y_test):
        time_auc = np.array([2,4,6,8,10,12])
        predictions = []
        for i, batch in enumerate(ds_test):
            x, e, t = batch
            p = model(x, False)

            predictions = predictions + [z[0] for z in p.numpy()]

        predictions = np.array(predictions)

        cum_auc = cumulative_dynamic_auc(survival_train=y_train, survival_test=y_test, estimate=np.exp(predictions),
                                         times=time_auc)

        return cum_auc

    def ci_ipcw(self, ds_test, y_train, y_test):
        predictions = []
        for i, batch in enumerate(ds_test):
            # print(i)
            x, e, t = batch
            # print(x.shape)
            p = model(x, False)

            predictions = predictions + [z[0] for z in p.numpy()]

        predictions = np.array(predictions)

        ipcw = concordance_index_ipcw(survival_train=y_train, survival_test=y_test, estimate=np.exp(predictions))

        return ipcw
    
    def validate(self, ds_val):
        for (i, batch) in enumerate(ds_val):
            x, e, t = batch
            p = self.val_step(x, e, t)
            try:
                ci = concordance_index_censored(tf.cast(tf.reshape(e, (-1,)), tf.bool), tf.reshape(t, (-1,)), tf.math.exp(tf.reshape(p, (-1, ))))
            except:
                ci = [0]
                
            self.val_ci(ci[0])

    def train(self, ds_train, ds_val=None, max_epochs=200, verbose=0):   
        for epoch in range(self.current_epoch, max_epochs):
            self.train_loss.reset_states()
            self.train_ci.reset_states()
            self.val_loss.reset_states()
            self.val_ci.reset_states()
            
            start = time.time()
            
            for (i, batch) in enumerate(ds_train):
                x, e, t = batch
                p = self.train_step(x, e, t)
                try:
                    ci = concordance_index_censored(tf.cast(tf.reshape(e, (-1,)), tf.bool), tf.reshape(t, (-1,)), tf.math.exp(tf.reshape(p, (-1, ))))
                except:
                    ci = [0]
                    
                self.train_ci(ci[0])
            
                if (verbose > 0) & (i % 50 == 0):
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, i, self.train_loss.result()))
        
            self.train_losses.append(self.train_loss.result().numpy()) 
            self.train_cis.append(self.train_ci.result().numpy())

            if ds_val is not None:
                self.validate(ds_val)
                self.val_losses.append(self.val_loss.result().numpy())    
                self.val_cis.append(self.val_ci.result().numpy())
            
            if verbose > 0:
                print ('Epoch {} Train Loss {:.4f} Val Loss {:.4f}'.format(epoch + 1, 
                           self.train_loss.result(), 
                           self.val_loss.result()))
                
                print ('Time taken for 1 epoch: {:.4f} secs\n'.format(time.time() - start))
            
            self.current_epoch += 1

# list of predictors

var = list()

# data data frame
dat_ = pd.DataFrame()


BATCH_SIZE = 512
BUFFER_SIZE = 4096

def generator(df):   
    for idx, batch in df.iterrows():
        x = (batch[var]) ##########
        #x = (x - offset) / scale
        yield (x.values,
               np.array(batch["event"]).reshape((-1)),
               np.array(batch["timeYear"]).reshape((-1)))

dl_metrics = pd.DataFrame(columns=("imp","cidx_harrell_test",
                                    "cidx_uno_test", "cidx_uno_lower","cidx_uno_upper"))


# i = 1
for i in range(1,41):
    imp = 'imp'+str(i)
    print(imp)
    
    df_ = dat_.loc[dat_['.imp'] == i,['id','timeYear','event']+var]
    df_['status'] = df_["event"].astype(bool)
    
    # d = df_.values
    # d.shape
    n_size = int(len(df_) * 0.8)

    # now = datetime.now()
    # dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    # print(dt_string)
    
    val_metrics = {}
    
    val_metrics["imp"] = "imp"+str(i)
    
    cidxs_uno_ci = []
    cidxs_uno_test = []
  
    # j = 0
    for j in range(100):
        name_j = 'iteration_'+str(j)
        print(name_j)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print(dt_string)
        
        # prepare train and test sets
        df_train_ = resample(df_, n_samples=n_size, replace=True)
        
        df_test = df_.loc[~df_['id'].isin(df_train_['id']),:]
        
        df_train, df_val = train_test_split(df_train_, test_size=0.2, random_state=123, stratify=df_train_['event'])
        
        # y_train = df_train.loc[:,['status','timeYear']].to_records(index=False)
        # y_test = df_test.loc[:,['status','timeYear']].to_records(index=False)
        
        y_train = df_train.loc[:,['status','timeYear']]
        y_test = df_test.loc[:,['status','timeYear']]
        
        # data scaling
        scaler = MinMaxScaler()
        scaler.fit(df_train.loc[:,var])
        df_train.loc[:,var] = scaler.transform(df_train.loc[:,var])
        df_val.loc[:,var] = scaler.transform(df_val.loc[:,var])
        df_test.loc[:,var] = scaler.transform(df_test.loc[:,var])    
        
        ds_train = tf.data.Dataset.from_generator(lambda: generator(df_train), output_types=(tf.float32, tf.int32, tf.float32))
        ds_train = ds_train.cache()
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.shuffle(BUFFER_SIZE)
        ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
        
        ds_val = tf.data.Dataset.from_generator(lambda: generator(df_val), output_types=(tf.float32, tf.int32, tf.float32))
        ds_val = ds_val.cache()
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.batch(BATCH_SIZE)
        
        ds_test = tf.data.Dataset.from_generator(lambda: generator(df_test), output_types=(tf.float32, tf.int32, tf.float32))
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(BATCH_SIZE)
        
        model = TestModel(num_blocks=4, hidden_size=32, rate=0.2, l2=1e-10, activation="selu", loss=custom_loss_breslow) #####
        model.train(ds_train, ds_val, max_epochs=1000, verbose=0)
        ci = model.evaluate(ds_test)
     
        cidxs_uno_test.append(model.ci_ipcw(ds_test, y_train, y_test)[0])
        
        # aucs = model.cum_auc(ds_test, y_train, y_test)
                
        
        # f+=1

    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(cidxs_uno_test, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(cidxs_uno_test, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
    
    val_metrics["cidx_uno_lower"] = lower
    val_metrics["cidx_uno_upper"] = upper
    val_metrics["cidx_uno_test"] = np.mean(cidxs_uno_test)
    
    print(str(round(np.mean(cidxs_uno_test),2))+'; '+str(round(lower,2))+'; '+str(round(upper,2)))
    
    
    dl_metrics = dl_metrics.append(val_metrics, ignore_index=True)
    






