#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 18:20:02 2018

@author: naman
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/footballUPD.csv"
fields=[]
rows=[]

with open(filename) as f:
    csvreader = csv.reader(f)
    fields = csvreader.next()
    for row in csvreader:
        row=map(int,row)
        rows.append(row)
        
rows=np.array(rows)


training_d=rows[rows[:,0]<400]
testing_d =rows[rows[:,0]>=400]

train_input=training_d[:,:-3]
train_output=training_d[:,-3:]

test_input=testing_d[:,:-3]
test_output=testing_d[:,-3:]


print("Loaded training data...")

RNN_Hidden = 180
num_layers=2
learning_rate = 0.001
n_input = 8
n_output = 3
epoch = 20
dropout=tf.constant(0.5)

inp_X = tf.placeholder(tf.float32, [None, None, n_input])
out_Y = tf.placeholder(tf.int32, [None, n_output])

cell = tf.contrib.rnn.LSTMCell(RNN_Hidden,state_is_tuple=True)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout)
cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

val, state = tf.nn.dynamic_rnn(cell, inp_X, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal((RNN_Hidden, n_output)))
bias = tf.Variable(np.zeros((1, n_output)), dtype=tf.float32)

val=tf.reshape(val,[-1,RNN_Hidden])
out_Y_res = tf.reshape(out_Y,[-1,n_output])

logits = tf.matmul(val, weight) + bias
logits=tf.reshape(logits,[-1,n_output])
prediction = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=out_Y_res))

optimizer = tf.train.AdamOptimizer(learning_rate)
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(out_Y, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

#execution
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

e_vec=[]

for i in range(epoch):
    incorrect=0
    for j in range(400):
        #print("epoch ",j)
        inp, out = train_input[train_input[:,0]==j],train_output[train_input[:,0]==j]
        s1=inp.shape[0]
        inp=np.pad(inp,((RNN_Hidden-s1,0),(0,0)),'constant')
        inp=inp.reshape((1,RNN_Hidden,n_input))
        out=np.pad(out,((RNN_Hidden-s1,0),(0,0)),'constant')
        sess.run(minimize,{inp_X: inp, out_Y: out})
    for j in range(100):
        #print("epoch test",j)
        inp, out = test_input[test_input[:,0]==j+400],test_output[test_input[:,0]==j+400]
        s1=inp.shape[0]
        inp=np.pad(inp,((RNN_Hidden-s1,0),(0,0)),'constant')
        inp=inp.reshape((1,RNN_Hidden,n_input))
        out=np.pad(out,((RNN_Hidden-s1,0),(0,0)),'constant')
        sess.run(minimize,{inp_X: inp, out_Y: out})
        incorrect += sess.run(error,{inp_X: inp, out_Y: out})
    print('Epoch {:2d} error {:3.1f}%'.format(i + 1, incorrect))
    e_vec.append(incorrect)
    
plt.plot(range(epoch),e_vec, color='g')
plt.xlabel('Epochs')
plt.ylabel('Error %')
plt.title('Error % after epochs')
plt.show()
sess.close()        