import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras import Sequential

raw_inputs = []
raw_labels = []

for line in open("/Users/bytedance/LearningProjects/practice_shouqianba/test_data.txt", 'r'):
    line_inputs, line_label = line.split(' ')
    line_inputs = [int(_) - 1 for _ in line_inputs.split(',')]
    line_label = int(line_label)
    print line_inputs, line_label
    
    raw_inputs.append(line_inputs)
    raw_labels.append(line_label)
   
raw_inputs = np.array(raw_inputs)
raw_labels = np.array(raw_labels)

classes = 2
raw_labels = np.eye(classes)[raw_labels]

with tf.Session() as sess:
    inputs = tf.keras.Input(shape=(4, ), dtype="int64")
    labels = tf.keras.Input(shape=(2, ), dtype="float32")
    
    emb = Embedding(5, 2)(inputs)
    concat_emb = tf.keras.layers.Flatten()(emb)
    y = concat_emb
    # y = tf.keras.layers.Dense(10, activation='relu')(concat_emb)
    prediction = tf.keras.layers.Dense(2, activation='softmax')(y)

    print prediction.shape
    print labels.shape
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=labels))
    train_optim = tf.train.AdamOptimizer().minimize(loss)
    
    init = tf.global_variables_initializer()
    sess.run(init)

    for _ in range(10):
        sess.run(train_optim, feed_dict={inputs: raw_inputs, labels: raw_labels})
    
    print sess.run(loss, feed_dict={inputs: raw_inputs, labels: raw_labels})
    print sess.run(prediction, feed_dict={inputs: raw_inputs})
    print sess.run(concat_emb, feed_dict={inputs: raw_inputs})