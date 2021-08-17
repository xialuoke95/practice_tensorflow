input_array = np.random.randint(10, size=(100, 10))
input_labels = np.random.randint(1, size=(100, 1))

with tf.Session() as sess:
    inputs = tf.keras.Input(shape=(10, ), dtype="float32")
    labels = tf.keras.Input(shape=(1, ), dtype="float32")
    
    emb = Embedding(10, 4)(inputs)
    concat_emb = tf.keras.layers.Flatten()(emb)
    y = tf.keras.layers.Dense(10, activation='relu')(concat_emb)
    prediction = tf.keras.layers.Dense(1, activation='softmax')(y)

    print prediction.shape
    print labels.shape
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=labels))
    train_optim = tf.train.AdamOptimizer().minimize(loss)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    batch_size = 100
    for _ in range(100):
        input_array = np.random.randint(10, size=(batch_size, 10))
        input_labels = np.random.randint(2, size=(batch_size, 1))
        sess.run(train_optim, feed_dict={inputs: input_array, labels: input_labels})
    
    print sess.run(loss, feed_dict={inputs: input_array, labels: input_labels})
    print sess.run(prediction, feed_dict={inputs: input_array})
    