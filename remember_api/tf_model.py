
#---- prepare net
# 这里optimizer & loss形式上都是张量

pred_output = tf.add(tf.matmul(self.inputs, self.weights), self.bias)
loss = tf.losses.mean_squared_error(output, pred_output)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

'''
grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), grad_clip) 
_optimizer = tf.train.AdamOptimizer(1e-3) 
optimizer = _optimizer.apply_gradients(zip(grads, tvars))
'''
# 直接计算并回传梯度是通过 apply_gradients, 封装在optimizer里


# train by feeds
while is(data_x, batch):
    _, loss = sess.run(
        [optimizer, loss],
        feed_dict={inputs: data_x, keep_prob: 0.8}
    )

# save
tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
saver.save(sess, path)

# load
saver.restore(sess, path)