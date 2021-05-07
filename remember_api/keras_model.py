
# docs: https://github.com/xialuoke95/ai_for_qipai/blob/main/model.py

from tensorflow.keras.models import Model

# prepare net
x = Conv2D(filters=32, kernel_size=3, padding="same", kernel_regularizer=l2(conf.l2_c), bias_regularizer=l2(conf.l2_c))(
    inputs
)
ph = Conv2D(filters=2, kernel_size=1, kernel_regularizer=l2(conf.l2_c), bias_regularizer=l2(conf.l2_c))(x)
net = Model(inputs=inputs, outputs=[ph])
net.compile(
    optimizer=keras.optimizers.SGD(lr=conf.lr, momentum=0.9), 
    loss=["categorical_crossentropy", "mean_squared_error"]
)

# train
# --: train_on_batch vs fit (fit_generator)
# https://stackoverflow.com/questions/49100556/what-is-the-use-of-train-on-batch-in-keras
net.fit(
    x=x,y=y,  # x = {"input1": train_x, "input2": train_x2},
    batch_size=conf.batch_size,
    epochs=100,
    validation_split=conf.validation_split,
    callbacks=[EarlyStopping()],
)

# save
net.save_weights(path)

# load
net.load_weights(path)