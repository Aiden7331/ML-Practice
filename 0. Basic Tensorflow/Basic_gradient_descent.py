import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

n_epochs=1000
learning_rate=0.01

scaler= StandardScaler()
housing=fetch_california_housing()
m,n=housing.data.shape

scaled_housing_data= scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias=np.c_[np.ones((m,1)),scaled_housing_data]

X=tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32, name="X")
y=tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32, name="y")
theta=tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")
y_pred=tf.matmul(X,theta, name="predictions")

error = y_pred-y
mse=tf.reduce_mean(tf.square(error),name="mse")
gradients=2/m * tf.gradients(mse,[theta])[0]
training_op=tf.assign(theta,theta - learning_rate * gradients)
saver = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0),name="theta")
init= tf.global_variables_initializer()
saver=tf.train.Saver({"weights":theta})

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch%100 == 0:
            save_path=saver.save(sess,"tmp/my_model.ckpt")
            print("Epoch",epoch,"MES=",mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    save_path=saver.save(sess,"/tmp/my_model_final.ckpt")
