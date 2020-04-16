import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir="tf_logs"
logdir="{}/run-{}/".format(root_logdir,now)

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

mse_summary=tf.summary.scalar('MSE',mse)
file_writer=tf.summary.FileWriter(logdir,tf.get_default_graph())

init= tf.global_variables_initializer()




with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch%100 == 0:
            print("Epoch",epoch,"MES=",mse.eval())
        sess.run(training_op)

    for batch_index in range(n_batches):
        X_batch,y_batch=fetch_batch(epoch,batch_index,batch_index)
        if batch_index % 10 ==0:
            summary_str=mse_summary.eval(feed_dict={X:X_batch,y:y_batch})
            step=epoch * n_batches+batch_index
            file_writer.add_summary(summary_str,step)
        sess.run(training_op, feed_dict={X:X_batch,y:y_batch})
    best_theta = theta.eval()
    file_writer.close()

