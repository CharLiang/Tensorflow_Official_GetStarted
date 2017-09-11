
# in this sample code, we run OLS and get the sum of variance, which is loss 





# import our dear tensorflow package
import tensorflow as tf






#Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

#placeholder is a promise to provide a value later
x = tf.placeholder(tf.float32)
linear_model = W * x + b


#all of functions are run in the tf.Session()
sess = tf.Session()


# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation as follows
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))




y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)

#sum all the squared errors to create a single scalar that abstracts the error of all examples using tf.reduce_sum
loss = tf.reduce_sum(squared_deltas)

#A loss function measures how far apart the current model is from the provided data
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
