import tensorflow as tf

#Function for weight variable
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#Function for bias variable
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#Function for max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#Function for convolutional layer
def conv_layer(input, shape, stride=1, padding='SAME'):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.conv2d(input, W, strides=[1, stride, stride, 1], padding='SAME') + b
        
#Function for de- convolutional layer
def deconv_layer(input, shape,op_shape):
    W = weight_variable(shape)
    b = bias_variable([shape[2]])
    return tf.nn.conv2d_transpose(input,W,output_shape=op_shape,strides=[1, 2, 2, 1],padding='SAME') + b

# Discriminator Network
def discriminator(X, keep_prob, is_train=True, reuse=False):
    with tf.variable_scope('discriminator'):
        if reuse:
            tf.get_variable_scope().reuse_variables()


        batch_size = tf.shape(X)[0]
        
        conv1 = conv_layer(X, shape=[5, 5, 3, 96],stride=2, padding='SAME')                             # Convolution
        drop1=tf.nn.dropout(conv1, keep_prob=keep_prob)
        batch_norm_1=tf.nn.relu(tf.contrib.layers.batch_norm(drop1))                                    #batch normalization

        conv2 = conv_layer(batch_norm_1, shape=[5, 5, 96, 192],stride=2,padding='SAME')                 # Convolution
        drop2=tf.nn.dropout(conv2, keep_prob=keep_prob)
        batch_norm_2=tf.nn.relu(tf.contrib.layers.batch_norm(drop2))                                    #batch normalization

        conv3 = conv_layer(batch_norm_2, shape=[5, 5, 192, 384],stride=2,padding='SAME')                # Convolution
        drop3=tf.nn.dropout(conv3, keep_prob=keep_prob)
        batch_norm_3=tf.nn.relu(tf.contrib.layers.batch_norm(drop3))                                    #batch normalization

        conv4 = conv_layer(batch_norm_3, shape=[3, 3, 384, 384],padding='SAME')                         # Convolution
        #drop4=tf.nn.dropout(conv4, keep_prob=keep_prob)
        batch_norm_4=tf.nn.relu(tf.contrib.layers.batch_norm(conv4))                                    #batch normalization
        print(batch_norm_4)

        W5 = weight_variable([4, 4, 384, 384])
        B5 = bias_variable([384])
        conv5=tf.nn.conv2d(batch_norm_4,W5,strides=[1, 1, 1, 1], padding='VALID')+B5                    # Convolution
        batch_norm_5=tf.nn.relu(tf.contrib.layers.batch_norm(conv5))                                    #batch normalization
        k=tf.nn.relu(batch_norm_5)
        
        W6 = weight_variable([384, 10+1])
        flat = tf.reshape(k,[batch_size,384])                                                           # Flatten to a 1D vector
        output = tf.matmul(flat, W6)                                                                    # Fully connected layer

        return tf.nn.softmax(output), output, flat

# Generator Network
def generator(Z, keep_prob, is_train=True):
    with tf.variable_scope('generator'):
        batch_size = tf.shape(Z)[0]
        i_shape= [100, 4*4*512]
        W1 = weight_variable(i_shape) 
        B1 = bias_variable([i_shape[1]])
        
        Z = tf.nn.relu(tf.matmul(Z, W1) + B1)
        Z = tf.reshape(Z, [batch_size, 4, 4, 512])
        Z = tf.contrib.layers.batch_norm(Z)
        
        de_conv1 = deconv_layer(Z, shape=[4, 4, 256, 512],op_shape=[batch_size, 8, 8, 256])                     #Deconvolution
        drop1=tf.nn.dropout(de_conv1, keep_prob=keep_prob)
        batch_norm_1=tf.nn.relu(tf.contrib.layers.batch_norm(drop1))

        de_conv2 = deconv_layer(batch_norm_1, shape=[6, 6, 128, 256],op_shape=[batch_size, 16, 16, 128])        #Deconvolution
        drop2=tf.nn.dropout(de_conv2, keep_prob=keep_prob)
        batch_norm_2=tf.nn.relu(tf.contrib.layers.batch_norm(drop2))

        de_conv3 = deconv_layer(batch_norm_2, shape=[6, 6, 64, 128],op_shape=[batch_size, 32, 32, 64])          #Deconvolution
        drop3=tf.nn.dropout(de_conv3, keep_prob=keep_prob)
        batch_norm_3=tf.nn.relu(tf.contrib.layers.batch_norm(drop3))
        
        inp=tf.nn.relu(batch_norm_3)
        o_shape= [3, 3, 64, 3]
        
        conv = conv_layer(inp, shape=o_shape)
        
        output = tf.nn.tanh(conv)

        return output