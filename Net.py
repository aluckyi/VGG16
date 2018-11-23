import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
import tensorflow.contrib.slim as slim

import config as cfg


class VGG16(object):
    def __init__(self):
        self.class_num = cfg.class_num
        self.train_set_mean = cfg.train_set_mean
        self.keep_prob = cfg.keep_prob
        self.initial_weights_path = cfg.initial_weights_path
        self.weights_path = cfg.weights_path

        self.regularizer = tfc.layers.l2_regularizer(cfg.reg_scale)

        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.images = tf.placeholder(tf.float32, [None, cfg.image_H, cfg.image_W, cfg.image_C], name="images")
        self.labels = tf.placeholder(tf.int32, [None], name="labels")

        mean = tf.constant(self.train_set_mean, dtype=tf.float32, shape=[1, 1, 1, 3])
        self.images = self.images - mean

        # build network
        self.parameters = []
        self.conv_layers()
        self.fc_layers()

        # loss and probs
        loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.fc8)
        tf.losses.add_loss(loss)
        self.loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', self.loss)

        self.probs = tf.nn.softmax(self.fc8, name="probs")
        self.label_preds = tf.argmax(self.probs, axis=1, name="label_preds", output_type=tf.int32)
        self.correct_labels = tf.equal(self.label_preds, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_labels, tf.float32))
        self.correct_times_in_batch = tf.reduce_sum(tf.cast(self.correct_labels, tf.int32))

        # restorer and saver
        self.restorer = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)

        self.saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=1)

    def conv_layers(self):
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')

            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(kernel))
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool5')

    def fc_layers(self):
        # fc6
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc6w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc6b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(fc6w))
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
            self.fc6 = tf.nn.relu(fc6l)
            self.drop6 = tfc.layers.dropout(self.fc6, keep_prob=self.keep_prob,
                                            is_training=self.is_training, scope="dropout6")
            self.parameters += [fc6w, fc6b]

        # fc7
        with tf.name_scope('fc7') as scope:
            fc7w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc7b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(fc7w))
            fc7l = tf.nn.bias_add(tf.matmul(self.drop6, fc7w), fc7b)
            self.fc7 = tf.nn.relu(fc7l)
            self.drop7 = tfc.layers.dropout(self.fc7, keep_prob=self.keep_prob,
                                            is_training=self.is_training, scope="dropout7")
            self.parameters += [fc7w, fc7b]

        # fc8
        with tf.name_scope('fc8') as scope:
            fc8w = tf.Variable(tf.truncated_normal([4096, self.class_num],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc8b = tf.Variable(tf.constant(0.0, shape=[self.class_num], dtype=tf.float32),
                               trainable=True, name='biases')
            tf.losses.add_loss(self.regularizer(fc8w))
            self.fc8 = tf.nn.bias_add(tf.matmul(self.drop7, fc8w), fc8b)

    def load_initial_weights(self, session):
        weights = np.load(self.initial_weights_path)
        keys = sorted(weights.keys())
        for i, param in enumerate(self.parameters):
            k = keys[i]
            # print(i, k, np.shape(weights[k]))
            session.run(param.assign(weights[k]))

    def load_weights(self, session):
        self.restorer.restore(session, self.weights_path)

    def save_weights(self, session):
        self.saver.save(session, self.weights_path)


if __name__ == '__main__':
    net = VGG16()
    for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(x)

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     v = tf.get_default_graph().get_tensor_by_name('conv1_1/biases:0')
    #     print(sess.run(v))
    #     net.load_initial_weights(sess)
    #     print('*' * 66)
    #     print(sess.run(v))


