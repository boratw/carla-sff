import numpy as np
import tensorflow.compat.v1 as tf
import math

class PredictBehavior:
    def __init__(self, name="0", reuse=False, learner_lr=0.0001, learner_reg=0.001, learner_prob_reg=0.0001):
        self.name = "PredictBehavior_" + name
        with tf.variable_scope("PredictBehavior_" + name, reuse=reuse):
            self.layer_input_map = tf.placeholder(tf.float32, [None, 256, 256, 3])
            self.layer_input_state = tf.placeholder(tf.float32, [None, 5])
            self.layer_input_target = tf.placeholder(tf.float32, [None, 2])
            
            with tf.variable_scope("ConvNet_" + name, reuse=reuse):
                conv_w1 = tf.get_variable("conv_w1", shape=[5, 5, 3, 16], dtype=tf.float32, 
                    initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01), trainable=True)
                conv_b1 = tf.get_variable("conv_b1", shape=[16], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)

                conv1 = tf.nn.conv2d(self.layer_input_map, conv_w1, strides=[1, 1, 1, 1], padding='VALID') + conv_b1
                conv1 = tf.nn.leaky_relu(conv1, alpha=0.05)
                conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                conv_w2 = tf.get_variable("conv_w2", shape=[5, 5, 16, 32], dtype=tf.float32, 
                    initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01), trainable=True)
                conv_b2 = tf.get_variable("conv_b2", shape=[32], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)

                conv2 = tf.nn.conv2d(conv1, conv_w2, strides=[1, 1, 1, 1], padding='VALID') + conv_b2
                conv2 = tf.nn.leaky_relu(conv2, alpha=0.05)
                conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                conv_w3 = tf.get_variable("conv_w3", shape=[5, 5, 32, 48], dtype=tf.float32, 
                    initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01), trainable=True)
                conv_b3 = tf.get_variable("conv_b3", shape=[48], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)

                conv3 = tf.nn.conv2d(conv2, conv_w3, strides=[1, 1, 1, 1], padding='VALID') + conv_b3
                conv3 = tf.nn.leaky_relu(conv3, alpha=0.05)
                conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                conv_w4 = tf.get_variable("conv_w4", shape=[5, 5, 48, 64], dtype=tf.float32, 
                    initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01), trainable=True)
                conv_b4 = tf.get_variable("conv_b4", shape=[64], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)

                conv4 = tf.nn.conv2d(conv3, conv_w4, strides=[1, 1, 1, 1], padding='VALID') + conv_b4
                conv4 = tf.nn.leaky_relu(conv4, alpha=0.05)
                conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


                conv_w5 = tf.get_variable("conv_w5", shape=[5, 5, 64, 64], dtype=tf.float32, 
                    initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.01), trainable=True)
                conv_b5 = tf.get_variable("conv_b5", shape=[64], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)

                conv5 = tf.nn.conv2d(conv4, conv_w5, strides=[1, 1, 1, 1], padding='VALID') + conv_b5
                conv5 = tf.nn.leaky_relu(conv5, alpha=0.05)
                conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                self.conv_fc = tf.layers.Flatten()(conv5)
                self.convnet_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

            with tf.variable_scope("ProbFC_" + name, reuse=reuse):
                fc = tf.concat([self.conv_fc, self.layer_input_state], axis=1)

                w1 = tf.get_variable("w1", shape=[1029, 256], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / 1029 + 256), math.sqrt(1.0 / 1029 + 256), dtype=tf.float32),
                    trainable=True)
                b1 = tf.get_variable("b1", shape=[256], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                fc1 = tf.matmul(fc, w1) + b1
                fc1 = tf.nn.leaky_relu(fc1, alpha=0.05)


                w2 = tf.get_variable("w2", shape=[256, 64], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / (256 + 64)), math.sqrt(1.0 / (256 + 64)), dtype=tf.float32),
                    trainable=True)
                b2 = tf.get_variable("b2", shape=[64], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                fc2 = tf.matmul(fc1, w2) + b2
                fc2 = tf.nn.leaky_relu(fc2, alpha=0.05)


                w3 = tf.get_variable("w3", shape=[64, 8], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / (64 + 40)), math.sqrt(1.0 / (64 + 40)), dtype=tf.float32),
                    trainable=True)
                b3 = tf.get_variable("b3", shape=[8], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                fc3 = tf.matmul(fc2, w3) + b3
                self.raw_prob = fc3
                self.probfc_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

            with tf.variable_scope("DistFC_" + name, reuse=reuse):
                fc = tf.concat([self.conv_fc, self.layer_input_state], axis=1)

                w1 = tf.get_variable("w1", shape=[1029, 256], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / 1029 + 256), math.sqrt(1.0 / 1029 + 256), dtype=tf.float32),
                    trainable=True)
                b1 = tf.get_variable("b1", shape=[256], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                fc1 = tf.matmul(fc, w1) + b1
                fc1 = tf.nn.leaky_relu(fc1, alpha=0.05)


                w2 = tf.get_variable("w2", shape=[256, 64], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / (256 + 64)), math.sqrt(1.0 / (256 + 64)), dtype=tf.float32),
                    trainable=True)
                b2 = tf.get_variable("b2", shape=[64], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                fc2 = tf.matmul(fc1, w2) + b2
                fc2 = tf.nn.leaky_relu(fc2, alpha=0.05)


                w3 = tf.get_variable("w3", shape=[64, 32], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(1.0 / (64 + 40)), math.sqrt(1.0 / (64 + 40)), dtype=tf.float32),
                    trainable=True)
                b3 = tf.get_variable("b3", shape=[32], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                fc3 = tf.matmul(fc2, w3) + b3
                self.mu, self.logsig = tf.split(fc3, [16, 16], 1)
                self.distfc_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

            target_output = tf.tile(self.layer_input_target, [1, 8])
            self.logsig_clipped = tf.clip_by_value(self.logsig, -5.0, 2.0)

            target_log_prob = - 0.5 * (self.mu - target_output) ** 2 / tf.exp(tf.clip_by_value(self.logsig, -2, 2)) - tf.clip_by_value(self.logsig, -2, 2)
            self.target_log_prob = tf.reduce_sum(tf.reshape(target_log_prob, [-1, 8, 2]), axis=2)
            self.target_prob = self.target_log_prob - tf.math.reduce_max(self.target_log_prob, axis=1, keepdims=True)
            self.target_prob = tf.exp(self.target_prob)
            self.target_prob = self.target_prob / tf.math.reduce_sum(self.target_prob, axis=1, keepdims=True)

            self.stable_raw_prob = self.raw_prob - tf.stop_gradient(tf.math.reduce_max(self.raw_prob, axis=1, keepdims=True))
            self.prob = tf.exp(self.stable_raw_prob)
            self.prob = self.prob / tf.math.reduce_sum(self.prob, axis=1, keepdims=True)

            self.maximum_prob = tf.math.reduce_mean(tf.math.reduce_max(self.prob, axis=1))
            self.average_logsig = tf.math.reduce_mean(self.logsig)
            self.prob_likelihood_loss = tf.reduce_mean((self.prob - tf.stop_gradient(self.target_prob)) ** 2)
            self.prob_regularization_loss = tf.reduce_mean(tf.nn.relu(self.raw_prob ** 2 - 1.))

            self.dist = tf.distributions.Normal(loc=self.mu, scale=tf.exp(self.logsig_clipped))
            self.reparameterized = self.dist.sample()
            likelihood_loss = tf.reduce_sum(tf.reshape((self.reparameterized - target_output) ** 2, [-1, 8, 2]), axis=2)
            self.dist_likelihood_loss = tf.reduce_mean(likelihood_loss * tf.clip_by_value(self.prob, 0.0001, 1.0))
            self.dist_regularization_loss = tf.reduce_mean(self.mu ** 2 + tf.exp(self.logsig_clipped) ** 2 - self.logsig_clipped ** 2 - 1.)

            self.raw_stoc_output = tf.reshape(self.reparameterized, [-1, 8, 2])
            self.stoc_output = self.raw_stoc_output * tf.reshape(self.prob, [-1, 8, 1])
            self.stoc_output = tf.reduce_mean(self.stoc_output, axis=2)


            self.prob_optimizer = tf.train.AdamOptimizer(learner_lr)
            self.prob_train_action = self.prob_optimizer.minimize(loss = self.prob_likelihood_loss + self.prob_regularization_loss * learner_prob_reg, var_list = self.probfc_params)
            self.dist_optimizer = tf.train.AdamOptimizer(learner_lr)
            self.dist_train_action = self.dist_optimizer.minimize(loss = self.dist_likelihood_loss + self.dist_regularization_loss * learner_reg, var_list = [*self.convnet_params, *self.distfc_params])

         
            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}


    def optimize_batch(self, input_map, input_state, input_target):
        input_list = {self.layer_input_map : input_map, self.layer_input_state : input_state, self.layer_input_target : input_target}
        sess = tf.get_default_session()
        if(self.reset_log_num):
            self.log_learner_prob_li = 0
            self.log_learner_prob_reg = 0
            self.log_learner_dist_li = 0
            self.log_learner_dist_reg = 0
            self.log_learner_prob = 0
            self.log_learner_logsig = 0
            self.log_num = 0
            self.reset_log_num = False

        _, l1, l2  = sess.run([self.prob_train_action, self.prob_likelihood_loss, self.prob_regularization_loss], input_list)
        _, l3, l4, l5, l6  = sess.run([self.dist_train_action, self.dist_likelihood_loss, self.dist_regularization_loss, self.maximum_prob, self.average_logsig], input_list)
        self.log_learner_prob_li += l1
        self.log_learner_prob_reg += l2
        self.log_learner_dist_li += l3
        self.log_learner_dist_reg += l4
        self.log_learner_prob += l5
        self.log_learner_logsig += l6
        self.log_num += 1

    def get_result(self, input_map, input_state):
        input_list = {self.layer_input_map : input_map, self.layer_input_state : input_state}
        sess = tf.get_default_session()

        l1, l2 = sess.run([self.prob, self.reparameterized], input_list)
        return l1, l2

    def network_initialize(self):
        sess = tf.get_default_session()
        self.log_learner_prob_li = 0
        self.log_learner_prob_reg = 0
        self.log_learner_dist_li = 0
        self.log_learner_dist_reg = 0
        self.log_learner_reg = 0
        self.log_learner_prob = 0
        self.log_learner_logsig = 0
        self.log_num = 0
        self.reset_log_num = False

    def network_update(self):
        self.reset_log_num = True

    def log_caption(self):
        return "\t"  + self.name + "_Learner_Prob_Likelihood\t" + self.name + "_Learner_Prob_Regularization\t" + self.name + "_Learner_Dist_Likelihood\t" + self.name \
            + "_Learner_Dist_Regularization\t" + self.name + "_Learner_Prob_Magnitude\t" + self.name + "_Learner_Logsig"
            
    
    def current_log(self):
        return "\t" + str(self.log_learner_prob_li / self.log_num) + "\t"  + str(self.log_learner_prob_reg / self.log_num) + "\t"  + str(self.log_learner_dist_li / self.log_num) + "\t" + str(self.log_learner_dist_reg / self.log_num) \
            + "\t" + str(self.log_learner_prob / self.log_num) + "\t" + str(self.log_learner_logsig / self.log_num)

    def log_print(self):
        print ( self.name + "\n" \
            + "\tLearner_Prob_Likelihood      : " + str(self.log_learner_prob_li / self.log_num) + "\n" \
            + "\tLearner_Prob_Regularization  : " + str(self.log_learner_prob_reg / self.log_num) + "\n" \
            + "\tLearner_Dist_Likelihood      : " + str(self.log_learner_dist_li / self.log_num) + "\n" \
            + "\tLearner_Dist_Regularization  : " + str(self.log_learner_dist_reg / self.log_num) + "\n" \
            + "\tLearner_Prob_Magnitude       : " + str(self.log_learner_prob / self.log_num) + "\n" \
            + "\tLearner_Logsig               : " + str(self.log_learner_logsig / self.log_num) + "\n" )