# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from ops import fc


# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
    def __init__(self,
                 action_size,
                 thread_index,  # -1 for global
                 device="/cpu:0"):
        self._action_size = action_size
        self._thread_index = thread_index
        self._device = device

    def prepare_loss(self, entropy_beta):
        with tf.device(self._device):
            # taken action (input for policy)
            self.a = tf.placeholder("float", [None, self._action_size])
            self.comm = tf.placeholder("float", [None, 5])

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder("float", [None])
            # R (input for value)
            self.r = tf.placeholder("float", [None])

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))
            log_pi2 = tf.log(tf.clip_by_value(self.pi2, 1e-20, 1.0))

            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
            entropy2 = -tf.reduce_sum(self.pi2 * log_pi2, reduction_indices=1)

            # policy loss (output)  (Adding minus, because the original paper's
            # objective function is for gradient ascent, but we use gradient
            # descent optimizer.)
            policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(
                log_pi, self.a), reduction_indices=1) * self.td +
                entropy * entropy_beta)

            policy_loss2 = - tf.reduce_sum(tf.reduce_sum(tf.multiply(
                log_pi2, self.comm), reduction_indices=1) * self.r +
                entropy2 * entropy_beta)

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradienet of policy and value are summed up
            self.total_loss = policy_loss + value_loss + policy_loss2

    def run_policy_and_value(self, sess, s_t):
        raise NotImplementedError()

    def run_policy(self, sess, s_t):
        raise NotImplementedError()

    def run_value(self, sess, s_t):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_netowrk, name=None):
        src_vars = src_netowrk.get_vars()
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.name_scope(name, "GameACNetwork", []) as name:
                for(src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    # weight initialization based on muupan's code
    # https://github.com/muupan/async-rl/blob/master/a3c_ale.py
    def _fc_variable(self, weight_shape):
        input_channels = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(
            weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(
            bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv_variable(self, weight_shape):
        w = weight_shape[0]
        h = weight_shape[1]
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
        d = 1.0 / np.sqrt(input_channels * w * h)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(
            weight_shape, minval=-d, maxval=d))
        bias = tf.Variable(tf.random_uniform(
            bias_shape, minval=-d, maxval=d))
        return weight, bias

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                            padding="VALID")


# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
    def __init__(self,
                 action_size,
                 thread_index,  # -1 for global
                 state_size,
                 device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, thread_index, device)

        scope_name = "net_" + str(self._thread_index)
        with tf.device(self._device), tf.variable_scope(scope_name) as scope:

            # lstm
            self.lstm = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

            # state (input)
            self.s = tf.placeholder("float", [None, state_size])
            self.s2 = tf.placeholder("float", [None, 1])

            self.W_fc3, self.b_fc3 = self._fc_variable([256, 1])

            h_fc1 = fc(self.s, 30, name='fc1')
            h_fc2 = fc(h_fc1, 20, name='fc1')
            h_fc3 = fc(h_fc2, 20, name='fc1')

            h_fc3_reshaped = tf.reshape(h_fc3, [1, -1, 20])
            # h_fc_reshaped = (1,5,256)

            # place holder for LSTM unrolling time step size.
            self.step_size = tf.placeholder(tf.float32, [1])

            self.initial_lstm_state0 = tf.placeholder(tf.float32, [1, 20])
            self.initial_lstm_state1 = tf.placeholder(tf.float32, [1, 20])
            self.initial_lstm_state = tf.contrib.rnn.LSTMStateTuple(
                self.initial_lstm_state0,
                self.initial_lstm_state1)

            # Unrolling LSTM up to LOCAL_T_MAX time steps. (= 5time steps.)
            # When episode terminates unrolling time steps becomes less than
            # LOCAL_TIME_STEP.
            # Unrolling step size is applied via self.step_size placeholder.
            # When forward propagating, step_size is 1.
            # (time_major = False, so output shape is
            # [batch_size, max_time, cell.output_size])
            lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
                self.lstm,
                h_fc3_reshaped,
                initial_state=self.initial_lstm_state,
                sequence_length=self.step_size,
                time_major=False,
                scope=scope)

            # lstm_outputs: (1,5,256) for back prop,(1,1,256) for forward prop.

            lstm_outputs = tf.reshape(lstm_outputs, [-1, 20])

            self.W_fc2, self.b_fc2 = self._fc_variable([20, action_size])
            # policy (output)
            self.pi = tf.nn.softmax(
                tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)

            self.W_fcx, self.b_fcx = self._fc_variable([1, 5])
            # policy (output)
            self.pi2 = tf.nn.softmax(
                tf.matmul(self.s2, self.W_fcx) + self.b_fcx)

            # value (output)
            v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
            self.v = tf.reshape(v_, [-1])

            scope.reuse_variables()
            self.W_lstm = tf.get_variable("basic_lstm_cell/weights")
            self.b_lstm = tf.get_variable("basic_lstm_cell/biases")

            self.reset_state()

    def reset_state(self):
        self.lstm_state_out = tf.contrib.rnn.LSTMStateTuple(np.zeros([1, 256]),
                                                            np.zeros([1, 256]))

    def run_policy_and_value(self, sess, s_t):
        # This run_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        pi2_out, pi_out, v_out, self.lstm_state_out = sess.run(
            [self.pi2, self.pi, self.v, self.lstm_state],
            feed_dict={self.s: [s_t],
                       self.initial_lstm_state0: self.lstm_state_out[0],
                       self.initial_lstm_state1: self.lstm_state_out[1],
                       self.step_size: [1]})

        # pi_out: (1,3), v_out: (1)
        return (pi_out[0], pi2_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        # This run_policy() is used for displaying the result with display tool.
        pi2_out, pi_out, self.lstm_state_out = sess.run(
            [self.pi2, self.pi, self.lstm_state],
            feed_dict={self.s: [s_t],
                       self.initial_lstm_state0: self.lstm_state_out[0],
                       self.initial_lstm_state1: self.lstm_state_out[1],
                       self.step_size: [1]})

        return pi_out[0], pi2_out[0]

    def run_value(self, sess, s_t):
        # This run_value() is used for calculating V for bootstrapping at the
        # end of LOCAL_T_MAX time step sequence.
        # When next sequcen starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v, self.lstm_state],
                            feed_dict={self.s: [s_t],
                                       self.initial_lstm_state0: self.lstm_state_out[0],
                                       self.initial_lstm_state1: self.lstm_state_out[1],
                                       self.step_size: [1]})

        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]

    def get_vars(self):
        return [self.W_fcx, self.b_fcx,
                self.W_lstm, self.b_lstm,
                self.W_fc2, self.b_fc2,
                self.W_fc3, self.b_fc3]
