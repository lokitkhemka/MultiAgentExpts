# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time

from game_state import GameState
from game_ac_network import GameACLSTMNetwork

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA

ACTION_SIZE = 5

LOG_INTERVAL = 1000
PERFORMANCE_LOG_INTERVAL = 5000


class A3CTrainingThread(object):
    def __init__(self,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        self.local_network = GameACLSTMNetwork(
            ACTION_SIZE, thread_index, 11, device)

        self.local_network.prepare_loss(ENTROPY_BETA)

        with tf.device(device):
            var_refs = [v._ref() for v in self.local_network.get_vars()]
            self.gradients = tf.gradients(
                self.local_network.total_loss, var_refs,
                gate_gradients=False,
                aggregation_method=None,
                colocate_gradients_with_ops=False)

        self.apply_gradients = grad_applier.apply_gradients(
            global_network.get_vars(),
            self.gradients)

        self.sync = self.local_network.sync_from(global_network)

        self.game_state = GameState(113 * thread_index)

        self.local_t = 0
        self.epSteps = 0

        self.initial_learning_rate = initial_learning_rate

        self.episode_reward = 0

        # variable controling log output
        self.prev_local_t = 0

    def _anneal_learning_rate(self, global_time_step):
        learning_rate = self.initial_learning_rate * \
            (self.max_global_time_step - global_time_step) / \
            self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    def choose_action(self, pi_values):
        return np.random.choice(range(len(pi_values)), p=pi_values)

    def _record_score(self, sess, summary_writer, summary_op,
                      score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)
        summary_writer.flush()

    def set_start_time(self, start_time):
        self.start_time = start_time

    def process(self, sess, global_t, summary_writer, summary_op, score_input):
        states = []
        states2 = []
        actions = []
        comms = []
        rewards = []
        values = []

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        start_lstm_state = self.local_network.lstm_state_out

        # t_max times loop
        for i in range(LOCAL_T_MAX):
            pi_, comm_, value_ = self.local_network.run_policy_and_value(
                sess, self.game_state.s_t, self.game_state.s2)
            action = self.choose_action(pi_)
            comm = self.choose_action(comm_)

            states.append(self.game_state.s_t)
            states2.append(self.game_state.s2)
            actions.append(action)
            comms.append(comm)
            values.append(value_)

            # process game
            self.game_state.process(action, comm)

            # receive game result
            reward = self.game_state.reward
            self.episode_reward += reward

            if (self.thread_index == 0) and (self.local_t % LOG_INTERVAL == 0):
                print("pi={}".format(pi_))
                print(" V={}".format(value_))
                print(" R={}".format(reward))

            # clip reward
            #  rewards.append(np.clip(reward, -1, 1))
            rewards.append(reward)

            self.local_t += 1
            self.epSteps += 1

            # s_t1 -> s_t
            self.game_state.update()

            if self.epSteps >= 100:
                self.epSteps = 0
                if(self.thread_index == 0 and self.local_t % LOG_INTERVAL == 0):
                    print("score={}".format(self.episode_reward))

                    self._record_score(sess, summary_writer, summary_op,
                                       score_input, self.episode_reward,
                                       global_t)

                self.episode_reward = 0
                self.game_state.reset()
                self.local_network.reset_state()
                break

        R = 0.0
        R = self.local_network.run_value(sess, self.game_state.s_t)

        actions.reverse()
        states.reverse()
        states2.reverse()
        rewards.reverse()
        values.reverse()
        comms.reverse()

        batch_si = []
        batch_s2 = []
        batch_a = []
        batch_c = []
        batch_td = []
        batch_R = []

        # compute and accmulate gradients
        for(ai, ri, si, Vi, ci, s2i) in zip(actions, rewards, states, values, comms, states2):
            R = ri + GAMMA * R
            td = R - Vi
            a = np.zeros([ACTION_SIZE])
            a[ai] = 1

            c = np.zeros([5])
            c[ci] = 1

            batch_si.append(si)
            batch_s2.append(s2i)
            batch_a.append(a)
            batch_c.append(c)
            batch_td.append(td)
            batch_R.append(R)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        batch_si.reverse()
        batch_s2.reverse()
        batch_a.reverse()
        batch_c.reverse()
        batch_td.reverse()
        batch_R.reverse()

        sess.run(
            self.apply_gradients,
            feed_dict={
                self.local_network.s: batch_si,
                self.local_network.a: batch_a,
                self.local_network.comm: batch_c,
                self.local_network.s2: batch_s2,
                self.local_network.td: batch_td,
                self.local_network.r: batch_R,
                self.local_network.initial_lstm_state: start_lstm_state,
                self.local_network.step_size: [len(batch_a)],
                self.learning_rate_input: cur_learning_rate})

        if (self.thread_index == 0) and \
           (self.local_t - self.prev_local_t >= PERFORMANCE_LOG_INTERVAL):
            self.prev_local_t += PERFORMANCE_LOG_INTERVAL
            elapsed_time = time.time() - self.start_time
            steps_per_sec = global_t / elapsed_time
            print("### Performance : {} STEPS in {:.0f} sec. \
                    {:.0f} STEPS/sec. {:.2f}M STEPS/hour".format(
                global_t, elapsed_time, steps_per_sec,
                steps_per_sec * 3600 / 1000000.))

        # return advanced local step size
        diff_local_t = self.local_t - start_local_t
        return diff_local_t
