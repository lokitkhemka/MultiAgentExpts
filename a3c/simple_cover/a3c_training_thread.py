# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time

from game_state import GameState
from game_ac_network import GameACFFNetwork

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

        # STATE_SIZE = 6 - 3 Landmarks + 5 (comm-size)
        self.local_network = GameACFFNetwork(
            ACTION_SIZE, thread_index, device)

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
        actions2 = []
        rewards = []
        values = []
        values2 = []

        # copy weights from shared to local
        sess.run(self.sync)

        start_local_t = self.local_t

        # t_max times loop
        for i in range(LOCAL_T_MAX):
            pi_, value_ = self.local_network.run_policy_and_value(
                sess, np.concatenate([self.game_state.s1_t, [self.epSteps]]))
            pi2_, value2_ = self.local_network.run_policy_and_value(
                sess, np.concatenate([self.game_state.s2_t, [self.epSteps]]))
            action = self.choose_action(pi_)
            action2 = self.choose_action(pi2_)

            states.append(np.concatenate([self.game_state.s1_t, [self.epSteps]]))
            states2.append(np.concatenate([self.game_state.s2_t, [self.epSteps]]))

            actions.append(action)
            actions2.append(action2)

            values.append(value_)
            values2.append(value2_)

            # process game
            self.game_state.process([action, action2])

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
        R = self.local_network.run_value(sess, np.concatenate([self.game_state.s1_t, [self.epSteps]]))
        R2 = self.local_network.run_value(sess, np.concatenate([self.game_state.s2_t, [self.epSteps]]))

        actions.reverse()
        actions2.reverse()
        states.reverse()
        states2.reverse()
        rewards.reverse()
        values.reverse()
        values2.reverse()

        batch_si = []
        batch_s2i = []
        batch_a = []
        batch_a2 = []
        batch_td = []
        batch_td2 = []
        batch_R = []
        batch_R2 = []

        # compute and accmulate gradients
        for(ai, a2i, ri, si, s2i, Vi, V2i) in zip(actions, actions2, rewards,
                                                  states, states2, values, values2):

            R = ri + GAMMA * R
            R2 = ri + GAMMA * R2
            td = R - Vi
            td2 = R2 - V2i

            a = np.zeros([5])
            a[ai] = 1

            a2 = np.zeros([5])
            a2[a2i] = 1

            batch_si.append(si)
            batch_s2i.append(s2i)
            batch_a.append(a)
            batch_a2.append(a2)
            batch_td.append(td)
            batch_td2.append(td2)
            batch_R.append(R)
            batch_R2.append(R2)

        cur_learning_rate = self._anneal_learning_rate(global_t)

        batch_si.reverse()
        batch_s2i.reverse()
        batch_a.reverse()
        batch_a2.reverse()
        batch_td.reverse()
        batch_td2.reverse()
        batch_R.reverse()
        batch_R2.reverse()

        sess.run(
            self.apply_gradients,
            feed_dict={
                self.local_network.s: batch_si,
                self.local_network.a: batch_a,
                self.local_network.td: batch_td,
                self.local_network.r: batch_R,
                self.learning_rate_input: cur_learning_rate})

        sess.run(
            self.apply_gradients,
            feed_dict={
                self.local_network.s: batch_s2i,
                self.local_network.a: batch_a2,
                self.local_network.td: batch_td2,
                self.local_network.r: batch_R2,
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
