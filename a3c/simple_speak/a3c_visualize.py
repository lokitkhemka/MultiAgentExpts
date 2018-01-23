
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from game_ac_network import GameACLSTMNetwork

from constants import ACTION_SIZE
from constants import CHECKPOINT_DIR
from constants import USE_GPU
from game_state import GameState


device = "/cpu:0"
if USE_GPU:
    device = "/gpu:0"

global_t = 0

stop_requested = False

global_network = GameACLSTMNetwork(ACTION_SIZE, -1, 11, device)

training_threads = []

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard


def choose_action(pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)


# init or load checkpoint with saver
saver = tf.train.Saver(max_to_keep=2)
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    tokens = checkpoint.model_checkpoint_path.split("-")
    # set global step
    print(tokens[1])
    global_t = int(tokens[1])
    print(">>> global step set: ", global_t)
    # set wall time
    wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
    with open(wall_t_fname, 'r') as f:
        wall_t = float(f.read())
else:
    print("Could not find old checkpoint")
    # set wall time
    wall_t = 0.0

gs = GameState(113)
rew = 0

for i in range(1000):
    pi_, comm_, value_ = global_network.run_policy_and_value(
        sess, gs.s_t, gs.s2)

    action = choose_action(pi_)
    comm = choose_action(comm_)
    gs.process(action, comm)
    #  print(pi_)
    #  print(comm_)
    #  print(gs.reward)
    rew += gs.reward
    gs.update()

print(rew)
gs.reset()
rew = 0

for i in range(1000):
    pi_, comm_, value_ = global_network.run_policy_and_value(
        sess, gs.s_t, gs.s2)

    action = choose_action(pi_)
    comm = choose_action(comm_)
    gs.process(action, comm)
    #  print(pi_)
    #  print(comm_)
    #  print(gs.reward)
    rew += gs.reward
    gs.update()

print(rew)

# set start time
