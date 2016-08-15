import argparse
import gym
import numpy      as np
import tensorflow as tf

from collections        import deque
from scipy.misc         import imresize
from tensorflow.contrib import layers
from threading          import Thread

from utils import (
    GymOnProcess,
    ExplorationSchedule,
    AtomicInt,
    SharedResource
)

def preproc_graph(session, input_shape):
    prev_frame = tf.placeholder(tf.uint8, [1] + list(input_shape))
    cur_frame  = tf.placeholder(tf.uint8, [1] + list(input_shape))
    out  = tf.maximum(tf.cast(prev_frame, tf.float32),
                      tf.cast(cur_frame, tf.float32))
    out  = tf.image.rgb_to_grayscale(out)
    out  = tf.image.resize_bilinear(out, (84, 84)) / 255.0
    def f(pf, cf):
        return session.run(out, {prev_frame:pf[None], cur_frame: cf[None]})[0]
    return f

def forward(image, num_actions):
    # Conv1
    out = layers.convolution2d(image, num_outputs=16, kernel_size=8, stride=4, activation_fn=tf.nn.relu, scope='conv1')
    out = layers.convolution2d(out, num_outputs=32, kernel_size=4, stride=2, activation_fn=tf.nn.relu, scope='conv2')
    out = layers.flatten(out, scope='flatten')
    out = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu, scope='fc1')

    action_logprobs = tf.nn.log_softmax(layers.fully_connected(out, num_outputs=num_actions, activation_fn=None, scope='fc_actor'))
    value           = layers.fully_connected(out, num_outputs=1, activation_fn=None, scope='fc_critic')
    value           = tf.reshape(value, [-1])
    return action_logprobs, value

def a3c_graph(args, session, num_actions, reuse):
    with tf.variable_scope('a3c', reuse=reuse):
        states_ph             = tf.placeholder(tf.float32, [None, 84, 84, args.context])
        actions_ph            = tf.placeholder(tf.int32,   [None])
        discounted_rewards_ph = tf.placeholder(tf.float32, [None])

        action_logprobs, value = forward(states_ph, num_actions)

        def run_actor(state):
            ret = session.run(action_logprobs, {states_ph: state[None]})[0]
            return np.argmax(ret)

        def run_critic(state):
            return session.run(value, {states_ph: state[None]})[0]

        action_mask = layers.one_hot_encoding(actions_ph, num_classes=num_actions)
        chosen_actions_logprobs = tf.reduce_sum(action_mask * action_logprobs, 1)

        actor_policy_advantage = chosen_actions_logprobs * (discounted_rewards_ph - value)
        actor_policy_advantage = tf.reduce_mean(actor_policy_advantage)

        actor_policy_entropy = tf.reduce_sum(-tf.exp(action_logprobs) * action_logprobs, 1)
        actor_policy_entropy = tf.reduce_mean(actor_policy_entropy)

        critic_error = (value - discounted_rewards_ph)**2
        critic_error = tf.reduce_mean(critic_error)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.5)

        update_op = tf.group(
            optimizer.minimize(-(actor_policy_advantage + args.beta * actor_policy_entropy)),
            optimizer.minimize(critic_error)
        )

        def perform_update(states, actions, discounted_rewards):
            session.run(update_op, {
                states_ph:             states,
                actions_ph:            actions,
                discounted_rewards_ph: discounted_rewards,
            })
        return run_actor, run_critic, perform_update

def run_simulator(args, global_num_frames, preproc_graphs, a3c_graphs):
    with preproc_graphs.lease() as preproc_single_frame:
        with a3c_graphs.lease() as a3c_graph:
            with GymOnProcess() as gym_on_process:
                env = gym_on_process.Env(args.env)
                action_space      = env.get_action_space()
                observation_space = env.get_observation_space()
                exploration       = ExplorationSchedule(args.exploration)

                unprocessed_images = deque([observation_space.low for _ in range(args.context)], maxlen=args.context)
                processed_images   = deque([np.zeros((84, 84, 1)) for _ in range(args.context)], maxlen=args.context)

                observations, actions, rewards = [], [], []

                run_actor, run_critic, perform_update = a3c_graph

                def preproc(observation):
                    unprocessed_images.append(observation)
                    processed_images.append(preproc_single_frame(unprocessed_images[-2], unprocessed_images[-1]))
                    return np.concatenate(processed_images, 2)

                while global_num_frames.get() < args.max_frames:
                    observation = env.reset()
                    exploration.reset()
                    done = False

                    observations.append(preproc(observation))

                    while not done and global_num_frames.get() < args.max_frames:
                        if exploration.should_be_random(global_num_frames.get()):
                            action = action_space.sample()
                        else:
                            action = run_actor(observations[-1])
                        next_observation, reward, done, info = env.step(action)
                        actions.append(action)
                        rewards.append(reward)

                        next_observation_preproc = preproc(next_observation)

                        global_num_frames.inc()
                        if done or len(observations) >= args.n_step:
                            assert len(observations) == len(actions) and len(actions) == len(rewards)
                            n = len(observations)
                            if done:
                                value_f = 0
                            else:
                                value_f = run_critic(next_observation_preproc)
                            discounted_rewards = [0 for i in range(n)]
                            for i in reversed(range(n)):
                                value_f = rewards[i] + args.gamma * value_f
                                discounted_rewards[i] = value_f

                            perform_update(np.array(observations), np.array(actions), np.array(discounted_rewards))
                            observations, actions, rewards = [], [], []

                        if not done:
                            observations.append(next_observation_preproc)

def main(args):
    with tf.device("cpu"):
        session_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=args.num_threads)
        session = tf.Session(config=session_config)

        env = gym.make(args.env)
        assert type(env.observation_space) is gym.spaces.Box
        input_shape = env.observation_space.shape
        assert type(env.action_space) is gym.spaces.Discrete
        num_actions = env.action_space.n
        env.close()

        global_num_frames = AtomicInt()
        preproc_graphs = SharedResource([preproc_graph(session, input_shape)
                                         for _ in range(args.num_threads)])
        a3c_graphs     = SharedResource([a3c_graph(args, session, num_actions, reuse=(i > 0))
                                         for i in range(args.num_threads)])

        session.run(tf.initialize_all_variables())

        threads = []

        for thread_idx in range(args.num_threads):
            thread = Thread(target=run_simulator, args=(args, global_num_frames, preproc_graphs, a3c_graphs))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

def parse_args():
    parser = argparse.ArgumentParser(description='Hogwild training on MNIST.')
    parser.add_argument('--num_threads', type=int,   default=9,         help='number of threads to use')
    parser.add_argument('--max_frames',  type=int,   default=2e8,       help='max number of game frames to learn for.')
    parser.add_argument('--n_step',      type=int,   default=5,         help='steps of simulation between every update.')
    parser.add_argument('--context',     type=int,   default=4,         help='how many past frames to use as input to the network.')
    parser.add_argument('--lr',          type=float, default=0.001,     help='learning rate.')
    parser.add_argument('--beta',        type=float, default=0.01,      help='coefficient in front of entropy regularization.')
    parser.add_argument('--gamma',       type=float, default=0.99,      help='discount_factor.')
    parser.add_argument('--env',         type=str,   default='Pong-v0', help='Which OpenAI gym env to use.')
    parser.add_argument('--exploration', type=str,   default='0.4:1:0.1:4e6-0.3:1:0.01:4e6-0.3:1:0.5:4e6',
            help='Exploration schedule. List of hyphen separated schedules. Each schedule consist of four colon spearated numbers. Probability of selecting that schedule, initial probability, final probability and number of frames over which the probability is linearly annealed between initial and final value.')
    args = parser.parse_args()

    print(str(ExplorationSchedule(args.exploration)))

    return args

if __name__ == '__main__':
    main(parse_args())
