import argparse
import tensorflow as tf

from threading import Thread

from utils import GymOnProcess, ExplorationSchedule, AtomicInt



def run_simulator(args, global_num_frames):
    with GymOnProcess() as gym_on_process:
        env = gym_on_process.Env(args.env)
        action_space = env.get_action_space()
        exploration = ExplorationSchedule(args.exploration)

        while global_num_frames.get() < args.max_frames:
            env.reset()
            exploration.reset()
            done = False
            episode_num_frames = 0

            while not done and global_num_frames.get() < args.max_frames:
                if exploration.should_be_random(global_num_frames.get()):
                    action = action_space.sample()
                else:
                    # TODO(szymon): select from policy
                    action = action_space.sample()
                observation, reward, done, info = env.step(action_space.sample()) # take a random action
                global_num_frames.inc()
                episode_num_frames += 1
                if done or episode_num_frames % args.n_step == 0:
                    # perform a3c update
                    pass

def main(args):
    with tf.device("cpu"):
        global_num_frames = AtomicInt()

        threads = []

        for thread_idx in range(args.num_threads):
            thread = Thread(target=run_simulator, args=(args, global_num_frames))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

def parse_args():
    parser = argparse.ArgumentParser(description='Hogwild training on MNIST.')
    parser.add_argument('--num_threads', type=int, default=9,         help='number of threads to use')
    parser.add_argument('--max_frames',  type=int, default=2e8,       help='max number of game frames to learn for.')
    parser.add_argument('--n_step',      type=int, default=4,         help='steps of simulation between every update.')
    parser.add_argument('--env',         type=str, default='Pong-v0', help='Which OpenAI gym env to use.')
    parser.add_argument('--exploration', type=str, default='0.4:1:0.1:4e6-0.3:1:0.01:4e6-0.3:1:0.5:4e6',
            help='Exploration schedule. List of hyphen separated schedules. Each schedule consist of four colon spearated numbers. Probability of selecting that schedule, initial probability, final probability and number of frames over which the probability is linearly annealed between initial and final value.')
    args = parser.parse_args()

    print(str(ExplorationSchedule(args.exploration)))

    return args

if __name__ == '__main__':
    main(parse_args())
