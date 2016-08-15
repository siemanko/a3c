import gym
import time

from multiprocessing.managers import BaseManager, NamespaceProxy
from threading import Thread, Lock


def make_wraper(*args, **kwargs):
    env = gym.make(*args, **kwargs)
    env.get_action_space = lambda: env.action_space
    env.get_observation_space = lambda: env.observation_space
    return env

class GymOnProcess(BaseManager):
    pass

GymOnProcess.register('Env', make_wraper)

class AtomicInt(object):
    def __init__(self, value=0):
        self._value = 0
        self._lock  = Lock()

    def inc(self):
        with self._lock:
            self._value += 1

    def get(self):
        return self._value

def main():
    TARGET_ITERATIONS = 10000
    NUM_THREADS = 12
    num_iterations = AtomicInt()

    start_time = time.time()
    def run_simulator():
        with GymOnProcess() as gym_on_process:
            env = gym_on_process.Env('Pong-v0')
            action_space = env.get_action_space()
            while num_iterations.get() < TARGET_ITERATIONS:
                env.reset()
                done = False
                while not done and num_iterations.get() < TARGET_ITERATIONS:
                    # env.render()
                    _, _, done, _ = env.step(action_space.sample()) # take a random action
                    num_iterations.inc()

    threads = [Thread(target=run_simulator) for _ in range(NUM_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    total_time = time.time() - start_time

    print('Total time for %d iterations: %.3f seconds' % (num_iterations.get(), total_time))

if __name__ == '__main__':
    main()
