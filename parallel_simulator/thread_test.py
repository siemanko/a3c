import gym
import time

from threading import Thread, Lock

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
    NUM_THREADS = 4
    num_iterations = AtomicInt()

    start_time = time.time()
    def run_simulator():
        while num_iterations.get() < TARGET_ITERATIONS:
            env = gym.make('Pong-v0')
            env.reset()
            done = False
            while not done and num_iterations.get() < TARGET_ITERATIONS:
                # env.render()
                _, _, done, _ = env.step(env.action_space.sample()) # take a random action
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
