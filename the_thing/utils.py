import gym
import numpy as np

from contextlib import contextmanager
from multiprocessing.managers import BaseManager
from queue import Queue
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

class SharedResource(object):
    def __init__(self, objs):
        self.q = Queue()
        while len(objs) > 0:
            self.q.put(objs.pop())

    @contextmanager
    def lease(self):
        r = None
        try:
            r = self.q.get()
            yield r
        finally:
            if r is not None:
                self.q.put(r)

class ExplorationSchedule(object):
    def __init__(self, desc):
        self.schedules = []
        for schedule_desc in desc.split('-'):
            selection_p, initial, final, steps = schedule_desc.split(':')
            selection_p, initial, final = map(float, [selection_p, initial, final])
            steps = int(float(steps))

            assert 0 <= selection_p <= 1
            assert 0 <= initial <= 1
            assert 0 <= final <= 1
            self.schedules.append((selection_p, initial, final, steps))
        self.current_schedule = None

    def reset(self):
        choice_idx = np.random.choice(len(self.schedules), p=[s[0] for s in self.schedules])
        self.current_schedule = self.schedules[choice_idx]

    def should_be_random(self, steps):
        assert self.current_schedule is not None, "must call reset first"
        _, initial, final, final_steps = self.current_schedule
        fraction_complete = min(steps / final_steps, 1.0)
        current_p = initial + (final - initial) * fraction_complete
        return np.random.random() < current_p

    def __str__(self):
        ret = 'Exploration schedule:\n'
        for schedule in self.schedules:
            ret += '  - With p=%.2f, annealing from %.2f to %.2f over %d steps\n' % schedule
        return ret
