import traceback

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Queue
from threading import Semaphore


class BlockOnFullThreadPool(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        max_workers = None
        if len(args) > 0:
            max_workers = args[0]
        elif 'max_workers' in kwargs:
            max_workers = kwargs['max_workers']
        queue_size = kwargs.pop('queue_size', 0)

        super(BlockOnFullThreadPool, self).__init__(*args, **kwargs)
        assert type(max_workers) is int and type(queue_size) is int
        self._block_on_full_sem = Semaphore(max_workers + queue_size)

    def submit(self, *args, **kwargs):
        self._block_on_full_sem.acquire()
        future = super(BlockOnFullThreadPool, self).submit(*args, **kwargs)
        future.add_done_callback(self._on_task_done)
        return future

    def _on_task_done(self, result):
        exp = result.exception()
        if exp is not None:
            msg = '-' * 80 + '\n'
            msg += 'Exception occured in ThreadPoolExecutor:\n'
            msg += ''.join(traceback.format_tb(exp.__traceback__))
            msg += exp.__class__.__name__ + ': ' + str(exp) + '\n'
            msg += '-' * 80
            print(msg)
        self._block_on_full_sem.release()



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



if __name__ == '__main__':
    import time

    def work(i):
        print('start work', i)
        time.sleep(1)
        if i == 5:
            print(1/0)
        print('end work', i)

    with BlockOnFullThreadPool(max_workers=2) as pool:
        for i in range(10):
            print('adding to queue')
            pool.submit(work, i)
        pool.shutdown(wait=True)
        print('done')
