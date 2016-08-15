# A3C implementation in Python


### Lessons learned
1. Hogwild - turns out that to run hogwild in Tensorflow one needs to create separate tensorflow graphs with shared variables. Example in `hogwild/mnist.py`
2. Parallel simulators with OpenAI gym - turns out that running more than 4 threads with Atari simulator in parallel is not supported. Check out `parallel_simulator/thread_test.py`. On second try, I forced gym to be run on a separate process, this way the bug does not occur (`parallel_simulator/process_test.py`)
