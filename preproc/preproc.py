import gym
import matplotlib.pyplot as plt
import tensorflow        as tf
import time

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

def main():
    session = tf.Session()

    env = gym.make('Pong-v0')
    last_observation = env.reset()
    preproc_f = preproc_graph(session, env.observation_space.shape)

    fig, ax = plt.subplots(figsize=(6,6))
    plt.ion()

    for _ in range(1000):
        observation, _, _, _ = env.step(env.action_space.sample())
        print("wtf?")
        pp = preproc_f(last_observation, observation)
        print("wtf!")

        ax.imshow(pp[:,:,0])
        plt.pause(0.05)

        print("Let the bodies hit the floor")

        last_observation = observation


if __name__ == '__main__':
    main()
