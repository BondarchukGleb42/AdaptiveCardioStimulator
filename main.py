import environment as env
import random
import numpy as np
import time
import tensorflow as tf
import tf_slim as slim
from tensorflow.python.saved_model import tag_constants

try:
    xrange = xrange
except:
    xrange = range

WEIGHTS_PATH = "weights_learned_new/model.ckpt"


def calc_loss(rewards):
    loss = 0
    for i in range(len(rewards)):
        loss += abs(1 - rewards[i]) * i * 0.001
    return loss

WEIGHTS_PATH="weights_learned_solve.ckpt"


def discount_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # Ниже инициализирована feed-forward часть нейросети.
        # Агент оценивает состояние среды и совершает действие
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size,
                                      biases_initializer=None, activation_fn=tf.nn.relu)
        hidden_2 = slim.fully_connected(hidden, h_size,
                                       biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden_2, a_size,
                                           activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)  # выбор действия


        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0,
                                tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder

        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]),
                                             self.indexes)
        # функция потерь
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) *
                                    self.reward_holder)

        tvars = tf.trainable_variables()
        self.exported = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,
                                                          tvars))


tf.reset_default_graph()  # Очищаем граф tensorflow

myAgent = agent(lr=1e-2, s_size=2, a_size=1, h_size=16)  # Инициализируем агента
saver = tf.train.Saver() # Инициализируем модуль для импорта/экспорта весов (встр. tensorflow)

total_episodes = 1001
init = tf.global_variables_initializer()


# Запуск графа tensorflow
with tf.Session() as sess:
    sess.run(init)

    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    for episode in range(total_episodes):
        env.reset()
        s, r, d = env.step(0)
        running_rewards = []
        ep_history = []
        for j in range(500):
            # Выбрать действие на основе вероятностей, оцененных нейросетью
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)
            s1, r, d = env.step(a)  # Получить награду за совершенное действие
            running_rewards.append(r)
            loss = calc_loss(running_rewards)
            ep_history.append([s, a, loss, s1])
            s = s1    
            if d:
                # Обновить нейросеть
                ep_history = np.array(ep_history)
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1],
                             myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if episode != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders,
                                                      gradBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                break

        if episode % 500 == 0 and episode != 0:
            save_path = saver.save(sess, WEIGHTS_PATH)
