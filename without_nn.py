import environment as env
import random
import numpy as np
import time


env.reset()
observation, reward, done, heart_is_use = env.step(0)
while True:
    
    random_n = random.randint(0, 400)
    if random_n > 200:
        action = 1
        if random_n > 300:
            action = 2
            if random_n > 380:
                action = 3
    else:
        action = 0
    
    if not heart_is_use:
        observation, reward, done, heart_is_use = env.step(action)
    else:
        _, _, _, heart_is_use = env.step(0)
    env.render()
    
    if done:
        env = env.reset()