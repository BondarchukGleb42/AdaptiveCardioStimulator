import environment as env
import random
import numpy as np
import time


env.reset()
observation, reward, done = env.step(0)
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
    print(action)
    
    observation, reward, done = env.step(action) 
    
    if done:
        env.reset()