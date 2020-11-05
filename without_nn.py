import environment as env
import random
import numpy as np
import time

env.reset()
obs, reward, done = env.step_without_render(0)
action = 0
while True:
    random_n = random.randint(0, 400)
    if random_n > 180:
        action = 1
        if random_n > 280:
            action = 2
            if random_n > 380:
                action = 3
    else:
        action = 0
    obs, reward, done = env.step_without_render(action)
    
    print(f"Объём крови: {obs[1]}")
    print(f"Сердцебиение: {obs[0]}")
    print(f"Действие: {action}")
    print("------------------------------")
    
    if done:
        env.reset()