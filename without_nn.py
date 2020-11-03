import environment as env
import random
import numpy as np
import time
import pandas as pd

env.reset()
obs, reward, done = env.step(0)
action = 0
df = pd.read_csv("heart_info.csv", sep=";")
while True:
    random_n = random.randint(0, 400)
    if random_n > 180:
        action1 = 1
        if random_n > 280:
            action1 = 2
            if random_n > 380:
                action1 = 3
    else:
        action1 = 0
    
    obs1, reward, done = env.step(action1)
    new_row = pd.Series({"Action": action1, "bpm": obs1[0], "blood_volume": obs1[1],
                         "last_action": action, "last_bpm": obs[0], "last_blood_volume": obs[1]})
    df = df.append(new_row, ignore_index=True)
    df.to_csv("heart_info.csv", sep=";")
    
    obs, action = obs1, action1
    
    if done:
        env.reset()