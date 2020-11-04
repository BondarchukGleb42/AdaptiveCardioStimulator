import pygame as pg
import pymunk.pygame_util
import random
import drower
from pymunk import Vec2d
import math
import numpy as np
import blood_volume_generator
import warnings
warnings.filterwarnings("ignore")

pymunk.pygame_util.positive_y_is_up = False
pg.font.init()
font = pg.font.Font(None, 100)

RES = WIDTH, HEIGHT = 800, 800
FPS = 60
_FPS = 1 / FPS

LOW_SHELF = 55
HIGH_SHELF = 140

pg.init()
surface = pg.display.set_mode(RES)
clock = pg.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(surface)

space = pymunk.Space()
space.gravity = 0, 2500
blood_v = 0

collision_types = {
    "blood_cells": 1,
    "cleaner": 2
}

drower.collision_types = collision_types
drower.space = space
drower.draw_heart()
blood_v = 0

target_bpm = 0


def analyze_br_func(x):
    return round(min((1 / (1 + math.exp(-(x / 6 - 4.4)))) * 0.0403 * x + 0.1, 2.2), 4)


analyze_br = [analyze_br_func(i) for i in range(1, 61)]


def generate_bpm_history():
    r = random.randint(LOW_SHELF + 6, HIGH_SHELF - 10)
    print(f'Current BPM: %dbpm' % r)
    return [r / 60 for i in range(60)]


def calc_reward_func(target_beat_rate):
    x1, y1 = target_beat_rate - 10, 0
    x2, y2 = target_beat_rate, 1
    x3, y3 = target_beat_rate + 10, 0

    a = (y3 - ((x3 * (y2 - y1) + x2 * y1 - x1 * y2) / (x2 - x1))) / (x3 * (x3 - x1 - x2) + x1 * x2)
    b = ((y2 - y1) / (x2 - x1)) - a * (x1 + x2)
    c = (x2 * y1 - x1 * y2) / (x2 - x1) + a * x1 * x2
    return a, b, c


a, b, c = calc_reward_func(60)


def get_reward(x):
    global a, b, c
    return round(a * x * x + b * x + c, 3)


def generate_target_bpm():
    global target_bpm
    r = random.randint(LOW_SHELF + 5, HIGH_SHELF - 25)
    print(f'Next objective: %dbpm' % r)
    target_bpm = r


class Temp_cleaner():
    def __init__(self):
        s1 = pymunk.Segment(space.static_body, (186, 547), (277, 652), 0.3)
        s2 = pymunk.Segment(space.static_body, (277, 652), (372, 710), 0.3)
        s3 = pymunk.Segment(space.static_body, (372, 710), (458, 740), 0.3)
        s3 = pymunk.Segment(space.static_body, (458, 740), (532, 746), 0.3)
        s4 = pymunk.Segment(space.static_body, (532, 746), (594, 685), 0.3)
        s5 = pymunk.Segment(space.static_body, (594, 685), (629, 576), 0.3)
        s6 = pymunk.Segment(space.static_body, (629, 576), (648, 518), 0.3)
        s7 = pymunk.Segment(space.static_body, (648, 518), (650, 368), 0.3)
        self.all_shapes = [s1, s2, s3, s4, s5, s6, s7]
        for i in range(len(self.all_shapes)):
            self.all_shapes[i].collision_type = collision_types["cleaner"]
            self.all_shapes[i].color = pg.color.THECOLORS["black"]

    def on(self):
        space.add(*self.all_shapes)

    def off(self):
        space.remove(*self.all_shapes)


temp_cleaner = Temp_cleaner()


class Blood_cell():
    def __init__(self, pos, impulse, color, moment=6.2, mass=0.5, radius=8):
        self.body = pymunk.Body(mass, moment)
        self.body.position = pos
        self.body.apply_impulse_at_local_point(Vec2d(impulse))
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.color = pg.color.THECOLORS[color]
        self.shape.collision_type = collision_types["blood_cells"]
        self.shape.elasticity = 0

    def spawn(self):
        space.add(self.body, self.shape)

    def destroy(self):
        space.remove(self.shape)


def spawn_blood_cell(pos, impulse, color):
    global blood_v
    for _ in range(2):
        new_cell = Blood_cell(pos, impulse, color)
        new_cell.spawn()
        blood_v += 1


def remove_blood_cell(arbiter, space, data):
    global blood_v
    cell_shape = arbiter.shapes[0]
    space.remove(cell_shape, cell_shape.body)
    blood_v -= 1
    return True


class Muscle():
    def __init__(self, from_, to_, vel):
        self.vel = vel
        self.from_ = from_
        self.to_ = to_

    def draw(self):
        self.body = pymunk.Body(500, pymunk.inf, pymunk.Body.KINEMATIC)
        self.body.position = ((self.from_[0] + self.from_[1]) / 350, (self.to_[0] + self.to_[1]) / 350)

        self.shape = pymunk.Segment(self.body, self.from_, self.to_, 4)
        self.shape.color = pg.color.THECOLORS["brown"]
        self.shape.elasticity = 0.8
        space.add(self.body, self.shape)
        self.is_use = False

    def up(self):
        self.body.velocity = (-self.vel[0], -self.vel[1])

    def down(self):
        self.body.velocity = self.vel

    def not_active(self):
        self.body.velocity = (0, 0)


class Border():
    def __init__(self, from_, to_):
        self.segment_shape = pymunk.Segment(space.static_body, from_, to_, 3)
        self.segment_shape.color = pg.color.THECOLORS["darkslategray"]

    def spawn(self):
        space.add(self.segment_shape)

    def destroy(self):
        space.remove(self.segment_shape)


class Heart():
    def __init__(self, borders, muscles, temp_cleaner):
        self.temp_cleaner = temp_cleaner
        self.borders = borders
        self.muscles = muscles
        self.t = 0
        self.is_use = False
        self.timing = generate_bpm_history()

    def use(self):
        self.is_use = True

        try:
            for border in self.borders:
                border.spawn()
        except:
            pass

    def update(self):
        if self.is_use:
            if self.t > 20:
                for border in self.borders:
                    border.destroy()
                self.t = 0
                self.is_use = False
                self.temp_cleaner.off()
            elif self.t <= 20 and self.is_use:
                if self.t == 1:
                    self.temp_cleaner.on()
                if self.t <= 10:
                    for muscle in muscles:
                        muscle.up()
                else:
                    for muscle in muscles:
                        muscle.down()
                self.t += 1
        else:
            for muscle in muscles:
                muscle.not_active()


h = space.add_collision_handler(
    collision_types["blood_cells"],
    collision_types["cleaner"])
h.begin = remove_blood_cell

bord1 = Border((650, 330), (518, 352))
bord2 = Border((297, 379), (186, 488))
borders = [bord1, bord2]

m1 = Muscle((178, 549), (272, 651), (-700, 800))
m2 = Muscle((272, 651), (391, 722), (0, 1100))
m3 = Muscle((391, 722), (462, 750), (0, 1200))
m4 = Muscle((462, 750), (546, 755), (0, 1200))
m5 = Muscle((543, 750), (519, 726), (0, 1200))
m6 = Muscle((519, 726), (475, 644), (0, 1200))
m7 = Muscle((475, 644), (431, 524), (0, 1200))
m8 = Muscle((431, 524), (412, 406), (0, 1200))
m9 = Muscle((546, 755), (609, 684), (0, 1200))
m10 = Muscle((609, 684), (638, 576), (700, 800))
m11 = Muscle((638, 576), (659, 527), (700, 800))
m12 = Muscle((657, 516), (661, 381), (500, 100))
muscles = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]

cleaner1 = drower.create_cleaner((303, 376), (413, 382))
cleaner2 = drower.create_cleaner((514, 354), (421, 404))
frames_timer = 0


def reset():
    global blood_v, heart, beat_rate, frames_timer, last_action

    space.remove(space.shapes)
    for m in muscles:
        m.draw()

    drower.draw_heart()
    heart = Heart(borders, muscles, temp_cleaner)

    cleaner1 = drower.create_cleaner((303, 376), (413, 382))
    cleaner2 = drower.create_cleaner((514, 354), (421, 404))

    blood_v = 0
    frames_timer = 0
    last_action = 0
    beat_rate = sum([analyze_br[i] * heart.timing[i] for i in range(60)])
    target_bpm = generate_target_bpm()


def step(action):
    global blood_v, beat_rate, frames_timer

    for _ in range(60):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                raise SystemExit

        spawn_blood_cell((215, 271), (0, 0), "blue")
        spawn_blood_cell((582, 187), (0, 0), "red")

        if not heart.is_use:
            if action == 1:
                if frames_timer % 30 == 0 and frames_timer != 0:
                    heart.use()
            elif action == 2:
                if frames_timer % 30 == 0:
                    heart.use()
            elif action == 3:
                if frames_timer % 15 == 0:
                    heart.use()

        heart.update()

        frames_timer += 1
        if frames_timer == 60:
            frames_timer = 0
            heart.timing.append(action)
            heart.timing.pop(0)

        beat_rate = sum([analyze_br[i] * heart.timing[i] for i in range(0, 60)])

        reward = get_reward(beat_rate)
        if (blood_v < 1100) and (55 < beat_rate < 140):
            done = False
        else:
            done = True

        pg.event.get()
        surface.fill(pg.Color("black"))
        text = font.render(str(int(beat_rate)), 5, (255, 180, 180))
        surface.blit(text, (650, 710))
        
        space.debug_draw(draw_options)
        pg.display.update()
        space.step(_FPS)

    return [beat_rate, blood_v], reward, done


def step_without_render(action):
    global beat_rate, blood_v, last_action
    
    blood_v = blood_volume_generator.predictor.predict(np.array([last_action, blood_v]).reshape(-1, 2))
    last_action = action
    
    heart.timing.append(action)
    heart.timing.pop(0)
    beat_rate = sum([analyze_br[i] * heart.timing[i] for i in range(0, 60)])
    
    reward = get_reward(beat_rate)
    if (blood_v < 1100) and (55 < beat_rate < 140):
        done = False
    else:
        done = True
    
    return [beat_rate, blood_v], reward, done