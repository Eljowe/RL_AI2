import base64
from functools import total_ordering
from lib2to3.pgen2 import driver
import os
import re
import time
from collections import deque
from io import BytesIO
from PIL import Image
import cv2
import gym
import numpy as np
from PIL import Image
from gym import spaces
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import cv2
from collections import namedtuple
import io
from time import sleep
from multiprocessing import Process, Queue
from stable_baselines3.common.vec_env import VecMonitor
#from custom_policy_dqn import CustomCnnPolicy, CustomMlpPolicy, CustomLnCnnPolicy, CustomLnMlpPolicy
import torch
import tf_slim as slim
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device("cuda")
x = torch.rand(5, 3)
x.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

class environment2048(gym.Env):

    def __init__(self,
                 screen_width,  # width of the compressed image
                 screen_height,  # height of the compressed image
                 chromedriver_path: str = 'chromedriver'
                 ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.chromedriver_path = chromedriver_path
        self.num_observation = 0
        self.score = 0

        self.action_space = spaces.Discrete(4)  # set of actions: do nothing, jump, down
        self.observation_space = spaces.Box(
            low=0,
            high=8192,
            shape=(4, 4),
            dtype=np.uint8
        )
        # connection to chrome
        _chrome_options = webdriver.ChromeOptions()
        _chrome_options.add_argument("--mute-audio")
        _chrome_options.add_argument("disable-infobars")
        _chrome_options.add_argument("window-size=600,900")
        #_chrome_options.add_argument("--headless")
        #_chrome_options.add_argument("--disable-gpu") # if running on Windows

        self._driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=_chrome_options)

        self.current_key = None
        # current state represented by 4 images
        self.state_queue = deque(maxlen=4)
        #kokeiltu vaihtaa 4->1
        self.actions_map = [
            Keys.ARROW_UP,  # up
            Keys.ARROW_DOWN,  # down
            Keys.ARROW_LEFT,
            Keys.ARROW_RIGHT
        ]
        action_chains = ActionChains(self._driver)
        self.keydown_actions = [action_chains.key_down(item) for item in self.actions_map]
        self.keyup_actions = [action_chains.key_up(item) for item in self.actions_map]

    def reset(self):
        try:
            self._driver.get('http://www.oispakaljaa.com/')
        except WebDriverException as e:
            print(e)

        self._driver.find_element(By.TAG_NAME, "body") \
            .send_keys(Keys.SPACE)


        WebDriverWait(self._driver, 10).until(
            EC.presence_of_element_located((
                By.CLASS_NAME,
                "katkoviesti"
            ))
        )

        # trigger game start
        self._driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)

        return self._next_observation()

    def _next_observation(self):
        grid2 = np.zeros((4,4)) 
        parent = self._driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div[6]')
        children = parent.find_elements(By.XPATH, "*")
        for child in children:
            try:
                info = child.get_attribute("class")
                grid2[int(info[28])-1][int(info[26])-1] = child.text
            except:
                pass

        """for row in grid2:
            print(row)"""
            #for printing grid for debug
        return grid2

    def _get_score(self):
        try:
            num = int(''.join(self._driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div/div[1]').text))
        except:
            num = 0
        return num

    def _get_done(self):
        text = ''
        try:
            text = self._driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div[4]/p').text
        except:
             pass
        if len(text) > 2:
            return True
        else:
            return False
        return self._driver.execute_script("return Runner.instance_.crashed")

    def step(self, action: int):
        self._driver.find_element(By.TAG_NAME, "body") \
            .send_keys(self.actions_map[action])

        obs = self._next_observation()

        done = self._get_done()
        if done:
            reward = -10
        else:
            reward = 0.01
        try:
            num = int(''.join(self._driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div/div[1]/div').text))
        except:
            num = 0
        reward += 0.5*num
        '''try:
            num = int(''.join(self._driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div/div[1]/div').text))
        except:
            num = 0
        score = self._get_score()
        reward = num*0.2
        if reward < 0:
            reward = 0
        if done:
            reward = -180'''
        time.sleep(.015)

        return obs, reward, done, {"score": self._get_score()}


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


import imageio

from tqdm import tqdm
models_dir = "models/PPO"
#models_dir = "DQNmodel/DQN"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if __name__ == '__main__':
    dir = "models/DQN"
    dir_path = f"{dir}/DQN.zip"
    env_lambda = lambda: environment2048(
        screen_width=4,
        screen_height=4,
        chromedriver_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "chromedriver"
        )
    )
    do_train = True
    Continue = False
    num_cpu = 1
    env = VecMonitor(SubprocVecEnv([env_lambda for i in range(num_cpu)]))

    if Continue and do_train:
        model_path = f"{models_dir}/rl_model_400000_steps"
        log_path = f"C:/Koodi/RL_AI/logs/PPO_2/"
        model = PPO.load(model_path, env=env, tensorboard_log=log_path)
        model.set_env(env)
        checkpoint_callback = CheckpointCallback(
            save_freq= 10000,
            save_path=dir
        )
        model.learn(
            total_timesteps=500000, log_interval=1, reset_num_timesteps=False
        )
        model.save(f"{models_dir}/{2221}")

    elif do_train and not Continue:
        checkpoint_callback = CheckpointCallback(
            save_freq= 50000,
            save_path=dir
        )
        model = PPO(
            policy="MlpPolicy",
            env = env,
            verbose=1,
            tensorboard_log="./logs/",
            n_epochs=12,
            n_steps=512,
            device='cuda'
        )
        """
        model = DQN(
            "MlpPolicy", 
            env, verbose=1,
            tensorboard_log="./logs/")
        """
        model.learn(
            total_timesteps=500000,           callback=[checkpoint_callback], log_interval=1
        )
        model.save(f"{models_dir}/{212}")
        '''TIMESTEPS = 500
        iters = 0
        for i in range(1,10):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=[checkpoint_callback], tb_log_name='PPO')
            model.save(f"{models_dir}/{TIMESTEPS*i*num_cpu}")'''
    elif not do_train:
        episodes = 5
        model_path = f"{models_dir}/rl_model_400000_steps"
        log_path = f"C:/Koodi/RL_AI/logs/PPO_2/"
        model = PPO.load(model_path, env=env, tensorboard_log=log_path)
        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()
    #model.save(f"{models_dir}/{num_cpu}")
    #model = PPO.load(f'{models_dir}/{num_cpu}.zip', env=env)
    exit()