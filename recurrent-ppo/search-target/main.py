from operator import itemgetter
import json, os, time
import numpy as np
from nd_to_json import json_to_nd, nd_to_json
from protopost import protopost_client as ppcl

REPEAT_ACTION = 4 #TODO: hardcoded repeat action N times
TRAINING_STEPS = 200000 #1000000 #TODO: hardcoded n steps
N_SCORE_EPISODES = 100 #use average total reward of last N episodes as the metric

MEMORY_VECTOR_SIZE = int(os.getenv("MEMORY_SIZE", 0))
BLACKOUT_RATE = float(os.getenv("BLACKOUT_RATE", 0))

ENV = lambda action=None: ppcl("http://env", action)
ENV_OBS = lambda: ppcl("http://env/obs")
GET_ACTION = lambda obs: ppcl("http://agent/step", obs)
GIVE_REWARD = lambda r: ppcl("http://agent/reward", r)

while True:
  try:
    GIVE_REWARD(0)
    break
  except Exception:
    print("Waiting for agent...")
    time.sleep(1)

while True:
  try:
    ENV_OBS()
    break
  except Exception:
    print("Waiting for env...")
    time.sleep(1)

def calc_last_n_avg():
  if len(last_n_episode_rewards) == 0:
    return float("nan")
  return  np.mean(last_n_episode_rewards).item()

#assumes obs is nd_to_json'd, but memory is not
def combine_obs(obs, memory):
  obs = json_to_nd(obs)

  #randomly black out observation
  if np.random.random() < BLACKOUT_RATE:
    obs = np.zeros_like(obs)

  obs = np.concatenate([memory, obs])
  obs = nd_to_json(obs)
  return obs

#assumes action is nd_to_json'd, returns action as nd_to_json'd, but leaves memory unconverted
def split_action(action):
  #parse
  action = json_to_nd(action)
  #split
  memory, action = action[:MEMORY_VECTOR_SIZE], action[MEMORY_VECTOR_SIZE:]
  #convert action
  action = nd_to_json(action)
  return action, memory

#get first observation
obs = ENV_OBS()
memory = np.zeros([MEMORY_VECTOR_SIZE])
#combine memory and obs
obs = combine_obs(obs, memory)

last_n_episode_rewards = []
episode_reward = 0
i = 0

while True:
  #get action
  action = GET_ACTION(obs)
  #split action
  action, memory = split_action(action)
  #step environment
  reward = 0
  for _ in range(REPEAT_ACTION):
    result = ENV(action)
    obs, done, r, info = itemgetter("obs", "done", "reward", "info")(result)
    episode_reward += r
    #append and reset episode reward, slice to last N episodes
    if done:
      last_n_episode_rewards.append(episode_reward)
      episode_reward = 0
      last_n_episode_rewards = last_n_episode_rewards[-N_SCORE_EPISODES:]
    reward += r

  #combine obs with memory again
  obs = combine_obs(obs, memory)
  #reward agent
  GIVE_REWARD(reward / 100.) #based on the range of rewards observed

  i += REPEAT_ACTION
  print(f"{i}/{TRAINING_STEPS}: {reward} ({calc_last_n_avg()} avg over last {N_SCORE_EPISODES} eps)")
  if i >= TRAINING_STEPS:
    break

#dump average of last N episodes
print(json.dumps({f"last_{N_SCORE_EPISODES}_eps_avg":calc_last_n_avg()}))
