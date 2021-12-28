from operator import itemgetter
import numpy as np
from nd_to_json import json_to_nd, nd_to_json
from protopost import protopost_client as ppcl

REPEAT_ACTION = 4 #TODO: hardcoded repeat action N times

MARIO = lambda action=None: ppcl("http://mario", action)
SURPRISE = lambda x: ppcl("http://ae/surprise-and-train", x)
GET_ACTION = lambda obs: ppcl("http://ppo/step", obs)
GIVE_REWARD = lambda r: ppcl("http://ppo/reward", r)
FEATURE_EXTRACTOR = lambda img: ppcl("http://feature-extractor", img)

#get first observation
obs = itemgetter("obs")(MARIO())
#extract features using keras app
obs = FEATURE_EXTRACTOR(obs)

i = 0
while True:
  #get action
  action = GET_ACTION(obs)
  #step mario env
  actual_reward = 0
  for _ in range(REPEAT_ACTION):
    result = MARIO(action)
    obs, done, r, info = itemgetter("obs", "done", "reward", "info")(result)
    actual_reward += r / 15.
  #extract features using keras app
  obs = FEATURE_EXTRACTOR(obs)
  #get surprise factor and train AE
  surprise = SURPRISE(obs)
  #give surprise reward to agent
  GIVE_REWARD(surprise)
  i += 1
  print(f"{i}: {actual_reward}")
