from operator import itemgetter
import numpy as np
from nd_to_json import json_to_nd, nd_to_json
from protopost import protopost_client as ppcl

REPEAT_ACTION = 4 #TODO: hardcoded repeat action N times

ENV = lambda action=None: ppcl("http://env", action)
GET_ACTION = lambda obs: ppcl("http://agent/step", obs)
GIVE_REWARD = lambda r: ppcl("http://agent/reward", r)

#get first observation
obs = itemgetter("obs")(ENV())

i = 0
while True:
  #get action
  action = GET_ACTION(obs)
  #step environment
  reward = 0
  for _ in range(REPEAT_ACTION):
    result = ENV(action)
    obs, done, r, info = itemgetter("obs", "done", "reward", "info")(result)
    reward += r / 15. #15 is the maximum (clipped) reward of the environment per step
  #reward agent
  GIVE_REWARD(reward)
  i += 1
  print(f"{i}: {reward}")
