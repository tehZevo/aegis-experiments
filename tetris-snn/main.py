import time
from operator import itemgetter
import numpy as np
from nd_to_json import json_to_nd, nd_to_json
from protopost import protopost_client as ppcl
import cv2

REPEAT_ACTION = 4 #TODO: hardcoded repeat action N times
REPEAT_SNN_STEP = 1
POTENTIAL_SCALE = 1

TETRIS_HOST = "http://127.0.0.1:8080"
SNN_HOST = "http://127.0.0.1:8081"

TETRIS_STEP = lambda action: ppcl(f"{TETRIS_HOST}", action)
TETRIS_OBS = lambda: ppcl(f"{TETRIS_HOST}/obs")

SNN_STEP = lambda: ppcl(f"{SNN_HOST}")
SNN_ADD_POTENTIAL = lambda key, x: ppcl(f"{SNN_HOST}/add-potential", {"key":key, "potential":x})
SNN_GET_SPIKES = lambda key, shape: ppcl(f"{SNN_HOST}/get-spikes", {"key":key, "shape":shape})
SNN_REWARD = lambda r: ppcl(f"{SNN_HOST}/reward", r)

#get first observation
obs = TETRIS_OBS()
obs = json_to_nd(obs)

ACTION_SHAPE = [6] #for argmax

cv2.namedWindow("tetris", cv2.WINDOW_NORMAL)
cv2.namedWindow("action_spikes", cv2.WINDOW_NORMAL)

i = 0
while True:
  t = time.time()
  #add obs to snn potential
  SNN_ADD_POTENTIAL("obs", nd_to_json(obs * POTENTIAL_SCALE))
  # print("add potential", time.time() - t)

  #step snn
  t = time.time()
  for _ in range(REPEAT_SNN_STEP):
    SNN_STEP()
  # print("snn step", time.time() - t)

  #get spikes of snn and map to action space
  t = time.time()
  action_spikes = SNN_GET_SPIKES("action", ACTION_SHAPE)
  action_spikes = json_to_nd(action_spikes)
  action = int(np.argmax(action_spikes))
  # print("get action", time.time() - t)

  #step tetris environment
  t = time.time()
  reward = 0
  for _ in range(REPEAT_ACTION):
    result = TETRIS_STEP(action)
    obs, done, r, info = itemgetter("obs", "done", "reward", "info")(result)
    reward += r
  # print("step tetris", time.time() - t)

  obs = json_to_nd(obs)

  #reward snn
  t = time.time()
  SNN_REWARD(reward)
  i += 1
  print(f"{i}: {reward}")
  # print("reward snn", time.time() - t)

  cv2.imshow("action_spikes", action_spikes)
  cv2.imshow("tetris", obs)
  cv2.waitKey(1)
