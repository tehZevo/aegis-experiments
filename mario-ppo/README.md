# Mario + Feature Extractor + PPO

Testing using a pretrained Keras app feature extractor to reduce the dimensionality of the Mario env's observation.

## Structure
![Diagram](diagram.drawio.png)

## Nodes used
*TODO*

## TODO
- add description of experiment
- document usage experiment
- document experiment results
- use randomsearch to find optimal hyperparameters
  - launch containers and link them with a network
- potentially combine with other mario-ppo experiments and use env vars to control curiosity and ae features on/off
  - this may require conditional compose file edits (eg obs size of PPO net..) so may not be a good idea
