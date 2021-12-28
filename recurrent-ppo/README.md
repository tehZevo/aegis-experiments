# PPO with recurrent memory

Split PPO's action into "action" and "memory" parts, concat the observation with "memory" as input to PPO.

## Structure
![Diagram](diagram.drawio.png)

## Nodes used
*TODO*

## Questions
- How does explicitly passing a memory vector affect the performance of the agent in Markovian environments?
- Does the explicit memory vector allow for solving non-Markovian environments?

## TODO
- add description of experiment
- document usage
- document experiment results
- on "sentience"
- separate compose profiles for recurrent/nonrecurrent nodes to compare?
