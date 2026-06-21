## Causal Chess

Python implementation of a chess engine with the following properties:

- selective tree search (not brute force)
- neural network based eval function (conv layers)
- eval function is trained _during_ search:
  - list of moves is evaluated with eval function, resulting in scores p_t for each move
  - scores are 0..1
  - if draw or checkmate is detected during move generation, scores are set to 0.5 or 0 or 1 (black/white)
  - the best n moves with  are executed (monte carlo)
  - again a list of moves with score p_{t+1} in the resulting position is executed and evaluated.
  - eval of best answer-move is used to train the eval function with loss (p_t - p_{t+1})
  - tree search up to a given depth
  - continuous self-play

While searching, the value function is continuously updated.

Python: the chess move generation is probably less computational intensive than the continuous update of the value function?
