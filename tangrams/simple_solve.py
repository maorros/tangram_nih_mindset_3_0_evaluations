from tangrams import *
import numpy as np
import random
import copy
# import matplotlib.pyplot as plt
import json

# you can find more puzzles in game_facilitator/tangram_paths.txt
# this is the task to be solved
json_task = '{"pieces": [["large triangle2", "180", "1 1"], ["square", "0", "2 0"], ["large triangle1", "0", "1 1"]], "size": "5 5"}'
task = Task()
task.create_from_json(json_task)

# this is a dummy task that contains all pieces and will be used later.
json_all_pieces = '{"pieces": [["square", "0", "0 0"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"], ["medium triangle", "0", "3 1"], ["large triangle2", "180", "1 1"]], "size": "5 5"}'
task_all_pieces = Task()
task_all_pieces.create_from_json(json_all_pieces)

#just to make sure that the solution will be the same for different runs
random.seed(1)
np.random.seed(1)

sol = Solver()
sol.set_initial_task(task)
sol.run_task(task, duration=300, stop=True)
seq = sol.get_seq_of_moves_v2(task_all_pieces)




