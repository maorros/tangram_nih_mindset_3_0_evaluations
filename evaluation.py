from game_facilitator.SelectionGeneratorCuriosity import *
from tangrams import *
# import tensorflow as tf
import numpy as np
import math
import pickle
import random
import matplotlib.pyplot as plt


world = 'w1' # the world to be computed
CONDITION = 'curious' #'curious'
sgc = SelectionGeneratorCuriosity()
sgc.load_dif_levels(directory='.', world = world)
json_all_pieces = '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}'
task_all_pieces = Task()
task_all_pieces.create_from_json(json_all_pieces)

worlds = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7','w8']
selection_sequence = []
solver_cache = {}
sol = Solver()
seq_lens = []
lens_arr = np.zeros((10,7))

for k in range(10):
    for world in range(1):
        sgc.load_dif_levels(directory='.', world=worlds[world])
        json_all_pieces = '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}'
        task_all_pieces = Task()
        task_all_pieces.create_from_json(json_all_pieces)
        for puzzle in range(7):
        # solve tangram --> solution (don't actually solve)
            print('k:', k, 'puzzle:',puzzle)
            task = Task()
            task.create_from_json(sgc.paths[1][puzzle])
            sol.set_initial_task(task)
            game = 1
            # if game > 0:
            #     sol.set_activation(out_list[selected])
            sol.run_task(task, duration=300, stop=True)
            seq = sol.get_seq_of_moves_v2(task_all_pieces)
            print(seq)
            print ('game: ', game)
            # solver_cache[options[selected][0]] = seq
            solver_cache[str(game+1)] = seq
            seq_lens.append(len(seq))
            lens_arr[k, puzzle] = len(seq)

print seq_lens
print lens_arr

# sgc.paths[0][puzzle]
# [6, 10, 70, 218, 217, 231, 81, 3, 8, 10, 225, 257, 128, 128, 7, 7, 49, 210, 143, 88, 90, 7, 8, 8, 148, 205, 152, 138, 7, 20, 39, 224, 176, 85, 124, 7, 48, 78, 241, 167, 142, 90, 6, 8, 10, 200, 202, 125, 228, 6, 25, 72, 214, 158, 141, 152, 7, 40, 10, 243, 192, 121, 112, 7, 27, 36, 217, 169, 100, 255]
#
# [[   6.   10.   70.  218.  217.  231.   81.]
#  [   3.    8.   10.  225.  257.  128.  128.]
#  [   7.    7.   49.  210.  143.   88.   90.]
#  [   7.    8.    8.  148.  205.  152.  138.]
#  [   7.   20.   39.  224.  176.   85.  124.]
#  [   7.   48.   78.  241.  167.  142.   90.]
#  [   6.    8.   10.  200.  202.  125.  228.]
#  [   6.   25.   72.  214.  158.  141.  152.]
#  [   7.   40.   10.  243.  192.  121.  112.]
#  [   7.   27.   36.  217.  169.  100.  255.]]

a = [[   4.,    5.,    7.,   11.,    7.,    9.,   57.],
 [   5.,    4.,    6.,    8.,    9.,   11.,  103.],
 [   4.,    5.,    6.,    9.,   34.,   29.,   87.],
 [   4.,    5.,    7.,    8.,   50.,   79.,   49.],
 [   4.,    5.,    5.,    8.,   29.,   61.,   29.],
 [   5.,    4.,    6.,    8.,    7.,   68.,   55.],
 [   4.,    4.,    6.,    8.,    7.,   10.,   60.],
 [   4.,    5.,    6.,    7.,    8.,   91.,  166.],
 [   5.,    6.,    7.,    8.,   46.,   97.,   65.],
 [   4.,    5.,    6.,    7.,   22.,   51.,   90.]]
arr = np.asarray(a)

[rows,columns] = arr.shape
arr.mean(axis=0)
yerr = arr.std(axis=0, ddof=1)/np.sqrt(rows)
plt.figure()
plt.bar([1,2,3,4,5,6,7],arr.mean(axis=0), align='center' ,width=0.5, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='Solution length')
plt.errorbar([1,2,3,4,5,6,7],arr.mean(axis=0) , color = 'orange', capsize=7, yerr=yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)
plt.xlabel('Curriculum Puzzles Series')
plt.ylabel('Moves to Solution')
# plt.title('Solution Length on Curriculum Series')
# sgc.paths[1][puzzle]



# sgc.update_game_result(player='Robot', user_selection=selected, game_result='S')
# 7, 8, 71, 257, 169, 109, 59, \
# 6, 6, 39, 56, 17, 105, 135, \
# 2, 5, 8, 148, 67, 62, 105, \
# 5, 5, 55, 66, 214, 211, 104, \
# 3, 5, 9, 210, 199, 150, 93, \
# 3, 16, 86, 224, 216, 198, 234, \
# 4, 3, 7, 11, 9, 56, 135

# [4, 5, 7, 11, 7, 9, 57, 5, 4, 6, 8, 9, 11, 103, 4, 5, 6, 9, 34, 29, 87, 4, 5, 7, 8, 50, 79, 49, 4, 5, 5, 8, 29, 61, 29, 5, 4, 6, 8, 7, 68, 55, 4, 4, 6, 8, 7, 10, 60, 4, 5, 6, 7, 8, 91, 166, 5, 6, 7, 8, 46, 97, 65, 4, 5, 6, 7, 22, 51, 90]
a = [[   4.,    5.,    7.,   11.,    7.,    9.,   57.],
 [   5.,    4.,    6.,    8.,    9.,   11.,  103.],
 [   4.,    5.,    6.,    9.,   34.,   29.,   87.],
 [   4.,    5.,    7.,    8.,   50.,   79.,   49.],
 [   4.,    5.,    5.,    8.,   29.,   61.,   29.],
 [   5.,    4.,    6.,    8.,    7.,   68.,   55.],
 [   4.,    4.,    6.,    8.,    7.,   10.,   60.],
 [   4.,    5.,    6.,    7.,    8.,   91.,  166.],
 [   5.,    6.,    7.,    8.,   46.,   97.,   65.],
 [   4.,    5.,    6.,    7.,   22.,   51.,   90.]]
