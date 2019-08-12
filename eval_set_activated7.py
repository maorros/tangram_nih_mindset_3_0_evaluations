# Solve the last puzzle, given you learned nothing/1/1,2/1,2,3/1,2,3,4/1,2,3,4,5/1,2,3,4,5,6/

from game_facilitator.SelectionGeneratorCuriosity import *
from tangrams import *
# import tensorflow as tf
import numpy as np
import math
import pickle
import random
import matplotlib.pyplot as plt
import time

def json_to_NN(json_str):
    sol = Solver()
    task = Task()
    task.create_from_json(json_str)
    sol.set_initial_task(task)

    # dictionary: key = name of node, value = number of node
    dic = {}
    for n in range(len(sol.networks[0].nodes)):
        # print sol.networks[0].nodes[n].name[0] + ' ' + sol.networks[0].nodes[n].name[1] + ' ' + sol.networks[0].nodes[n].name[2]
        dic[sol.networks[0].nodes[n].name[0] + ' ' + sol.networks[0].nodes[n].name[1] + ' ' + sol.networks[0].nodes[n].name[2]] = n

    # generate a random tangram with N pieces
    # task.random_task(sol.networks[0], number_pieces=number_pieces)
    training_task = task
    training_input = (np.minimum(task.x, 1)).flatten()  # only 0/1 (not 1,2,5)

    # solve the orignial task using the solution
    activation = np.zeros_like(sol.networks[0].a)
    for piece in task.solution:
        node_num = dic[piece.name[0] + ' ' + piece.name[1] + ' ' + piece.name[2]]
        activation[node_num] = 1
    training_output = activation
    return training_task, training_input, training_output

start_time = time.time()

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
n_runs = 30
seq_lens_act_T = []
lens_arr_act_T = np.zeros((n_runs,7))
seq_lens_act_F = []
lens_arr_act_F = np.zeros((n_runs,7))

# for k in range(10):
#     for puzzle in range(7):
#         # solve tangram --> solution (don't actually solve)
#         print('k:', k, 'puzzle:', puzzle)
#         task = Task()
#         task.create_from_json(sgc.paths[1][puzzle])
#         sol.set_initial_task(task)
#         game = 1
#         # if game > 0:
#         #     sol.set_activation(out_list[selected])
#         sol.run_task(task, duration=300, stop=True)
#         seq = sol.get_seq_of_moves_v2(task_all_pieces)
#         print(seq)
#         print ('game: ', game)
#         # solver_cache[options[selected][0]] = seq
#         solver_cache[str(game + 1)] = seq
#         seq_lens.append(len(seq))
#         lens_arr[k, puzzle] = len(seq)

with open('curious_y_output_e11_1e5.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [selection_sequence, training_set_input, training_set_output, global_out_list, global_H_list] = pickle.load(f)

# test the solution length for each puzzle k with and without the seed after learning puzzle k-1.

act = False
# for act in [False, True]:
for k in range(n_runs):
    for world in range(1):
        sgc.load_dif_levels(directory='.', world=worlds[world])
        json_all_pieces = '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}'
        task_all_pieces = Task()
        task_all_pieces.create_from_json(json_all_pieces)
        for puzzle in range(1):
        # solve tangram --> solution (don't actually solve)
            print('k:', k, 'puzzle:',puzzle)
            task = Task()
            task.create_from_json(sgc.paths[1][6])
            training_task, training_input, training_output = json_to_NN(sgc.paths[1][6])
            sol.set_initial_task(task)
            game = 1
            # if game > 0:
            # print('training_output', training_output)
            if act:
                # sol.set_activation(training_output)
                sol.set_activation(global_out_list[7*(puzzle)+6])
            sol.run_task(task, duration=300, stop=True)
            seq = sol.get_seq_of_moves_v2(task_all_pieces)
            # print(seq)
            # print ('game: ', game)
            # solver_cache[options[selected][0]] = seq
            solver_cache[str(game+1)] = seq
            if act:
                seq_lens_act_T.append(len(seq))
                lens_arr_act_T[k, puzzle] = len(seq)
            else:
                seq_lens_act_F.append(len(seq))
                lens_arr_act_F[k, puzzle] = len(seq)

act = True
# for act in [False, True]:
for k in range(n_runs):
    for world in range(1):
        sgc.load_dif_levels(directory='.', world=worlds[world])
        json_all_pieces = '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}'
        task_all_pieces = Task()
        task_all_pieces.create_from_json(json_all_pieces)
        for puzzle in range(0,6):
        # solve tangram --> solution (don't actually solve)
            print('k:', k, 'puzzle:',puzzle)
            task = Task()
            task.create_from_json(sgc.paths[1][6])
            training_task, training_input, training_output = json_to_NN(sgc.paths[1][6])
            sol.set_initial_task(task)
            game = 1
            # if game > 0:
            # print('training_output', training_output)
            if act:
                # sol.set_activation(training_output)
                sol.set_activation(global_out_list[7*(puzzle)+6])
            sol.run_task(task, duration=300, stop=True)
            seq = sol.get_seq_of_moves_v2(task_all_pieces)
            # print(seq)
            # print ('game: ', game)
            # solver_cache[options[selected][0]] = seq
            solver_cache[str(game+1)] = seq
            if act:
                seq_lens_act_T.append(len(seq))
                lens_arr_act_T[k, puzzle] = len(seq)
            else:
                seq_lens_act_F.append(len(seq))
                lens_arr_act_F[k, puzzle] = len(seq)

print seq_lens_act_T
print lens_arr_act_T
print seq_lens_act_F
print lens_arr_act_F


# 'eval_set_activation.pkl' - graph with correct activation (just move the pieces) vs. without activation
# 'eval_set_activation2_training.pkl' - solving puzzle k after learning puzzle k-1.

save = True
if save is True:
    with open('eval_set_activation7_30_runs_training.pkl', 'wb') as f:
        pickle.dump([seq_lens_act_T, lens_arr_act_T, seq_lens_act_F, lens_arr_act_F], f, pickle.HIGHEST_PROTOCOL)

print("--- %s seconds = %s minutes ---" % ((time.time() - start_time), (time.time() - start_time) / 60.0))


####################################################################3
import pickle

with open('eval_set_activation7_30_runs_training.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    [seq_lens_act_T, lens_arr_act_T, seq_lens_act_F, lens_arr_act_F] = pickle.load(f)


# arr = lens_arr_act_T[:,:]
# the first column of F is with no training, and all T is with trainings
arr = np.concatenate((lens_arr_act_F[:,0:1], lens_arr_act_T[:,0:-1]), axis=1)
[rows,columns] = arr.shape
arr.mean(axis=0)
yerr = arr.std(axis=0, ddof=1)/np.sqrt(rows)
plt.figure()
plt.bar([0,1,2,3,4,5,6],arr.mean(axis=0), align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='with activation')
plt.errorbar([0,1,2,3,4,5,6],arr.mean(axis=0) , color = 'orange', capsize=7, yerr=yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)
plt.xticks(np.arange(7), ('None', '1', '2', '3', '4','5','6'))
#
# arr = lens_arr_act_F[:,:]
# [rows,columns] = arr.shape
# arr.mean(axis=0)
# yerr = arr.std(axis=0, ddof=1)/np.sqrt(rows)
#
# plt.bar([1+0.3,2+0.3,3+0.3,4+0.3,5+0.3,6+0.3,7+0.3],arr.mean(axis=0), align='center' ,width=0.3, color = 'blue', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='no activation')
# plt.errorbar([1+0.3,2+0.3,3+0.3,4+0.3,5+0.3,6+0.3,7+0.3],arr.mean(axis=0) , color = 'orange', capsize=7, yerr=yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)

# plt.legend()
plt.xlabel('Round')
plt.ylabel('Moves to Solution')
plt.title('Solving last puzzle in curriculum. Selecting always Middle')
# plt.title('Solution Length on Curriculum Series')
# sgc.paths[1][puzzle]
plt.show()

# with activation
# [[  0.   4.   5.   8.  42.  63.  57.]
#  [  0.   3.   5.   8.  30. 104.  63.]
#  [  0.   3.   5.   8.   9.  43.  38.]
#  [  0.   3.   5.   7.  35.  64. 115.]
#  [  0.   3.   5.   7.   8.  42. 155.]
#  [  0.   3.   5.   7.   8.  41. 101.]
#  [  0.   3.   5.   7.   9.  67. 113.]
#  [  0.   3.   5.   8.   9.  11.  77.]
#  [  0.   3.   5.   8.   8.  10.  55.]
#  [  0.   3.   5.   8.  38.  71.  67.]]
#
# without activation
# [[  0.   7.   6.   7.   6. 106. 122.]
#  [  0.   4.   7.   9.   7.  35. 119.]
#  [  0.   4.   5.   7.  21.  58.  76.]
#  [  0.   3.   5.   7.   8.  89. 165.]
#  [  0.   3.   6.   8.   9.  45. 101.]
#  [  0.   3.   6.   7.  20.  54.  69.]
#  [  0.   5.   5.   7.  19.  92.  95.]
#  [  0.   4.   6.   9.  16. 109. 180.]
#  [  0.   4.   6.  11.   9.  51. 224.]
#  [  0.   5.   5.   8.   9.  74.  57.]]