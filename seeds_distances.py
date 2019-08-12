# graph distances from real solution to NN solution.

from game_facilitator.SelectionGeneratorCuriosity import *
from tangrams import *
import tensorflow as tf
import numpy as np
import math
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib import style
import time

start_time = time.time()

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

#############################################


# 'curious_y_output_e5_1e5.pkl'
# 'curious_y_output_e5_1e3.pkl'
# 'curious_y_output_e5_1e2.pkl'
# 'curious_y_output_e5_1e0_tr50.pkl'
# 'curious_y_output_e6_1e5_2.pkl'

# 'curious_y_output_e8_1e5.pkl' - learn puzzle k, ref all puzzles
# 'curious_y_output_e7_1e5_2.pkl' - learn puzzles 1,...,k, ref all puzzles

###################
# Figure of 'Distance of each puzzle to learned solution' after learning 1,...,k

with open('curious_y_output_e7_1e5_2.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [selection_sequence, training_set_input, training_set_output, global_out_list, global_H_list] = pickle.load(f)


save = False
if save is True:
    sgc = SelectionGeneratorCuriosity()
    sgc.load_dif_levels(directory='.', world='w1')

    curric_puzzles = []
    curric_puzzles_arr = []
    for k in range(7):
        training_task, training_input, training_output = json_to_NN(sgc.paths[1][k])
        curric_puzzles.append(copy.deepcopy(training_output))
        curric_puzzles_arr.append(np.asarray(copy.deepcopy(training_output)))

    with open('curric_puzzles.pkl', 'wb') as f:
        pickle.dump([curric_puzzles, curric_puzzles_arr], f, pickle.HIGHEST_PROTOCOL)

with open('curric_puzzles.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [curric_puzzles, curric_puzzles_arr] = pickle.load(f)

#############################################
# load seeds with no learning

with open('curious_y_output_e9_1e5.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [selection_sequence, training_set_input, training_set_output, global_out_list_rand, global_H_list_rand] = pickle.load(f)

plt.figure()

distances = []
for k in range(7):
    out_arr = np.asarray(global_out_list_rand[k])
    dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[k],2)))
    distances.append(dist)
    plt.plot(global_out_list_rand[k], label=str(k))
plt.figure()
plt.plot(distances)
plt.xlabel('un-Learned puzzle k')
plt.ylabel('Distance')
plt.legend()
plt.title('Distance of each puzzle to un-learned solution')

plt.figure()
plt.plot(global_H_list_rand)
plt.xlabel('un-Learned puzzle k')
plt.ylabel('LE')
plt.legend()
plt.title('Learnability Estimation with no learning')

print(distances)

###################
# Figure of 'Distance of each puzzle to learned solution' after learning 1,...,k

with open('curious_y_output_e7_1e5_2.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [selection_sequence, training_set_input, training_set_output, global_out_list, global_H_list] = pickle.load(f)


plt.figure()

for puz in range(7):
    distances = []
    for k in range(49):
        out_arr = np.asarray(global_out_list[k])
        dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[puz],2)))
        distances.append(dist)
    plt.plot(distances[0::8], label='puzzle '+str(puz))
    plt.xlabel('Learned puzzles 1,...,k')
    plt.ylabel('Distance')
plt.legend()
plt.title('Distance of each puzzle to learned solution')
print(distances)


plt.figure()
dist_learned_current = []
dist_learned_previous = []
dist_learned_pre_previous = []

for puz in range(1,7):
    out_arr = np.asarray(global_out_list[(puz)*8])
    dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[puz],2)))
    dist_learned_current.append(dist)

    out_arr = np.asarray(global_out_list[(puz-1) * 8 + 1])
    dist = np.sqrt(np.sum(np.power(out_arr - curric_puzzles_arr[puz], 2)))
    dist_learned_previous.append(dist)

plt.bar([2,3,4,5,6,7], dist_learned_previous, align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned until previous')
# plt.plot(dist_learned_previous, label='learned prvious')
plt.bar([2+0.3,3+0.3,4+0.3,5+0.3,6+0.3,7+0.3],dist_learned_current, align='center' ,width=0.3, color = 'blue', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned until current')

# plt.plot(dist_learned_current, label='learned current')
plt.legend()
plt.xlabel('Current Puzzle')
plt.ylabel('Distance')
plt.title('Distance between real solution to seed solution')

 # dist_learned_current

plt.figure()
dist_learned_current = []
dist_learned_previous = []
dist_learned_pre_previous = []

for puz in range(2,7):
    out_arr = np.asarray(global_out_list[(puz)*8])
    dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[puz],2)))
    dist_learned_current.append(dist)

    out_arr = np.asarray(global_out_list[(puz-1) * 8 + 1])
    dist = np.sqrt(np.sum(np.power(out_arr - curric_puzzles_arr[puz], 2)))
    dist_learned_previous.append(dist)

    out_arr = np.asarray(global_out_list[(puz - 2) * 8 + 2])
    dist = np.sqrt(np.sum(np.power(out_arr - curric_puzzles_arr[puz], 2)))
    dist_learned_pre_previous.append(dist)

plt.bar([3,4,5,6,7], dist_learned_pre_previous, align='center' ,width=0.3, color = 'green', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned until pre-previous')
# plt.plot(dist_learned_previous, label='learned prvious')
plt.bar([3+0.3,4+0.3,5+0.3,6+0.3,7+0.3],dist_learned_previous, align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned until previous')
plt.bar([3+0.6,4+0.6,5+0.6,6+0.6,7+0.6],dist_learned_current, align='center' ,width=0.3, color = 'blue', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned until current')

# plt.plot(dist_learned_current, label='learned current')
plt.legend()
plt.xlabel('Current Puzzle')
plt.ylabel('Distance')
plt.title('Distance between real solution to seed solution')

################################
# Figure of distances between puzzle k (left box) and k+1 (middle box) from real solutions after learning puzzle 1,...,k

plt.figure()
dist_learned_left = []
dist_learned_middle = []
LE_left = []
LE_middle = []

for puz in range(0,6):
    out_arr = np.asarray(global_out_list[(puz)*8])
    dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[puz],2)))
    dist_learned_left.append(dist)
    LE_left.append(global_H_list[(puz)*8])

    out_arr = np.asarray(global_out_list[(puz) * 8 + 1])
    dist = np.sqrt(np.sum(np.power(out_arr - curric_puzzles_arr[puz+1], 2)))
    dist_learned_middle.append(dist)
    LE_middle.append(global_H_list[(puz) * 8 + 1])

plt.bar([1,2,3,4,5,6], dist_learned_left, align='center' ,width=0.3, color = 'green', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='left puzzle')
# plt.plot(dist_learned_previous, label='learned prvious')
plt.bar([1+0.3, 2+0.3, 3+0.3,4+0.3,5+0.3, 6+0.3],dist_learned_middle, align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='middle puzzle')

# plt.plot(dist_learned_current, label='learned current')
plt.legend()
plt.xlabel('Learned Puzzles 1,...,k')
plt.ylabel('Distance')
plt.title('Distance between real solution to seed solution')

plt.figure()
plt.bar([1,2,3,4,5,6], LE_left, align='center' ,width=0.3, color = 'green', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='left puzzle')
# plt.plot(dist_learned_previous, label='learned prvious')
plt.bar([1+0.3, 2+0.3, 3+0.3,4+0.3,5+0.3, 6+0.3],LE_middle, align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='middle puzzle')
plt.xlabel('Learned Puzzle 1,...,k')
plt.ylabel('Learnability Estimator')

# plt.plot(LE_left, label='left')
# plt.plot(LE_middle, label='middle')
plt.legend()


###################
# Figure of 'Distance of each puzzle to learned solution' after learning puzzle k

with open('curious_y_output_e8_1e5.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [selection_sequence, training_set_input, training_set_output, global_out_list, global_H_list] = pickle.load(f)


save = False
if save is True:
    sgc = SelectionGeneratorCuriosity()
    sgc.load_dif_levels(directory='.', world='w1')

    curric_puzzles = []
    curric_puzzles_arr = []
    for k in range(7):
        training_task, training_input, training_output = json_to_NN(sgc.paths[1][k])
        curric_puzzles.append(copy.deepcopy(training_output))
        curric_puzzles_arr.append(np.asarray(copy.deepcopy(training_output)))

    with open('curric_puzzles.pkl', 'wb') as f:
        pickle.dump([curric_puzzles, curric_puzzles_arr], f, pickle.HIGHEST_PROTOCOL)

with open('curric_puzzles.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [curric_puzzles, curric_puzzles_arr] = pickle.load(f)


plt.figure()

for puz in range(7):
    distances = []
    for k in range(49):
        out_arr = np.asarray(global_out_list[k])
        dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[puz],2)))
        distances.append(dist)
    plt.plot(distances[0::8], label='puzzle '+str(puz))
    plt.xlabel('Learned puzzle')
    plt.ylabel('Distance')
plt.legend()
plt.title('Distance of each puzzle to learned solution')
print(distances)


plt.figure()
dist_learned_current = []
dist_learned_previous = []
dist_learned_pre_previous = []

for puz in range(1,7):
    out_arr = np.asarray(global_out_list[(puz)*8])
    dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[puz],2)))
    dist_learned_current.append(dist)

    out_arr = np.asarray(global_out_list[(puz-1) * 8 + 1])
    dist = np.sqrt(np.sum(np.power(out_arr - curric_puzzles_arr[puz], 2)))
    dist_learned_previous.append(dist)

plt.bar([2,3,4,5,6,7], dist_learned_previous, align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned previous')
# plt.plot(dist_learned_previous, label='learned prvious')
plt.bar([2+0.3,3+0.3,4+0.3,5+0.3,6+0.3,7+0.3],dist_learned_current, align='center' ,width=0.3, color = 'blue', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned current')

# plt.plot(dist_learned_current, label='learned current')
plt.legend()
plt.xlabel('Current Puzzle')
plt.ylabel('Distance')
plt.title('Distance between real solution to seed solution')

plt.figure()
dist_learned_current = []
dist_learned_previous = []
dist_learned_pre_previous = []

for puz in range(2,7):
    out_arr = np.asarray(global_out_list[(puz)*8])
    dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[puz],2)))
    dist_learned_current.append(dist)

    out_arr = np.asarray(global_out_list[(puz-1) * 8 + 1])
    dist = np.sqrt(np.sum(np.power(out_arr - curric_puzzles_arr[puz], 2)))
    dist_learned_previous.append(dist)

    out_arr = np.asarray(global_out_list[(puz - 2) * 8 + 2])
    dist = np.sqrt(np.sum(np.power(out_arr - curric_puzzles_arr[puz], 2)))
    dist_learned_pre_previous.append(dist)

plt.bar([3,4,5,6,7], dist_learned_pre_previous, align='center' ,width=0.3, color = 'green', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned pre-previous')
# plt.plot(dist_learned_previous, label='learned prvious')
plt.bar([3+0.3,4+0.3,5+0.3,6+0.3,7+0.3],dist_learned_previous, align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned previous')
plt.bar([3+0.6,4+0.6,5+0.6,6+0.6,7+0.6],dist_learned_current, align='center' ,width=0.3, color = 'blue', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='learned current')

# plt.plot(dist_learned_current, label='learned current')
plt.legend()
plt.xlabel('Current Puzzle')
plt.ylabel('Distance')
plt.title('Distance between real solution to seed solution')

################################
# Figure of distances between puzzle k (left box) and k+1 (middle box) from real solutions after learning puzzle k

plt.figure()
dist_learned_left = []
dist_learned_middle = []
LE_left = []
LE_middle = []

for puz in range(0,6):
    out_arr = np.asarray(global_out_list[(puz)*8])
    dist = np.sqrt(np.sum(np.power(out_arr-curric_puzzles_arr[puz],2)))
    dist_learned_left.append(dist)
    LE_left.append(global_H_list[(puz)*8])

    out_arr = np.asarray(global_out_list[(puz) * 8 + 1])
    dist = np.sqrt(np.sum(np.power(out_arr - curric_puzzles_arr[puz+1], 2)))
    dist_learned_middle.append(dist)
    LE_middle.append(global_H_list[(puz) * 8 + 1])

plt.bar([1,2,3,4,5,6], dist_learned_left, align='center' ,width=0.3, color = 'green', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='left puzzle')
# plt.plot(dist_learned_previous, label='learned prvious')
plt.bar([1+0.3, 2+0.3, 3+0.3,4+0.3,5+0.3, 6+0.3],dist_learned_middle, align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='middle puzzle')

# plt.plot(dist_learned_current, label='learned current')
plt.legend()
plt.xlabel('Learned Puzzles k')
plt.ylabel('Distance')
plt.title('Distance between real solution to seed solution')

plt.figure()
plt.bar([1,2,3,4,5,6], LE_left, align='center' ,width=0.3, color = 'green', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='left puzzle')
# plt.plot(dist_learned_previous, label='learned prvious')
plt.bar([1+0.3, 2+0.3, 3+0.3,4+0.3,5+0.3, 6+0.3],LE_middle, align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='middle puzzle')
plt.xlabel('Learned Puzzle k')
plt.ylabel('Learnability Estimator')
plt.legend()




#
# plt.figure()
# plt.plot(distances)
print("--- %s seconds = %s minutes ---" % ((time.time() - start_time) , (time.time() - start_time)/60.0))

plt.show()