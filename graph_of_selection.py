from game_facilitator.SelectionGeneratorCuriosity import *
from tangrams import *
# import tensorflow as tf
import numpy as np
import math
import pickle
import random
import matplotlib.pyplot as plt
import time
import csv

plt.figure()

plt.rcParams.update({'font.size': 20})

# load data of selecting only left
with open('eval_set_activation6_30_runs_training.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    [seq_lens_act_T, lens_arr_act_T, seq_lens_act_F, lens_arr_act_F] = pickle.load(f)


# arr = lens_arr_act_T[:,:]
# the first column of F is with no training, and all T is with trainings
arr = np.concatenate((lens_arr_act_F[:,0:1], lens_arr_act_T[:,0:-1]), axis=1)
arr_0 = arr
[rows,columns] = arr.shape
sol_last_only_left =  arr.mean(axis=0)
sol_last_only_left_yerr = arr.std(axis=0, ddof=1)/np.sqrt(rows)

offset = 0
plt.bar([0+offset,1+offset,2+offset,3+offset,4+offset,5+offset,6+offset],sol_last_only_left, align='center' ,width=0.2, color = 'white', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='selected only left')
plt.errorbar([0+offset,1+offset,2+offset,3+offset,4+offset,5+offset,6+offset],sol_last_only_left , color = 'orange', capsize=7, yerr=sol_last_only_left_yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)
plt.xticks(np.arange(7), ('None', '1', '2', '3', '4','5','6'))

plt.xlabel('Round')
plt.ylabel('Moves to Solution')
# plt.title('Solving last puzzle after each step in real game.')
# plt.title('Solution Length on Curriculum Series')
# sgc.paths[1][puzzle]

# load data of selecting only left
with open('eval_set_activation7_30_runs_training.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    [seq_lens_act_T, lens_arr_act_T, seq_lens_act_F, lens_arr_act_F] = pickle.load(f)


# arr = lens_arr_act_T[:,:]
# the first column of F is with no training, and all T is with trainings
arr = np.concatenate((lens_arr_act_F[:,0:1], lens_arr_act_T[:,0:-1]), axis=1)
arr_1 = arr
[rows,columns] = arr.shape
sol_last_only_left =  arr.mean(axis=0)
sol_last_only_left_yerr = arr.std(axis=0, ddof=1)/np.sqrt(rows)

offset = 0.2
plt.bar([0+offset,1+offset,2+offset,3+offset,4+offset,5+offset,6+offset],sol_last_only_left, align='center' ,width=0.2, color = 'black',  edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='selected only middle')
plt.errorbar([0+offset,1+offset,2+offset,3+offset,4+offset,5+offset,6+offset],sol_last_only_left , color = 'orange', capsize=7, yerr=sol_last_only_left_yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)
plt.xticks(np.arange(7), ('None', '1', '2', '3', '4','5','6'))

plt.xlabel('Round')
plt.ylabel('Moves to Solution')
# plt.title('Solving last puzzle after each step in real game.')


# load data of real game
with open('eval_set_activation8_30_runs_training.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    [seq_lens_act_T, lens_arr_act_T, seq_lens_act_F, lens_arr_act_F] = pickle.load(f)


# arr = lens_arr_act_T[:,:]
# the first column of F is with no training, and all T is with trainings
arr = np.concatenate((lens_arr_act_F[:,0:1], lens_arr_act_T[:,0:-1]), axis=1)
arr_2 = arr
[rows,columns] = arr.shape
sol_last_real_game =  arr.mean(axis=0)
sol_last_real_game_yerr = arr.std(axis=0, ddof=1)/np.sqrt(rows)

offset = 0.4
plt.bar([0+offset,1+offset,2+offset,3+offset,4+offset,5+offset,6+offset],sol_last_real_game, align='center' ,width=0.2, color = 'grey', hatch='//',edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='actual selections')
plt.errorbar([0+offset,1+offset,2+offset,3+offset,4+offset,5+offset,6+offset],sol_last_real_game , color = 'orange', capsize=7, yerr=sol_last_real_game_yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)
plt.xticks(np.arange(7)+0.2, ('before learning', '1', '2', '3', '4','5','6'))

plt.legend()
plt.xlabel('Round')
plt.ylabel('Moves to Solution')
# plt.title('Solving last puzzle after each step in real game.')
# plt.title('Solution Length on Curriculum Series')
# sgc.paths[1][puzzle]

save = True
if save:
    with open('number_of_moves.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        first_row = ['number run', 'Condition (left=0, middle=1, actual=2)', 'round (0-no learning)', 'number of moves', 'real 7-puzzle OR ? 7-puzzle (0=real, 1=?)']
        writer.writerow(first_row)
        for k in range(np.size(arr_0,0)):
            for j in range(np.size(arr_0,1)):
                row = [k, 0, j, arr_0[k,j],0]
                writer.writerow(row)
        for k in range(np.size(arr_1,0)):
            for j in range(np.size(arr_1,1)):
                row = [k, 1, j, arr_1[k,j],0]
                writer.writerow(row)
        for k in range(np.size(arr_2,0)):
            for j in range(np.size(arr_2,1)):
                row = [k, 2, j, arr_2[k,j],0]
                writer.writerow(row)



    csvFile.close()




plt.show()
