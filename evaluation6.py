# test effect of training and number of pieces on H.

from game_facilitator.SelectionGeneratorCuriosity import *
from tangrams import *
import tensorflow as tf
import numpy as np
import math
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# auxiliary
#json_all_pieces = '{"pieces": [["square", "0", "0 0"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"], ["medium triangle", "0", "3 1"], ["large triangle2", "180", "1 1"]], "size": "5 5"}'
json_all_pieces = '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}'
task_all_pieces = Task()
task_all_pieces.create_from_json(json_all_pieces)
random.seed(1)
np.random.seed(1)

# initialization: (curious ==> 2, non-curoius ==>1)
world = 'w1' # the world to be computed
CONDITION = 'curious' #'curious'
# CONDITION = 'not_curious'
H_THRESH_CURIOUS = 0.1
H_THRESH_NOT_CURIOUS = 0.5
epoch_num = 100000   # should be 100000

sgc = SelectionGeneratorCuriosity()
sgc.load_dif_levels(directory='.', world = world)

sol = Solver()
task = Task()
#task.create_from_json('{"pieces": [["large triangle2", "270", "1 0"], ["medium triangle", "180", "2 2"], ["square", "0", "0 0"], ["small triangle1", "180", "3 2"], ["large triangle1", "0", "1 2"], ["parrallelogram", "90", "2 0"]], "size": "5 5"}')
task.create_from_json('{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}')
sol.set_initial_task(task)
#todo: check if the above is still relevant

# ---- tf -----
input_size = 289
num_hidden = 312  # 40
output_size = 312
learning_rate = 0.1

def fill_feed_dict(inputlabel, outlabel):
    feed_diction = {inp: inputlabel, label: outlabel}
    # feed_diction = {'input': [1]*100, 'label': [0]*100}
    return feed_diction

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

global_H_list = []
global_out_list = []
training_set_input = []
training_set_output = []

# initialize NN
world = 'w1'  # the world to be computed
CONDITION = 'curious'  # 'curious'
# CONDITION = 'not_curious'
H_THRESH_CURIOUS = 0.1
H_THRESH_NOT_CURIOUS = 0.5
# epoch_num = 1000  # should be 100000

sgc = SelectionGeneratorCuriosity()
sgc.load_dif_levels(directory='.', world=world)

sol = Solver()
task = Task()
# task.create_from_json('{"pieces": [["large triangle2", "270", "1 0"], ["medium triangle", "180", "2 2"], ["square", "0", "0 0"], ["small triangle1", "180", "3 2"], ["large triangle1", "0", "1 2"], ["parrallelogram", "90", "2 0"]], "size": "5 5"}')
task.create_from_json(
    '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}')
sol.set_initial_task(task)

tf.set_random_seed(1)
inp = tf.placeholder(tf.float32, shape=(None, input_size), name='input')
# label = tf.placeholder(tf.float32, shape=(None, output_size), name='label')
label = tf.placeholder(tf.float32, shape=None, name='label')

weights_1 = tf.Variable(
    tf.truncated_normal([input_size, num_hidden], stddev=1.0 / math.sqrt(float(num_hidden)), seed=1),
    name='weights_1')
biases_1 = tf.Variable(tf.zeros([num_hidden]), name='biases_1')

# weights_2 = tf.Variable(tf.truncated_normal([num_hidden, output_size], stddev=1.0 / math.sqrt(float(num_hidden))),name='weights_2')
# biases_2 = tf.Variable(tf.zeros([num_hidden]), name='biases_2')

pre_act = tf.matmul(inp, weights_1)
preactivation = pre_act + biases_1
activation = tf.nn.sigmoid(preactivation)
output = activation
loss = tf.reduce_mean(tf.square(output - label))
# loss = -tf.reduce_mean(output * tf.log(label))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

saver = tf.train.Saver()
# path = "/home/gorengordon/catkin_ws/src/tangram_nih_curiosity_sandbox/tangrams/curric.ckpt"
path = "./curric.ckpt"
tf.set_random_seed(1)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    tf.set_random_seed(1)
    sess.run(init)
    save_path = saver.save(sess, path)

for iter_puzzle in range(7):
    # ------------------
    # THE OUTPUT
    selection_sequence = []
    solver_cache = {}

    H0_list = []
    H1_list = []
    H2_list = []
    thresh_list = []
    #
    training_task, training_input, training_output = json_to_NN(sgc.paths[1][iter_puzzle])
    training_set_input.append(training_input)
    training_set_output.append(training_output)

    #   train (with init network/without)
    print('start training...')
    with tf.Session() as sess:
        # init = tf.initialize_all_variables()
        # sess.run(init)
        saver.restore(sess, path)
        # print("Model restored for training.")
        tf.set_random_seed(1)

        for epoch in range(epoch_num):
            mini_batch_inp = np.array(training_set_input)
            mini_batch_out = np.array(training_set_output)
            feed_dict = fill_feed_dict(mini_batch_inp, mini_batch_out)
            maor, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            # if step % 100 == 0:

            # print('loss {0}'.format(loss_value))
            # print epoch

        save_path = saver.save(sess, path)
        # print('model saved after training game: ', game, ' and selection: ', selected)
    print('finished training.')

    for ref_puzzle in range(7):
        print('iter:', iter_puzzle, ' ref:', ref_puzzle)
        training_task, training_input, training_output = json_to_NN(sgc.paths[1][ref_puzzle])

        with tf.Session() as sess:
        # # sess.run(init)
        #             # Restore variables from disk.
        #             saver.restore(sess, path)
            saver.restore(sess, path)
            temp_inp = np.array([training_input])
        # temp_out = np.array([training_output[step]])


            out = sess.run([output], feed_dict={inp: temp_inp, label: None})
            out = out[0][0]
            H = np.sum((out - 1) * np.log2(1 - out) - out * np.log2(out)) / len(out)
            # H_list.append(H)
            # out_list.append(out)
            global_H_list.append(H)
            global_out_list.append(out)


# save the relevant output to file

save = True
if save is True:
    if CONDITION == 'curious':
        with open('curious_y_output_e6_1e5_2.pkl', 'wb') as f:
            pickle.dump([selection_sequence,training_set_input,training_set_output, global_out_list, global_H_list], f, pickle.HIGHEST_PROTOCOL)
        # with open('../agent/' + 'solve_cache_curiosity_' + world + '.pkl', 'wb') as f:
        #     pickle.dump(solver_cache, f, pickle.HIGHEST_PROTOCOL)

    if CONDITION == 'not_curious':
        with open('not_curious_y_output_e6_1e5_2.pkl', 'wb') as f:
            pickle.dump([selection_sequence,training_set_input,training_set_output, global_out_list, global_H_list], f, pickle.HIGHEST_PROTOCOL)
        # with open('../agent/' + 'solve_cache_curiosity_non_' + world + '.pkl', 'wb') as f:
        #     pickle.dump(solver_cache, f, pickle.HIGHEST_PROTOCOL)


##################################################3

with open('curious_y_output_e6_1e5_2.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [selection_sequence, training_set_input, training_set_output, global_out_list, global_H_list] = pickle.load(f)


plt.figure()
plt.plot(global_H_list[0::7], label='1 piece')
plt.plot(global_H_list[1::7], label='2 pieces')
plt.plot(global_H_list[2::7], label='3 pieces')
plt.plot(global_H_list[3::7], label='4 pieces')
plt.plot(global_H_list[4::7], label='5 pieces')
plt.plot(global_H_list[5::7], label='6 pieces')
plt.plot(global_H_list[6::7], label='7 pieces')

plt.xticks([0,1,2,3,4,5,6],['1', '2', '3', '4', '5','6','7'])
plt.legend()
plt.ylabel('Learnability Estimator')
plt.xlabel('Puzzles learned from curriculum')
#


puzzle_n = 0
plt.figure()
plt.xlabel('Output Layer')
plt.ylabel('Neuron Value')
plt.suptitle('Puzzles Learned: '+str(puzzle_n+1))
for k in range(0+puzzle_n*7,7+puzzle_n*7):
    plt.subplot(7,1,k+1-puzzle_n*7)
    out = global_out_list[k]
    plt.plot(out)
    H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
    plt.plot(H)
    plt.text(1, 1, 'H=' + str(global_H_list[k]))
    plt.ylabel('inp puz: '+str(k+1-puzzle_n*7))
# plt.title('Puzzles Learned: '+str(puzzle_n+1))
plt.xlabel('Output Layer')
# plt.ylabel('Neuron Value')
# plt.figure()
# plt.subplot(5,3,1)
# plt.plot(global_out_list[0])
# plt.text(1,1,'H='+str(global_H_list[0]))
# out = global_out_list[0]
# H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
# plt.plot(H)
#
# plt.subplot(5,3,2)
# plt.plot(global_out_list[1])
# plt.text(1,1,'H='+str(global_H_list[1]))
# out = global_out_list[1]
# H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
# plt.plot(H)
#
# plt.subplot(5,3,3)
# plt.plot(global_out_list[2])
# plt.text(1,1,'H='+str(global_H_list[2]))
# out = global_out_list[2]
# H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
# plt.plot(H)
#
# plt.subplot(5,3,4)
# plt.plot(global_out_list[3])
# plt.text(1,1,'H='+str(global_H_list[3]))
# out = global_out_list[3]
# H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
# plt.plot(H)
#
# plt.subplot(5,3,5)
# plt.plot(global_out_list[4])
# plt.text(1,1,'H='+str(global_H_list[4]))
# out = global_out_list[4]
# H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
# plt.plot(H)
#
# plt.show()