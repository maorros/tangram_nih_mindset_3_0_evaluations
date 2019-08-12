# test effect of number of pieces on H.

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
epoch_num = 100   # should be 100000

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

for puzzle in range(7):
    world = 'w1'  # the world to be computed
    CONDITION = 'curious' #'curious'
    # CONDITION = 'not_curious'
    H_THRESH_CURIOUS = 0.1
    H_THRESH_NOT_CURIOUS = 0.5
    epoch_num = 100000  # should be 100000

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

    weights_1 = tf.Variable(tf.truncated_normal([input_size, num_hidden], stddev=1.0 / math.sqrt(float(num_hidden)), seed=1),
                            name='weights_1')
    biases_1 = tf.Variable(tf.zeros([num_hidden]), name='biases_1')

    #weights_2 = tf.Variable(tf.truncated_normal([num_hidden, output_size], stddev=1.0 / math.sqrt(float(num_hidden))),name='weights_2')
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
    # ------------------



    # THE OUTPUT
    selection_sequence = []
    solver_cache = {}

    H0_list = []
    H1_list = []
    H2_list = []
    thresh_list = []
    training_task, training_input, training_output = json_to_NN(sgc.paths[1][puzzle])

    training_set_input = []
    training_set_output = []

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

    #
    # # for all games:
    # for game in range(0, 6):
    #     sgc.player = 'Robot'
    #
    #     # selection:
    #     #   present 3,
    #     options = sgc.get_current_selection('Robot')
    #     selected = None
    #
    #     if game == 0:
    #         if CONDITION == 'curious':
    #             selected = 1
    #             thresh_list.append(H_THRESH_CURIOUS)
    #         else:
    #             selected = 0
    #             thresh_list.append(H_THRESH_NOT_CURIOUS)
    #     else:
    #         #   calculate H,
    #         with tf.Session() as sess:
    #             # sess.run(init)
    #             # Restore variables from disk.
    #             saver.restore(sess, path)
    #             # print("Model restored for H calculation.")
    #             H_list = []
    #             out_list = []
    #             for opt in options:  # exclude uncertainty
    #                 # convert opt to input to nn  ***
    #                 training_task, training_input, training_output = json_to_NN(opt[0])
    #
    #                 # saver.restore(sess, path)
    #                 temp_inp = np.array([training_input])
    #                 # temp_out = np.array([training_output[step]])
    #
    #                 out = sess.run([output], feed_dict={inp: temp_inp, label:None})
    #                 out = out[0][0]
    #                 H = np.sum((out - 1) * np.log2(1 - out) - out * np.log2(out)) / len(out)
    #                 H_list.append(H)
    #                 out_list.append(out)
    #                 global_H_list.append(H)
    #                 global_out_list.append(out)
    #
    #
    #         H0_list.append(H_list[0])
    #         H1_list.append(H_list[1])
    #         H2_list.append(H_list[2])
    #
    #         known_H_list = H_list[:-1]
    #         # select according to condition and H ==> (0,1,2)
    #         if CONDITION == 'curious':
    #             if max(known_H_list[0], known_H_list[1]) < H_THRESH_CURIOUS:
    #                 selected = 2
    #             elif known_H_list[0] > known_H_list[1]:
    #                 selected = 0
    #             else:
    #                 selected = 1
    #             # update threshold
    #             H_THRESH_CURIOUS = max(known_H_list) * 0.8
    #             thresh_list.append(H_THRESH_CURIOUS)
    #         elif CONDITION == 'not_curious':
    #             if min(known_H_list[0], known_H_list[1]) > H_THRESH_NOT_CURIOUS:
    #                 selected = 2
    #             elif known_H_list[0] < known_H_list[1]:
    #                 selected = 0
    #             else:
    #                 selected = 1
    #             # update threshold
    #             H_THRESH_NOT_CURIOUS = min(known_H_list) * 1.2
    #             thresh_list.append(H_THRESH_NOT_CURIOUS)
    #         print(H_list, selected)
    #         print('new H_THRESH_NOT_CURIOUS threshold: ', H_THRESH_NOT_CURIOUS)
    #         print('new H_THRESH_CURIOUS threshold: ', H_THRESH_CURIOUS)
    #
    #     selection_sequence.append(selected)
    #
    #
        # solve tangram --> solution (don't actually solve)
        # task = Task()
        # task.create_from_json(options[selected][0])
        # sol.set_initial_task(task)
        # if game > 0:
        #     sol.set_activation(out_list[selected])
        # sol.run_task(task, duration=300, stop=True)
        # seq = sol.get_seq_of_moves_v2(task_all_pieces)
        # print(seq)
        # print ('game: ', game)
        # # solver_cache[options[selected][0]] = seq
        # solver_cache[str(game+1)] = seq

    #     # create an input/output pair
    #     training_task, training_input, training_output = json_to_NN(options[selected][0])
    #
    #     # training
    #     #   add last solution to training set
    #     training_set_input.append(training_input)
    #     training_set_output.append(training_output)
    #
    #     #   train (with init network/without)
    #     print('start training...')
    #     with tf.Session() as sess:
    #         # init = tf.initialize_all_variables()
    #         # sess.run(init)
    #         saver.restore(sess, path)
    #         # print("Model restored for training.")
    #         tf.set_random_seed(1)
    #
    #         for epoch in range(epoch_num):
    #             mini_batch_inp = np.array(training_set_input)
    #             mini_batch_out = np.array(training_set_output)
    #             feed_dict = fill_feed_dict(mini_batch_inp, mini_batch_out)
    #             maor, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
    #             # if step % 100 == 0:
    #
    #             # print('loss {0}'.format(loss_value))
    #             # print epoch
    #
    #         save_path = saver.save(sess, path)
    #         # print('model saved after training game: ', game, ' and selection: ', selected)
    #     print('finished training.')
    #
    #     # update game ...
    #     sgc.update_game_result(player='Robot', user_selection=selected, game_result='S')
    # # print('k:',k)
    # print('H0:')
    # print(H0_list)
    # print('H1:')
    # print(H1_list)
    # print('H2:')
    # print(H2_list)
    # print('thresh_list')
    # print(thresh_list)
    # print('selection_seq:')
    # print(selection_sequence)

# save the relevant output to file

save = True
if save is True:
    if CONDITION == 'curious':
        with open('curious_y_output_e4.pkl', 'wb') as f:
            pickle.dump([selection_sequence,training_set_input,training_set_output, global_out_list, global_H_list], f, pickle.HIGHEST_PROTOCOL)
        # with open('../agent/' + 'solve_cache_curiosity_' + world + '.pkl', 'wb') as f:
        #     pickle.dump(solver_cache, f, pickle.HIGHEST_PROTOCOL)

    if CONDITION == 'not_curious':
        with open('not_curious_y_output_e4.pkl', 'wb') as f:
            pickle.dump([selection_sequence,training_set_input,training_set_output, global_out_list, global_H_list], f, pickle.HIGHEST_PROTOCOL)
        # with open('../agent/' + 'solve_cache_curiosity_non_' + world + '.pkl', 'wb') as f:
        #     pickle.dump(solver_cache, f, pickle.HIGHEST_PROTOCOL)


##################################################3

with open('curious_y_output_e4.pkl','rb') as f:  # Python 3: open(..., 'rb')
    [selection_sequence, training_set_input, training_set_output, global_out_list, global_H_list] = pickle.load(f)


plt.figure()
plt.subplot(5,3,1)
plt.plot(global_out_list[0])
plt.text(1,1,'H='+str(global_H_list[0]))
out = global_out_list[0]
H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
plt.plot(H)

plt.subplot(5,3,2)
plt.plot(global_out_list[1])
plt.text(1,1,'H='+str(global_H_list[1]))
out = global_out_list[1]
H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
plt.plot(H)

plt.subplot(5,3,3)
plt.plot(global_out_list[2])
plt.text(1,1,'H='+str(global_H_list[2]))
out = global_out_list[2]
H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
plt.plot(H)

plt.subplot(5,3,4)
plt.plot(global_out_list[3])
plt.text(1,1,'H='+str(global_H_list[3]))
out = global_out_list[3]
H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
plt.plot(H)

plt.subplot(5,3,5)
plt.plot(global_out_list[4])
plt.text(1,1,'H='+str(global_H_list[4]))
out = global_out_list[4]
H = (out - 1) * np.log2(1 - out) - out * np.log2(out)
plt.plot(H)

plt.show()