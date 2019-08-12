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
# CONDITION = 'curious' #'curious'
CONDITION = 'not_curious'
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


for k in range(10):
    world = 'w1'  # the world to be computed
    # CONDITION = 'curious' #'curious'
    CONDITION = 'not_curious'
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

    tf.set_random_seed(k)
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

    training_set_input = []
    training_set_output = []

    # THE OUTPUT
    selection_sequence = []
    solver_cache = {}

    H0_list = []
    H1_list = []
    H2_list = []
    thresh_list = []

# for all games:
    for game in range(0, 6):
        sgc.player = 'Robot'

        # selection:
        #   present 3,
        options = sgc.get_current_selection('Robot')
        selected = None

        if game == 0:
            if CONDITION == 'curious':
                selected = 1
                thresh_list.append(H_THRESH_CURIOUS)
            else:
                selected = 0
                thresh_list.append(H_THRESH_NOT_CURIOUS)
        else:
            #   calculate H,
            with tf.Session() as sess:
                # sess.run(init)
                # Restore variables from disk.
                saver.restore(sess, path)
                # print("Model restored for H calculation.")
                H_list = []
                out_list = []
                for opt in options:  # exclude uncertainty
                    # convert opt to input to nn  ***
                    training_task, training_input, training_output = json_to_NN(opt[0])

                    # saver.restore(sess, path)
                    temp_inp = np.array([training_input])
                    # temp_out = np.array([training_output[step]])

                    out = sess.run([output], feed_dict={inp: temp_inp, label:None})
                    out = out[0][0]
                    H = np.sum((out - 1) * np.log2(1 - out) - out * np.log2(out)) / len(out)
                    H_list.append(H)
                    out_list.append(out)

            H0_list.append(H_list[0])
            H1_list.append(H_list[1])
            H2_list.append(H_list[2])

            known_H_list = H_list[:-1]
            # select according to condition and H ==> (0,1,2)
            if CONDITION == 'curious':
                if max(known_H_list[0], known_H_list[1]) < H_THRESH_CURIOUS:
                    selected = 2
                elif known_H_list[0] > known_H_list[1]:
                    selected = 0
                else:
                    selected = 1
                # update threshold
                H_THRESH_CURIOUS = max(known_H_list) * 0.8
                thresh_list.append(H_THRESH_CURIOUS)
            elif CONDITION == 'not_curious':
                if min(known_H_list[0], known_H_list[1]) > H_THRESH_NOT_CURIOUS:
                    selected = 2
                elif known_H_list[0] < known_H_list[1]:
                    selected = 0
                else:
                    selected = 1
                # update threshold
                H_THRESH_NOT_CURIOUS = min(known_H_list) * 1.2
                thresh_list.append(H_THRESH_NOT_CURIOUS)
            print(H_list, selected)
            print('new H_THRESH_NOT_CURIOUS threshold: ', H_THRESH_NOT_CURIOUS)
            print('new H_THRESH_CURIOUS threshold: ', H_THRESH_CURIOUS)

        selection_sequence.append(selected)


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

        # create an input/output pair
        training_task, training_input, training_output = json_to_NN(options[selected][0])

        # training
        #   add last solution to training set
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

        # update game ...
        sgc.update_game_result(player='Robot', user_selection=selected, game_result='S')
    print('k:',k)
    print('H0:')
    print(H0_list)
    print('H1:')
    print(H1_list)
    print('H2:')
    print(H2_list)
    print('thresh_list')
    print(thresh_list)
    print('selection_seq:')
    print(selection_sequence)


save = False
if save is True:
    if CONDITION == 'curious':
        with open('../agent/' + 'selection_cache_curiosity_' + world + '.pkl', 'wb') as f:
            pickle.dump(selection_sequence, f, pickle.HIGHEST_PROTOCOL)
        with open('../agent/' + 'solve_cache_curiosity_' + world + '.pkl', 'wb') as f:
            pickle.dump(solver_cache, f, pickle.HIGHEST_PROTOCOL)

    if CONDITION == 'not_curious':
        with open('../agent/' + 'selection_cache_curiosity_non_' + world + '.pkl', 'wb') as f:
            pickle.dump(selection_sequence, f, pickle.HIGHEST_PROTOCOL)
        with open('../agent/' + 'solve_cache_curiosity_non_' + world + '.pkl', 'wb') as f:
            pickle.dump(solver_cache, f, pickle.HIGHEST_PROTOCOL)
# best results:
# learning_rate = 0.1
# epoch number = 100000 (always)
# use the previous learned network
# Therehold = max(H_list) * 0.8

# Curious cond
# H0:
# [0.11346905048076923, 0.08278946998791817, 0.07842263197287536, 0.06131742550776555, 0.05882688669057993]
# H1:
# [0.11395586453951322, 0.08430432050656049, 0.08022682483379658, 0.06171872676947178, 0.06166776021321615]
# H2:
# [0.2791171440711388, 0.09026265755677834, 0.06952023506164551, 0.06547288405589569, 0.05174011450547438]
# thresh_list
# [0.1, 0.09116469163161058, 0.06744345640524839, 0.06418145986703726, 0.049374981415577425, 0.04933420817057292]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0_list_ = [0.11346905048076923, 0.08278946998791817, 0.07842263197287536, 0.06131742550776555, 0.05882688669057993]
H1_list_ = [0.11395586453951322, 0.08430432050656049, 0.08022682483379658, 0.06171872676947178, 0.06166776021321615]
H2_list_ = [0.2791171440711388, 0.09026265755677834, 0.06952023506164551, 0.06547288405589569, 0.05174011450547438]
thresh_list_ = [0.1, 0.09116469163161058, 0.06744345640524839, 0.06418145986703726, 0.049374981415577425, 0.04933420817057292]

plt.figure()
plt.plot([0,1,2,3,4], H0_list_, label='H0_C+')
plt.plot([0,1,2,3,4], H1_list_, label='H1_C+')
# plt.plot([0,1,2,3,4], H2_list_, label='H2_C+')
plt.plot(range(5),thresh_list_[:-1], label='thresh_C+')
plt.legend()
plt.title('Curious')


# Not Curious
# start training...
# finished training.
# ([0.12987299454517853, 0.13164873612232697, 0.40515828743959087], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15584759345421423)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09876047036586663, 0.09913221995035808, 0.10317173982277894], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11851256443903996)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.08234661664718236, 0.08288803467383751, 0.08623511363298465], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09881593997661883)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07157079378763835, 0.07298611371945112, 0.07422620822221805], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08588495254516602)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06485632138374524, 0.0658725591806265, 0.06893178132864144], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07782758566049429)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# Backend TkAgg is interactive backend. Turning interactive mode on.
# H0:
# [0.12987299454517853, 0.09876047036586663, 0.08234661664718236, 0.07157079378763835, 0.06485632138374524]
# H1:
# [0.13164873612232697, 0.09913221995035808, 0.08288803467383751, 0.07298611371945112, 0.0658725591806265]
# H2:
# [0.40515828743959087, 0.10317173982277894, 0.08623511363298465, 0.07422620822221805, 0.06893178132864144]
# thresh_list
# [0.5]
# selection_seq:
# [0, 0, 0, 0, 0, 0]

#this list is of curious
#thresh_list_ = [0.1, 0.09116469163161058, 0.06744345640524839, 0.06418145986703726, 0.049374981415577425, 0.04933420817057292]
H0_list_ = [0.12987299454517853, 0.09876047036586663, 0.08234661664718236, 0.07157079378763835, 0.06485632138374524]
H1_list_ = [0.13164873612232697, 0.09913221995035808, 0.08288803467383751, 0.07298611371945112, 0.0658725591806265]
H2_list_ = [0.40515828743959087, 0.10317173982277894, 0.08623511363298465, 0.07422620822221805, 0.06893178132864144]
thresh_list_ = [0.5, 0.15584759345421423, 0.11851256443903996, 0.09881593997661883, 0.08588495254516602, 0.07782758566049429]

# plt.figure()
plt.plot([0,1,2,3,4], H0_list_, label='H0_C-')
plt.plot([0,1,2,3,4], H1_list_, label='H1_C-')
# plt.plot([0,1,2,3,4], H2_list_, label='H2_C-')
plt.plot(range(5),thresh_list_[:-1], label='thresh_C-')
plt.legend()
plt.title('Not Curious')

# plt.plot([0.11346905048076923, 0.08278946998791817, 0.07842263197287536, 0.06131742550776555, 0.05882688669057993],
#          [0.11395586453951322, 0.08430432050656049, 0.08022682483379658, 0.06171872676947178, 0.06166776021321615])



# long run

H0 = np.zeros([10,5])
H1 = np.zeros([10,5])
H2 = np.zeros([10,5])
thresh_list = np.zeros([10,6])

# start training...
# finished training.
# ([0.11347271845890926, 0.11518041904156025, 0.27849415021064955], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.0921443352332482)
# start training...
# finished training.
# ([0.08348496754964192, 0.08556005893609463, 0.08861489173693535], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.0684480471488757)
# start training...
# finished training.
# ([0.07981277123475686, 0.08119863118880834, 0.07129629453023274], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06495890495104667)
# start training...
# finished training.
# ([0.06151482386466784, 0.06284719247084397, 0.06813911291269156], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.05027775397667518)
# start training...
# finished training.
# ([0.05914945480151054, 0.06343544446505033, 0.053946929100232244], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.050748355572040264)
# start training...
# finished training.
# ('k:', 0)
# H0:
H0[0,:] = [0.11347271845890926, 0.08348496754964192, 0.07981277123475686, 0.06151482386466784, 0.05914945480151054]
# H1:
H1[0,:] = [0.11518041904156025, 0.08556005893609463, 0.08119863118880834, 0.06284719247084397, 0.06343544446505033]
# H2:
H2[0,:] = [0.27849415021064955, 0.08861489173693535, 0.07129629453023274, 0.06813911291269156, 0.053946929100232244]
# thresh_list
thresh_list[0,:] = [0.1, 0.0921443352332482, 0.0684480471488757, 0.06495890495104667, 0.05027775397667518, 0.050748355572040264]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.11346905048076923, 0.11395586453951322, 0.2791171440711388], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09116469163161058)
# start training...
# finished training.
# ([0.08278946998791817, 0.08430432050656049, 0.09026265755677834], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06744345640524839)
# start training...
# finished training.
# ([0.07842263197287536, 0.08022682483379658, 0.06952023506164551], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06418145986703726)
# start training...
# finished training.
# ([0.06131742550776555, 0.06171872676947178, 0.06547288405589569], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.049374981415577425)
# start training...
# finished training.
# ([0.05882688669057993, 0.06166776021321615, 0.05174011450547438], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.04933420817057292)
# start training...
# finished training.
# ('k:', 1)
# H0:
H0[1,:] =  [0.11346905048076923, 0.08278946998791817, 0.07842263197287536, 0.06131742550776555, 0.05882688669057993]
# H1:
H1[1,:] =  [0.11395586453951322, 0.08430432050656049, 0.08022682483379658, 0.06171872676947178, 0.06166776021321615]
# H2:
H2[1,:] =  [0.2791171440711388, 0.09026265755677834, 0.06952023506164551, 0.06547288405589569, 0.05174011450547438]
# thresh_list
thresh_list[1,:] = [0.1, 0.09116469163161058, 0.06744345640524839, 0.06418145986703726, 0.049374981415577425, 0.04933420817057292]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.1134634507008088, 0.11554634876740284, 0.2789236704508464], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09243707901392229)
# start training...
# finished training.
# ([0.08362185649382763, 0.0846700912866837, 0.088746890043601], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06773607302934696)
# start training...
# finished training.
# ([0.07877421379089355, 0.08066952534210987, 0.06920414093213204], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06453562027368791)
# start training...
# finished training.
# ([0.061670144399007164, 0.06276901563008626, 0.06521005508227226], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.05021521250406901)
# start training...
# finished training.
# ([0.05937407566950871, 0.06276965141296387, 0.05210334826738407], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.050215721130371094)
# start training...
# finished training.
# ('k:', 2)
# H0:
H0[2,:] =  [0.1134634507008088, 0.08362185649382763, 0.07877421379089355, 0.061670144399007164, 0.05937407566950871]
# H1:
H1[2, :] = [0.11554634876740284, 0.0846700912866837, 0.08066952534210987, 0.06276901563008626, 0.06276965141296387]
# H2:
H2[2, :] = [0.2789236704508464, 0.088746890043601, 0.06920414093213204, 0.06521005508227226, 0.05210334826738407]
# thresh_list
thresh_list[2,:] = [0.1, 0.09243707901392229, 0.06773607302934696, 0.06453562027368791, 0.05021521250406901, 0.050215721130371094]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.11347570174779648, 0.11525513575627254, 0.28731835194123095], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09220410860501804)
# start training...
# finished training.
# ([0.0834201666025015, 0.08606820228772286, 0.09056526575333033], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06885456183017828)
# start training...
# finished training.
# ([0.07999642690022786, 0.08215477527716221, 0.0713639381604317], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06572382022172978)
# start training...
# finished training.
# ([0.06200741498898237, 0.06441116944337502, 0.06852195201775967], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.05152893555470002)
# start training...
# finished training.
# ([0.06141057992592836, 0.06469203264285357, 0.053504301951481745], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.05175362611428286)
# start training...
# finished training.
# ('k:', 3)
# H0:
H0[3,:] =  [0.11347570174779648, 0.0834201666025015, 0.07999642690022786, 0.06200741498898237, 0.06141057992592836]
# H1:
H1[3,:] = [0.11525513575627254, 0.08606820228772286, 0.08215477527716221, 0.06441116944337502, 0.06469203264285357]
# H2:
H2[3, :] = [0.28731835194123095, 0.09056526575333033, 0.0713639381604317, 0.06852195201775967, 0.053504301951481745]
# thresh_list
thresh_list[3, :] = [0.1, 0.09220410860501804, 0.06885456183017828, 0.06572382022172978, 0.05152893555470002, 0.05175362611428286]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.11346798676710862, 0.11495647674951798, 0.27932079021747297], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09196518139961439)
# start training...
# finished training.
# ([0.08328893857124524, 0.08324316220405774, 0.08879892031351726], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.0666311508569962)
# start training...
# finished training.
# ([0.07770167864285983, 0.08025005536201672, 0.06989843417436649], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06420004428961339)
# start training...
# finished training.
# ([0.06077707119477101, 0.06049899565867889, 0.06620358198116987], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.04862165695581681)
# start training...
# finished training.
# ([0.05785866272755158, 0.06050458321204552, 0.0514520620688414], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.04840366656963642)
# start training...
# finished training.
# ('k:', 4)
# H0:
H0[4,:] =  [0.11346798676710862, 0.08328893857124524, 0.07770167864285983, 0.06077707119477101, 0.05785866272755158]
# H1:
H1[4, :] = [0.11495647674951798, 0.08324316220405774, 0.08025005536201672, 0.06049899565867889, 0.06050458321204552]
# H2:
H2[4, :] = [0.27932079021747297, 0.08879892031351726, 0.06989843417436649, 0.06620358198116987, 0.0514520620688414]
# thresh_list
thresh_list[4,:] = [0.1, 0.09196518139961439, 0.0666311508569962, 0.06420004428961339, 0.04862165695581681, 0.04840366656963642]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.11347249838022086, 0.11399271549322666, 0.27597490946451825], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09119417239458133)
# start training...
# finished training.
# ([0.08283067360902444, 0.0833291518382537, 0.08762673842601287], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06666332147060297)
# start training...
# finished training.
# ([0.07844160764645307, 0.07971666409419133, 0.06841120964441544], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06377333127535306)
# start training...
# finished training.
# ([0.06059177105243389, 0.06151855908907377, 0.06490301474546775], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.04921484727125902)
# start training...
# finished training.
# ([0.05843828274653508, 0.0630810138506767, 0.052424699832231574], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.050464811080541364)
# start training...
# finished training.
# ('k:', 5)
# H0:
H0[5,:] =  [0.11347249838022086, 0.08283067360902444, 0.07844160764645307, 0.06059177105243389, 0.05843828274653508]
# H1:
H1[5,:] = [0.11399271549322666, 0.0833291518382537, 0.07971666409419133, 0.06151855908907377, 0.0630810138506767]
# H2:
H2[5, :] = [0.27597490946451825, 0.08762673842601287, 0.06841120964441544, 0.06490301474546775, 0.052424699832231574]
# thresh_list
thresh_list[5,:] = [0.1, 0.09119417239458133, 0.06666332147060297, 0.06377333127535306, 0.04921484727125902, 0.050464811080541364]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.11346985743596004, 0.11451553687071189, 0.2766713117941832], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09161242949656952)
# start training...
# finished training.
# ([0.0831148685553135, 0.08362645369309646, 0.08783264649220002], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06690116295447716)
# start training...
# finished training.
# ([0.07845414601839505, 0.0792690790616549, 0.0690454458579039], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06341526324932392)
# start training...
# finished training.
# ([0.06043700682811248, 0.060593476662269004, 0.06736921652769431], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.048474781329815204)
# start training...
# finished training.
# ([0.058368866260235124, 0.0625155461140168, 0.051748997125870146], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.050012436891213444)
# start training...
# finished training.
# ('k:', 6)
# H0:
H0[6,:] =  [0.11346985743596004, 0.0831148685553135, 0.07845414601839505, 0.06043700682811248, 0.058368866260235124]
# H1:
H1[6, :] = [0.11451553687071189, 0.08362645369309646, 0.0792690790616549, 0.060593476662269004, 0.0625155461140168]
# H2:
H2[6, :] = [0.2766713117941832, 0.08783264649220002, 0.0690454458579039, 0.06736921652769431, 0.051748997125870146]
# thresh_list
thresh_list[6,:] = [0.1, 0.09161242949656952, 0.06690116295447716, 0.06341526324932392, 0.048474781329815204, 0.050012436891213444]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.11347247392703326, 0.11432971709813827, 0.289613528129382], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09146377367851062)
# start training...
# finished training.
# ([0.08301143768506172, 0.08443788381723258, 0.09309357251876439], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06755030705378606)
# start training...
# finished training.
# ([0.07809860278398563, 0.08099971062097794, 0.07135255520160382], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06479976849678236)
# start training...
# finished training.
# ([0.06117083476139949, 0.06137644938933544, 0.06801094764318222], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.04910115951146835)
# start training...
# finished training.
# ([0.05843694393451397, 0.062082877525916465, 0.05352485485565968], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.049666302020733175)
# start training...
# finished training.
# ('k:', 7)
# H0:
H0[7,:] =  [0.11347247392703326, 0.08301143768506172, 0.07809860278398563, 0.06117083476139949, 0.05843694393451397]
# H1:
H1[7,:] = [0.11432971709813827, 0.08443788381723258, 0.08099971062097794, 0.06137644938933544, 0.062082877525916465]
# H2:
H2[7, :] = [0.289613528129382, 0.09309357251876439, 0.07135255520160382, 0.06801094764318222, 0.05352485485565968]
# thresh_list
thresh_list[7, :] = [0.1, 0.09146377367851062, 0.06755030705378606, 0.06479976849678236, 0.04910115951146835, 0.049666302020733175]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.11347348873431866, 0.11417914659549029, 0.2867718231983674], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09134331727639224)
# start training...
# finished training.
# ([0.08289508330516326, 0.08603981213691907, 0.09124232561160357], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06883184970953525)
# start training...
# finished training.
# ([0.07925139329372308, 0.08215872446695964, 0.07241650728078988], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06572697957356771)
# start training...
# finished training.
# ([0.06174264810024164, 0.06388518749139248, 0.06868667480273125], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.051108149993113985)
# start training...
# finished training.
# ([0.0606388617784549, 0.06474456420311561, 0.05434442789126665], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.05179565136249249)
# start training...
# finished training.
# ('k:', 8)
# H0:
H0[8,:] =  [0.11347348873431866, 0.08289508330516326, 0.07925139329372308, 0.06174264810024164, 0.0606388617784549]
# H1:
H1[8, :] = [0.11417914659549029, 0.08603981213691907, 0.08215872446695964, 0.06388518749139248, 0.06474456420311561]
# H2:
H2[8, :] = [0.2867718231983674, 0.09124232561160357, 0.07241650728078988, 0.06868667480273125, 0.05434442789126665]
# thresh_list
thresh_list[8, :] = [0.1, 0.09134331727639224, 0.06883184970953525, 0.06572697957356771, 0.051108149993113985, 0.05179565136249249]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
# start training...
# finished training.
# ([0.11346902602758163, 0.11546399043156551, 0.2765959959763747], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.09237119234525241)
# start training...
# finished training.
# ([0.0834356821500338, 0.08338991800944011, 0.0877385017199394], 2)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06674854572002704)
# start training...
# finished training.
# ([0.07832693442320213, 0.07991670950865135, 0.06958823326306465], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.06393336760692107)
# start training...
# finished training.
# ([0.06106909727438902, 0.06411300561366937, 0.06824514193412586], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.0512904044909355)
# start training...
# finished training.
# ([0.052492294556055315, 0.054122509100498296, 0.06418392597100674], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.5)
# ('new H_THRESH_CURIOUS threshold: ', 0.04329800728039864)
# start training...
# finished training.
# ('k:', 9)
# H0:
H0[9,:] =  [0.11346902602758163, 0.0834356821500338, 0.07832693442320213, 0.06106909727438902, 0.052492294556055315]
# H1:
H1[9,:] = [0.11546399043156551, 0.08338991800944011, 0.07991670950865135, 0.06411300561366937, 0.054122509100498296]
# H2:
H2[9, :] = [0.2765959959763747, 0.0877385017199394, 0.06958823326306465, 0.06824514193412586, 0.06418392597100674]
# thresh_list
thresh_list[9,:] = [0.1, 0.09237119234525241, 0.06674854572002704, 0.06393336760692107, 0.0512904044909355, 0.04329800728039864]
# selection_seq:
# [1, 1, 2, 1, 1, 1]
[rows,columns] = H0.shape
H0.mean(axis=0)
H0_yerr = H0.std(axis=0, ddof=1)/np.sqrt(rows)
H1.mean(axis=0)
H1_yerr = H1.std(axis=0, ddof=1)/np.sqrt(rows)
thresh_list_mean = thresh_list.mean(axis=0)
thresh_list_err = thresh_list.std(axis=0, ddof=1)/np.sqrt(rows)

# plt.figure()
# plt.bar([1,2,3,4,5],H0.mean(axis=0), align='center' ,width=0.5, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='Solution length')
# plt.errorbar([1,2,3,4,5],H0.mean(axis=0) , color = 'orange', capsize=7, yerr=yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)
# plt.title('Solution Length on Curriculum Series')

plt.figure()
# plt.plot([0,1,2,3,4], H0.mean(axis=0), label='H0_C+')
plt.errorbar([0,1,2,3,4],H0.mean(axis=0) , color = 'orange', capsize=7, yerr=H0_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H0_C+')
plt.errorbar([0,1,2,3,4],H1.mean(axis=0) , color = 'blue', capsize=7, yerr=H1_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H1_C+')
plt.errorbar([0,1,2,3,4],thresh_list_mean[:-1] , color = 'red', capsize=7, yerr=thresh_list_err[:-1], ecolor = 'black', elinewidth = 2, linewidth = 1,label='threshold_C+')
plt.legend()
# plt.plot([0,1,2,3,4], H1.mean(axis=0), label='H1_C+')
# plt.plot([0,1,2,3,4], H2_list_, label='H2_C+')

#
# plt.plot(range(5),thresh_list_mean[:-1], label='thresh_C+')
# plt.legend()
# plt.title('Curious')



# not curious

# Cm - Curious minus

H0_Cm = np.zeros([10,5])
H1_Cm = np.zeros([10,5])
H2_Cm = np.zeros([10,5])
thresh_list_Cm = np.zeros([10,6])

# /home/maor/miniconda2/envs/magenta/bin/python /home/maor/PycharmProjects/tangram_nih_mindset_3_0/evaluation2.py
# WARNING:tensorflow:From /home/maor/miniconda2/envs/magenta/lib/python2.7/site-packages/tensorflow/python/util/tf_should_use.py:107: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
# Instructions for updating:
# Use `tf.global_variables_initializer` instead.
# 2019-02-03 13:53:40.395623: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
# start training...
# finished training.
# ([0.12987992702386317, 0.1316548983256022, 0.40491810823098207], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.1558559124286358)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09875928438626803, 0.10026484269362229, 0.10161656599778396], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11851114126352164)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.08230034510294597, 0.08358034109457946, 0.08796992668738732], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09876041412353516)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07222700730348244, 0.07396932137318146, 0.07741571084046975], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08667240876417893)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06544209137941018, 0.06718795727460812, 0.07228418496938852], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07853050965529221)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 0)
# H0:
H0_Cm[0,:] =  [0.12987992702386317, 0.09875928438626803, 0.08230034510294597, 0.07222700730348244, 0.06544209137941018]
# H1:
H1_Cm[0,:] = [0.1316548983256022, 0.10026484269362229, 0.08358034109457946, 0.07396932137318146, 0.06718795727460812]
# H2:
H2_Cm[0,:] = [0.40491810823098207, 0.10161656599778396, 0.08796992668738732, 0.07741571084046975, 0.07228418496938852]
# thresh_list
thresh_list_Cm[0,:] = [0.5, 0.1558559124286358, 0.11851114126352164, 0.09876041412353516, 0.08667240876417893, 0.07853050965529221]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
# start training...
# finished training.
# ([0.12987299454517853, 0.13164873612232697, 0.40515828743959087], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15584759345421423)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09876047036586663, 0.09913221995035808, 0.10317173982277894], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11851256443903996)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.08234661664718236, 0.08288803467383751, 0.08623511363298465], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09881593997661883)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07157079378763835, 0.07298611371945112, 0.07422620822221805], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08588495254516602)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06485632138374524, 0.0658725591806265, 0.06893178132864144], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07782758566049429)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 1)
# H0:
H0_Cm[1,:] = [0.12987299454517853, 0.09876047036586663, 0.08234661664718236, 0.07157079378763835, 0.06485632138374524]
# H1:
H1_Cm[1,:] = [0.13164873612232697, 0.09913221995035808, 0.08288803467383751, 0.07298611371945112, 0.0658725591806265]
# H2:
H2_Cm[1,:] = [0.40515828743959087, 0.10317173982277894, 0.08623511363298465, 0.07422620822221805, 0.06893178132864144]
# thresh_list
thresh_list_Cm[1,:] = [0.5, 0.15584759345421423, 0.11851256443903996, 0.09881593997661883, 0.08588495254516602, 0.07782758566049429]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
# start training...
# finished training.
# ([0.12987080598488832, 0.12946869776799128, 0.4000820991320488], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15536243732158952)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09312689610016651, 0.09502421892606296, 0.09670771085298978], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11175227532019981)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07807016372680664, 0.07911427815755208, 0.07976591892731495], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09368419647216797)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06840743774022812, 0.06933983778342223, 0.0722035566965739], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08208892528827373)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06131737048809345, 0.06261183665348934, 0.06550499109121469], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07358084458571214)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 2)
# H0:
H0_Cm[2,:] =  [0.12987080598488832, 0.09312689610016651, 0.07807016372680664, 0.06840743774022812, 0.06131737048809345]
# H1:
H1_Cm[2,:] = [0.12946869776799128, 0.09502421892606296, 0.07911427815755208, 0.06933983778342223, 0.06261183665348934]
# H2:
H2_Cm[2,:] = [0.4000820991320488, 0.09670771085298978, 0.07976591892731495, 0.0722035566965739, 0.06550499109121469]
# thresh_list
thresh_list_Cm[2,:] = [0.5, 0.15536243732158952, 0.11175227532019981, 0.09368419647216797, 0.08208892528827373, 0.07358084458571214]
# selection_seq:
# [0, 1, 0, 0, 0, 0]
# start training...
# finished training.
# ([0.129881712106558, 0.13306346306434044, 0.4190563299717047], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15585805452786958)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09984216323265663, 0.10151990254720052, 0.1055299196487818], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11981059587918795)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.0830539434384077, 0.08426590454884064, 0.08815865638928536], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09966473212608924)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07274778072650616, 0.07491540297483787, 0.07700468332339556], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.0872973368718074)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06582861680250901, 0.06811174979576698, 0.07165324993622608], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07899434016301081)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 3)
# H0:
H0_Cm[3,:] =  [0.129881712106558, 0.09984216323265663, 0.0830539434384077, 0.07274778072650616, 0.06582861680250901]
# H1:
H1_Cm[3,:] = [0.13306346306434044, 0.10151990254720052, 0.08426590454884064, 0.07491540297483787, 0.06811174979576698]
# H2:
H2_Cm[3,:] = [0.4190563299717047, 0.1055299196487818, 0.08815865638928536, 0.07700468332339556, 0.07165324993622608]
# thresh_list
thresh_list_Cm[3,:] = [0.5, 0.15585805452786958, 0.11981059587918795, 0.09966473212608924, 0.0872973368718074, 0.07899434016301081]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
# start training...
# finished training.
# ([0.12987647912441155, 0.1298984136336889, 0.4036997037056165], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15585177494929386)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09741142468574719, 0.09885931626344338, 0.10087002240694486], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11689370962289662)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.08137710277850811, 0.08252859115600586, 0.08453731659131172], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09765252333420973)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07144226172031501, 0.071486968260545, 0.07341624528933795], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.085730714064378)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06480633906829052, 0.0647571514814328, 0.06778352688520382], 1)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07770858177771935)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 4)
# H0:
H0_Cm[4,:] =  [0.12987647912441155, 0.09741142468574719, 0.08137710277850811, 0.07144226172031501, 0.06480633906829052]
# H1:
H1_Cm[4,:] = [0.1298984136336889, 0.09885931626344338, 0.08252859115600586, 0.071486968260545, 0.0647571514814328]
# H2:
H2_Cm[4,:] = [0.4036997037056165, 0.10087002240694486, 0.08453731659131172, 0.07341624528933795, 0.06778352688520382]
# thresh_list
thresh_list_Cm[4,:] = [0.5, 0.15585177494929386, 0.11689370962289662, 0.09765252333420973, 0.085730714064378, 0.07770858177771935]
# selection_seq:
# [0, 0, 0, 0, 0, 1]
# start training...
# finished training.
# ([0.1298805628067408, 0.13103466767531174, 0.3998455145420172], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15585667536808895)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09828409781822792, 0.09890904182042831, 0.09922617521041478], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.1179409173818735)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.0820311155074682, 0.08251593663142277, 0.08413627820137219], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09843733860896182)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07131142494006035, 0.07187650142571865, 0.07438066066839756], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08557370992807241)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06457674808991261, 0.06490382781395546, 0.06992923907744579], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07749209770789513)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 5)
# H0:
H0_Cm[5,:] =  [0.1298805628067408, 0.09828409781822792, 0.0820311155074682, 0.07131142494006035, 0.06457674808991261]
# H1:
H1_Cm[5,:] = [0.13103466767531174, 0.09890904182042831, 0.08251593663142277, 0.07187650142571865, 0.06490382781395546]
# H2:
H2_Cm[5,:] = [0.3998455145420172, 0.09922617521041478, 0.08413627820137219, 0.07438066066839756, 0.06992923907744579]
# thresh_list
thresh_list_Cm[5,:] = [0.5, 0.15585667536808895, 0.1179409173818735, 0.09843733860896182, 0.08557370992807241, 0.07749209770789513]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
# start training...
# finished training.
# ([0.12987610010000375, 0.13100778139554536, 0.39959491827549076], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.1558513201200045)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09826136858035357, 0.0991516174414219, 0.09983241252410106], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11791364229642429)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.08199499814938276, 0.08286010302030124, 0.08417427845490284], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.0983939977792593)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07163950113149789, 0.07185111290369278, 0.0733175644507775], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08596740135779747)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06492290741358048, 0.0654746691385905, 0.06827776248638447], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07790748889629658)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 6)
# H0:
H0_Cm[6,:] =  [0.12987610010000375, 0.09826136858035357, 0.08199499814938276, 0.07163950113149789, 0.06492290741358048]
# H1:
H1_Cm[6,:] = [0.13100778139554536, 0.0991516174414219, 0.08286010302030124, 0.07185111290369278, 0.0654746691385905]
# H2:
H2_Cm[6,:] = [0.39959491827549076, 0.09983241252410106, 0.08417427845490284, 0.0733175644507775, 0.06827776248638447]
# thresh_list
thresh_list_Cm[6,:] = [0.5, 0.1558513201200045, 0.11791364229642429, 0.0983939977792593, 0.08596740135779747, 0.07790748889629658]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
# start training...
# finished training.
# ([0.1298830081255008, 0.13013935089111328, 0.4143926180326022], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15585960975060095)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09759936577234513, 0.09847062673324193, 0.10444361124283229], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11711923892681414)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.08146132566989997, 0.08199435013991135, 0.08819250571422088], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09775359080387996)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07096157318506485, 0.07205813358991574, 0.07679225848271297], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08515388782207782)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06447727863605206, 0.06582843340360202, 0.07155821873591496], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07737273436326246)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 7)
# H0:
H0_Cm[7,:] = [0.1298830081255008, 0.09759936577234513, 0.08146132566989997, 0.07096157318506485, 0.06447727863605206]
# H1:
H1_Cm[7,:] = [0.13013935089111328, 0.09847062673324193, 0.08199435013991135, 0.07205813358991574, 0.06582843340360202]
# H2:
H2_Cm[7,:] = [0.4143926180326022, 0.10444361124283229, 0.08819250571422088, 0.07679225848271297, 0.07155821873591496]
# thresh_list
thresh_list_Cm[7,:] = [0.5, 0.15585960975060095, 0.11711923892681414, 0.09775359080387996, 0.08515388782207782, 0.07737273436326246]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
# start training...
# finished training.
# ([0.12988168765337038, 0.13101578981448442, 0.41544410509940904], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15585802518404446)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09827118042187813, 0.09910787680210212, 0.10458740821251503], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11792541650625375)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.08201895004663712, 0.08261530215923603, 0.08811780733939929], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09842274005596453)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07139778137207031, 0.0738836312905336, 0.07816592240944886], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08567733764648437)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06462599069644244, 0.06715795932671963, 0.07278999915489784], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07755118883573092)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 8)
# H0:
H0_Cm[8,:] = [0.12988168765337038, 0.09827118042187813, 0.08201895004663712, 0.07139778137207031, 0.06462599069644244]
# H1:
H1_Cm[8,:] = [0.13101578981448442, 0.09910787680210212, 0.08261530215923603, 0.0738836312905336, 0.06715795932671963]
# H2:
H2_Cm[8,:] = [0.41544410509940904, 0.10458740821251503, 0.08811780733939929, 0.07816592240944886, 0.07278999915489784]
# thresh_list
thresh_list_Cm[8,:] = [0.5, 0.15585802518404446, 0.11792541650625375, 0.09842274005596453, 0.08567733764648437, 0.07755118883573092]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
# start training...
# finished training.
# ([0.12987751838488457, 0.130550408974672, 0.3994696201422276], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.15585302206186147)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.09791753230950771, 0.0993496454679049, 0.09958481788635254], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.11750103877140924)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.08163771873865372, 0.08324361458802834, 0.08521917538765149], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.09796526248638446)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.07198673639542018, 0.07210634916256635, 0.07577882668910882], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.08638408367450422)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ([0.06501274231152657, 0.06508607130784255, 0.07098911970089643], 0)
# ('new H_THRESH_NOT_CURIOUS threshold: ', 0.07801529077383187)
# ('new H_THRESH_CURIOUS threshold: ', 0.1)
# start training...
# finished training.
# ('k:', 9)
# H0:
H0_Cm[9,:] = [0.12987751838488457, 0.09791753230950771, 0.08163771873865372, 0.07198673639542018, 0.06501274231152657]
# H1:
H1_Cm[9,:] = [0.130550408974672, 0.0993496454679049, 0.08324361458802834, 0.07210634916256635, 0.06508607130784255]
# H2:
H2_Cm[9,:] = [0.3994696201422276, 0.09958481788635254, 0.08521917538765149, 0.07577882668910882, 0.07098911970089643]
# thresh_list
thresh_list_Cm[9,:] = [0.5, 0.15585302206186147, 0.11750103877140924, 0.09796526248638446, 0.08638408367450422, 0.07801529077383187]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
#
# Process finished with exit code 0

[rows,columns] = H0_Cm.shape
H0_Cm.mean(axis=0)
H0_Cm_yerr = H0_Cm.std(axis=0, ddof=1)/np.sqrt(rows)
H1_Cm.mean(axis=0)
H1_Cm_yerr = H1_Cm.std(axis=0, ddof=1)/np.sqrt(rows)
thresh_list_Cm_mean = thresh_list_Cm.mean(axis=0)
thresh_list_Cm_err = thresh_list_Cm.std(axis=0, ddof=1)/np.sqrt(rows)

# plt.figure()
# plt.bar([1,2,3,4,5],H0.mean(axis=0), align='center' ,width=0.5, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='Solution length')
# plt.errorbar([1,2,3,4,5],H0.mean(axis=0) , color = 'orange', capsize=7, yerr=yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)
# plt.title('Solution Length on Curriculum Series')

plt.figure()
# plt.plot([0,1,2,3,4], H0.mean(axis=0), label='H0_C+')
plt.errorbar([0,1,2,3,4],H0_Cm.mean(axis=0) , color = 'orange', capsize=7, yerr=H0_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H0_C-')
plt.errorbar([0,1,2,3,4],H1_Cm.mean(axis=0) , color = 'blue', capsize=7, yerr=H1_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H1_C-')
plt.errorbar([0,1,2,3,4],thresh_list_Cm_mean[:-1] , color = 'red', capsize=7, yerr=thresh_list_Cm_err[:-1], ecolor = 'black', elinewidth = 2, linewidth = 1,label='threshold_C-')
plt.legend()

######################################################################3

# long run

import numpy as np
import matplotlib.pyplot as plt

H0 = np.zeros([10,5])
H1 = np.zeros([10,5])
H2 = np.zeros([10,5])
thresh_list = np.zeros([10,6])

# H0:
H0[0,:] = [0.11347271845890926, 0.08348496754964192, 0.07981277123475686, 0.06151482386466784, 0.05914945480151054]
# H1:
H1[0,:] = [0.11518041904156025, 0.08556005893609463, 0.08119863118880834, 0.06284719247084397, 0.06343544446505033]
# H2:
H2[0,:] = [0.27849415021064955, 0.08861489173693535, 0.07129629453023274, 0.06813911291269156, 0.053946929100232244]
# thresh_list
thresh_list[0,:] = [0.1, 0.0921443352332482, 0.0684480471488757, 0.06495890495104667, 0.05027775397667518, 0.050748355572040264]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0[1,:] =  [0.11346905048076923, 0.08278946998791817, 0.07842263197287536, 0.06131742550776555, 0.05882688669057993]
# H1:
H1[1,:] =  [0.11395586453951322, 0.08430432050656049, 0.08022682483379658, 0.06171872676947178, 0.06166776021321615]
# H2:
H2[1,:] =  [0.2791171440711388, 0.09026265755677834, 0.06952023506164551, 0.06547288405589569, 0.05174011450547438]
# thresh_list
thresh_list[1,:] = [0.1, 0.09116469163161058, 0.06744345640524839, 0.06418145986703726, 0.049374981415577425, 0.04933420817057292]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0[2,:] =  [0.1134634507008088, 0.08362185649382763, 0.07877421379089355, 0.061670144399007164, 0.05937407566950871]
# H1:
H1[2, :] = [0.11554634876740284, 0.0846700912866837, 0.08066952534210987, 0.06276901563008626, 0.06276965141296387]
# H2:
H2[2, :] = [0.2789236704508464, 0.088746890043601, 0.06920414093213204, 0.06521005508227226, 0.05210334826738407]
# thresh_list
thresh_list[2,:] = [0.1, 0.09243707901392229, 0.06773607302934696, 0.06453562027368791, 0.05021521250406901, 0.050215721130371094]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0[3,:] =  [0.11347570174779648, 0.0834201666025015, 0.07999642690022786, 0.06200741498898237, 0.06141057992592836]
# H1:
H1[3,:] = [0.11525513575627254, 0.08606820228772286, 0.08215477527716221, 0.06441116944337502, 0.06469203264285357]
# H2:
H2[3, :] = [0.28731835194123095, 0.09056526575333033, 0.0713639381604317, 0.06852195201775967, 0.053504301951481745]
# thresh_list
thresh_list[3, :] = [0.1, 0.09220410860501804, 0.06885456183017828, 0.06572382022172978, 0.05152893555470002, 0.05175362611428286]
# selection_seq:
# [1, 1, 2, 1, 2, 1]
H0[4,:] =  [0.11346798676710862, 0.08328893857124524, 0.07770167864285983, 0.06077707119477101, 0.05785866272755158]
# H1:
H1[4, :] = [0.11495647674951798, 0.08324316220405774, 0.08025005536201672, 0.06049899565867889, 0.06050458321204552]
# H2:
H2[4, :] = [0.27932079021747297, 0.08879892031351726, 0.06989843417436649, 0.06620358198116987, 0.0514520620688414]
# thresh_list
thresh_list[4,:] = [0.1, 0.09196518139961439, 0.0666311508569962, 0.06420004428961339, 0.04862165695581681, 0.04840366656963642]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0[5,:] =  [0.11347249838022086, 0.08283067360902444, 0.07844160764645307, 0.06059177105243389, 0.05843828274653508]
# H1:
H1[5,:] = [0.11399271549322666, 0.0833291518382537, 0.07971666409419133, 0.06151855908907377, 0.0630810138506767]
# H2:
H2[5, :] = [0.27597490946451825, 0.08762673842601287, 0.06841120964441544, 0.06490301474546775, 0.052424699832231574]
# thresh_list
thresh_list[5,:] = [0.1, 0.09119417239458133, 0.06666332147060297, 0.06377333127535306, 0.04921484727125902, 0.050464811080541364]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0[6,:] =  [0.11346985743596004, 0.0831148685553135, 0.07845414601839505, 0.06043700682811248, 0.058368866260235124]
# H1:
H1[6, :] = [0.11451553687071189, 0.08362645369309646, 0.0792690790616549, 0.060593476662269004, 0.0625155461140168]
# H2:
H2[6, :] = [0.2766713117941832, 0.08783264649220002, 0.0690454458579039, 0.06736921652769431, 0.051748997125870146]
# thresh_list
thresh_list[6,:] = [0.1, 0.09161242949656952, 0.06690116295447716, 0.06341526324932392, 0.048474781329815204, 0.050012436891213444]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0[7,:] =  [0.11347247392703326, 0.08301143768506172, 0.07809860278398563, 0.06117083476139949, 0.05843694393451397]
# H1:
H1[7,:] = [0.11432971709813827, 0.08443788381723258, 0.08099971062097794, 0.06137644938933544, 0.062082877525916465]
# H2:
H2[7, :] = [0.289613528129382, 0.09309357251876439, 0.07135255520160382, 0.06801094764318222, 0.05352485485565968]
# thresh_list
thresh_list[7, :] = [0.1, 0.09146377367851062, 0.06755030705378606, 0.06479976849678236, 0.04910115951146835, 0.049666302020733175]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0[8,:] =  [0.11347348873431866, 0.08289508330516326, 0.07925139329372308, 0.06174264810024164, 0.0606388617784549]
# H1:
H1[8, :] = [0.11417914659549029, 0.08603981213691907, 0.08215872446695964, 0.06388518749139248, 0.06474456420311561]
# H2:
H2[8, :] = [0.2867718231983674, 0.09124232561160357, 0.07241650728078988, 0.06868667480273125, 0.05434442789126665]
# thresh_list
thresh_list[8, :] = [0.1, 0.09134331727639224, 0.06883184970953525, 0.06572697957356771, 0.051108149993113985, 0.05179565136249249]
# selection_seq:
# [1, 1, 2, 1, 2, 1]

H0[9,:] =  [0.11346902602758163, 0.0834356821500338, 0.07832693442320213, 0.06106909727438902, 0.052492294556055315]
# H1:
H1[9,:] = [0.11546399043156551, 0.08338991800944011, 0.07991670950865135, 0.06411300561366937, 0.054122509100498296]
# H2:
H2[9, :] = [0.2765959959763747, 0.0877385017199394, 0.06958823326306465, 0.06824514193412586, 0.06418392597100674]
# thresh_list
thresh_list[9,:] = [0.1, 0.09237119234525241, 0.06674854572002704, 0.06393336760692107, 0.0512904044909355, 0.04329800728039864]
# selection_seq:
# [1, 1, 2, 1, 1, 1]
[rows,columns] = H0.shape
H0.mean(axis=0)
H0_yerr = H0.std(axis=0, ddof=1)/np.sqrt(rows)
H1.mean(axis=0)
H1_yerr = H1.std(axis=0, ddof=1)/np.sqrt(rows)
thresh_list_mean = thresh_list.mean(axis=0)
thresh_list_err = thresh_list.std(axis=0, ddof=1)/np.sqrt(rows)

plt.figure()
# plt.plot([0,1,2,3,4], H0.mean(axis=0), label='H0_C+')
plt.errorbar([0,1,2,3,4],H0.mean(axis=0) , color = 'orange', capsize=7, yerr=H0_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H0_C+')
plt.errorbar([0,1,2,3,4],H1.mean(axis=0) , color = 'blue', capsize=7, yerr=H1_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H1_C+')
plt.errorbar([0,1,2,3,4],thresh_list_mean[:-1] , color = 'red', capsize=7, yerr=thresh_list_err[:-1], ecolor = 'black', elinewidth = 2, linewidth = 1,label='threshold_C+')
plt.legend()
# plt.plot([0,1,2,3,4], H1.mean(axis=0), label='H1_C+')
# plt.plot([0,1,2,3,4], H2_list_, label='H2_C+')

#
# plt.plot(range(5),thresh_list_mean[:-1], label='thresh_C+')
# plt.legend()
# plt.title('Curious')



# not curious

# Cm - Curious minus

H0_Cm = np.zeros([10,5])
H1_Cm = np.zeros([10,5])
H2_Cm = np.zeros([10,5])
thresh_list_Cm = np.zeros([10,6])

H0_Cm[0,:] =  [0.12987992702386317, 0.09875928438626803, 0.08230034510294597, 0.07222700730348244, 0.06544209137941018]
# H1:
H1_Cm[0,:] = [0.1316548983256022, 0.10026484269362229, 0.08358034109457946, 0.07396932137318146, 0.06718795727460812]
# H2:
H2_Cm[0,:] = [0.40491810823098207, 0.10161656599778396, 0.08796992668738732, 0.07741571084046975, 0.07228418496938852]
# thresh_list
thresh_list_Cm[0,:] = [0.5, 0.1558559124286358, 0.11851114126352164, 0.09876041412353516, 0.08667240876417893, 0.07853050965529221]
# selection_seq:
# [0, 0, 0, 0, 0, 0]

H0_Cm[1,:] = [0.12987299454517853, 0.09876047036586663, 0.08234661664718236, 0.07157079378763835, 0.06485632138374524]
# H1:
H1_Cm[1,:] = [0.13164873612232697, 0.09913221995035808, 0.08288803467383751, 0.07298611371945112, 0.0658725591806265]
# H2:
H2_Cm[1,:] = [0.40515828743959087, 0.10317173982277894, 0.08623511363298465, 0.07422620822221805, 0.06893178132864144]
# thresh_list
thresh_list_Cm[1,:] = [0.5, 0.15584759345421423, 0.11851256443903996, 0.09881593997661883, 0.08588495254516602, 0.07782758566049429]
# selection_seq:
# [0, 0, 0, 0, 0, 0]

H0_Cm[2,:] =  [0.12987080598488832, 0.09312689610016651, 0.07807016372680664, 0.06840743774022812, 0.06131737048809345]
# H1:
H1_Cm[2,:] = [0.12946869776799128, 0.09502421892606296, 0.07911427815755208, 0.06933983778342223, 0.06261183665348934]
# H2:
H2_Cm[2,:] = [0.4000820991320488, 0.09670771085298978, 0.07976591892731495, 0.0722035566965739, 0.06550499109121469]
# thresh_list
thresh_list_Cm[2,:] = [0.5, 0.15536243732158952, 0.11175227532019981, 0.09368419647216797, 0.08208892528827373, 0.07358084458571214]
# selection_seq:
# [0, 1, 0, 0, 0, 0]

H0_Cm[3,:] =  [0.129881712106558, 0.09984216323265663, 0.0830539434384077, 0.07274778072650616, 0.06582861680250901]
# H1:
H1_Cm[3,:] = [0.13306346306434044, 0.10151990254720052, 0.08426590454884064, 0.07491540297483787, 0.06811174979576698]
# H2:
H2_Cm[3,:] = [0.4190563299717047, 0.1055299196487818, 0.08815865638928536, 0.07700468332339556, 0.07165324993622608]
# thresh_list
thresh_list_Cm[3,:] = [0.5, 0.15585805452786958, 0.11981059587918795, 0.09966473212608924, 0.0872973368718074, 0.07899434016301081]
# selection_seq:
# [0, 0, 0, 0, 0, 0]

H0_Cm[4,:] =  [0.12987647912441155, 0.09741142468574719, 0.08137710277850811, 0.07144226172031501, 0.06480633906829052]
# H1:
H1_Cm[4,:] = [0.1298984136336889, 0.09885931626344338, 0.08252859115600586, 0.071486968260545, 0.0647571514814328]
# H2:
H2_Cm[4,:] = [0.4036997037056165, 0.10087002240694486, 0.08453731659131172, 0.07341624528933795, 0.06778352688520382]
# thresh_list
thresh_list_Cm[4,:] = [0.5, 0.15585177494929386, 0.11689370962289662, 0.09765252333420973, 0.085730714064378, 0.07770858177771935]
# selection_seq:
# [0, 0, 0, 0, 0, 1]

H0_Cm[5,:] =  [0.1298805628067408, 0.09828409781822792, 0.0820311155074682, 0.07131142494006035, 0.06457674808991261]
# H1:
H1_Cm[5,:] = [0.13103466767531174, 0.09890904182042831, 0.08251593663142277, 0.07187650142571865, 0.06490382781395546]
# H2:
H2_Cm[5,:] = [0.3998455145420172, 0.09922617521041478, 0.08413627820137219, 0.07438066066839756, 0.06992923907744579]
# thresh_list
thresh_list_Cm[5,:] = [0.5, 0.15585667536808895, 0.1179409173818735, 0.09843733860896182, 0.08557370992807241, 0.07749209770789513]
# selection_seq:
# [0, 0, 0, 0, 0, 0]

H0_Cm[6,:] =  [0.12987610010000375, 0.09826136858035357, 0.08199499814938276, 0.07163950113149789, 0.06492290741358048]
# H1:
H1_Cm[6,:] = [0.13100778139554536, 0.0991516174414219, 0.08286010302030124, 0.07185111290369278, 0.0654746691385905]
# H2:
H2_Cm[6,:] = [0.39959491827549076, 0.09983241252410106, 0.08417427845490284, 0.0733175644507775, 0.06827776248638447]
# thresh_list
thresh_list_Cm[6,:] = [0.5, 0.1558513201200045, 0.11791364229642429, 0.0983939977792593, 0.08596740135779747, 0.07790748889629658]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
H0_Cm[7,:] = [0.1298830081255008, 0.09759936577234513, 0.08146132566989997, 0.07096157318506485, 0.06447727863605206]
# H1:
H1_Cm[7,:] = [0.13013935089111328, 0.09847062673324193, 0.08199435013991135, 0.07205813358991574, 0.06582843340360202]
# H2:
H2_Cm[7,:] = [0.4143926180326022, 0.10444361124283229, 0.08819250571422088, 0.07679225848271297, 0.07155821873591496]
# thresh_list
thresh_list_Cm[7,:] = [0.5, 0.15585960975060095, 0.11711923892681414, 0.09775359080387996, 0.08515388782207782, 0.07737273436326246]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
H0_Cm[8,:] = [0.12988168765337038, 0.09827118042187813, 0.08201895004663712, 0.07139778137207031, 0.06462599069644244]
# H1:
H1_Cm[8,:] = [0.13101578981448442, 0.09910787680210212, 0.08261530215923603, 0.0738836312905336, 0.06715795932671963]
# H2:
H2_Cm[8,:] = [0.41544410509940904, 0.10458740821251503, 0.08811780733939929, 0.07816592240944886, 0.07278999915489784]
# thresh_list
thresh_list_Cm[8,:] = [0.5, 0.15585802518404446, 0.11792541650625375, 0.09842274005596453, 0.08567733764648437, 0.07755118883573092]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
H0_Cm[9,:] = [0.12987751838488457, 0.09791753230950771, 0.08163771873865372, 0.07198673639542018, 0.06501274231152657]
# H1:
H1_Cm[9,:] = [0.130550408974672, 0.0993496454679049, 0.08324361458802834, 0.07210634916256635, 0.06508607130784255]
# H2:
H2_Cm[9,:] = [0.3994696201422276, 0.09958481788635254, 0.08521917538765149, 0.07577882668910882, 0.07098911970089643]
# thresh_list
thresh_list_Cm[9,:] = [0.5, 0.15585302206186147, 0.11750103877140924, 0.09796526248638446, 0.08638408367450422, 0.07801529077383187]
# selection_seq:
# [0, 0, 0, 0, 0, 0]
[rows,columns] = H0_Cm.shape
H0_Cm.mean(axis=0)
H0_Cm_yerr = H0_Cm.std(axis=0, ddof=1)/np.sqrt(rows)
H1_Cm.mean(axis=0)
H1_Cm_yerr = H1_Cm.std(axis=0, ddof=1)/np.sqrt(rows)
thresh_list_Cm_mean = thresh_list_Cm.mean(axis=0)
thresh_list_Cm_err = thresh_list_Cm.std(axis=0, ddof=1)/np.sqrt(rows)

# plt.figure()
# plt.bar([1,2,3,4,5],H0.mean(axis=0), align='center' ,width=0.5, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 2, label='Solution length')
# plt.errorbar([1,2,3,4,5],H0.mean(axis=0) , color = 'orange', capsize=7, yerr=yerr, ecolor = 'black', elinewidth = 2, linewidth = 0)
# plt.title('Solution Length on Curriculum Series')

####################################################
# each graph seperatly

plt.figure()
# plt.plot([0,1,2,3,4], H0.mean(axis=0), label='H0_C+')
# plt.bar([0,1,2,3,4],H0_Cm.mean(axis=0) , align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 1, label='H0_C-')
# plt.bar([0.3,1.3,2.3,3.3,4.3],H1_Cm.mean(axis=0) , align='center' ,width=0.3, color = 'blue', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 1, label='H1_C-')
plt.errorbar([0,1,2,3,4],H0_Cm.mean(axis=0),  capsize=7, color='green', yerr=H0_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='Left puzzle C-')
plt.errorbar([0,1,2,3,4], H1_Cm.mean(axis=0) ,capsize=7, color='blue',  yerr=H1_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='Middle puzzle C-')

# plt.errorbar([0,1,2,3,4],H0_Cm.mean(axis=0) ,  capsize=7, yerr=H0_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H0_C-')
# plt.errorbar([0.5,1.5,2.5,3.5,4.5],H1_Cm.mean(axis=0) , capsize=7, yerr=H1_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H1_C-')
plt.errorbar([0,1,2,3,4],thresh_list_Cm_mean[:-1] , color = 'red', capsize=7, yerr=thresh_list_Cm_err[:-1], ecolor = 'black', elinewidth = 2, linewidth = 1,label='Threshold C-')
plt.legend()
plt.xticks([0,1,2,3,4],['1', '2', '3', '4', '5'])
plt.xlabel('Round Number')
plt.ylabel('Learnability Estimator')

plt.figure()
# plt.plot([0,1,2,3,4], H0.mean(axis=0), label='H0_C+')
# plt.bar([0,1,2,3,4],H0_Cm.mean(axis=0) , align='center' ,width=0.3, color = 'orange', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 1, label='H0_C-')
# plt.bar([0.3,1.3,2.3,3.3,4.3],H1_Cm.mean(axis=0) , align='center' ,width=0.3, color = 'blue', edgecolor = 'black', capsize=7, ecolor = 'black', linewidth = 1, label='H1_C-')
plt.errorbar([0,1,2,3,4],H0.mean(axis=0),  capsize=7, color='green', yerr=H0_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='Left puzzle C+')
plt.errorbar([0,1,2,3,4], H1.mean(axis=0) ,capsize=7, color='blue',  yerr=H1_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='Middle puzzle C+')

# plt.errorbar([0,1,2,3,4],H0_Cm.mean(axis=0) ,  capsize=7, yerr=H0_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H0_C-')
# plt.errorbar([0.5,1.5,2.5,3.5,4.5],H1_Cm.mean(axis=0) , capsize=7, yerr=H1_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H1_C-')
plt.errorbar([0,1,2,3,4],thresh_list_mean[:-1] , color = 'red', capsize=7, yerr=thresh_list_err[:-1], ecolor = 'black', elinewidth = 2, linewidth = 1,label='Threshold C+')
plt.legend()
plt.xticks([0,1,2,3,4],['1', '2', '3', '4', '5'])
plt.xlabel('Round Number')
plt.ylabel('Learnability Estimator')

######################################################################3
# both graphs together


[rows,columns] = H0.shape
H0.mean(axis=0)
H0_yerr = H0.std(axis=0, ddof=1)/np.sqrt(rows)
H1.mean(axis=0)
H1_yerr = H1.std(axis=0, ddof=1)/np.sqrt(rows)
thresh_list_mean = thresh_list.mean(axis=0)
thresh_list_err = thresh_list.std(axis=0, ddof=1)/np.sqrt(rows)

plt.figure()
# plt.plot([0,1,2,3,4], H0.mean(axis=0), label='H0_C+')
plt.errorbar([0,1,2,3,4],H0.mean(axis=0) , color = 'blue', fmt='-', capsize=7, yerr=H0_yerr, ecolor = 'black', elinewidth = 2, linewidth = 2,label='Left puzzle C+')
plt.errorbar([0,1,2,3,4],H1.mean(axis=0) , color = 'blue', fmt='--', capsize=7, yerr=H1_yerr, ecolor = 'black', elinewidth = 2, linewidth = 2,label='Middle puzzle C+')
# plt.errorbar([0,1,2,3,4],thresh_list_mean[:-1] , color = 'red', capsize=7, yerr=thresh_list_err[:-1], ecolor = 'black', elinewidth = 2, linewidth = 1,label='threshold_C+')

plt.errorbar([0,1,2,3,4],H0_Cm.mean(axis=0),  capsize=7, color='red', fmt='-', yerr=H0_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 2,label='Left puzzle C-')
plt.errorbar([0,1,2,3,4], H1_Cm.mean(axis=0) ,capsize=7, color='red', fmt='--', yerr=H1_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 2,label='Middle puzzle C-')

# plt.errorbar([0,1,2,3,4],H0_Cm.mean(axis=0) ,  capsize=7, yerr=H0_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H0_C-')
# plt.errorbar([0.5,1.5,2.5,3.5,4.5],H1_Cm.mean(axis=0) , capsize=7, yerr=H1_Cm_yerr, ecolor = 'black', elinewidth = 2, linewidth = 1,label='H1_C-')
# plt.errorbar([0,1,2,3,4],thresh_list_Cm_mean[:-1] , color = 'red', capsize=7, yerr=thresh_list_Cm_err[:-1], ecolor = 'black', elinewidth = 2, linewidth = 1,label='threshold_C-')
plt.xticks([0,1,2,3,4],['1', '2', '3', '4', '5'])
plt.legend()
plt.ylabel('Learnability Estimator')
plt.xlabel('Round Number')

######################################33

`

#############################################

