from tangrams import *
import json
import numpy as np
# import matplotlib.pyplot as plt


class SelectionGeneratorCuriosity:

    def __init__(self):
        self.N_paths = 3  # This value is overridden in load_paths(),
        self.max_level = 6
        self.challenge_index = 9 # column number of the challenge level (zero based)
        self.challenge_counter = 0 # counts the challenges
        self.is_last_challenge = False # indicates whether the last game was a challenge
        self.paths = []
        self.path_indexes = np.zeros([self.N_paths], dtype=np.int)
        self.current_level = 2
#        self.same_game_counter_child = 0
        self.child_selection_history = []
        self.robot_selection_history = []
        # self.increased_child = True
        # self.increased_robot = True
        # self.same_game_counter_child = 0
        # self.same_game_counter_robot = 0
        self.child_path_idx = 0
        self.robot_path_idx = 1
        self.other_path_idx = 2
        self.other_rot_path_idx = 3
        self.hard_path_idx = 4
        self.impossible_path_idx = 5

        for n in range(self.N_paths):
            self.paths.append([])

    def load_dif_levels(self, filename='dif1', directory='.', world = 'w1'):
        # read tangrams for each difficulty from tangram_levels.txt
        # file format is:
        #  dif 0
        # {"pieces": [["square", "90", "1 1"], ["small triangle2", "180", "0 1"]], "size": "5 5"}
        # {"pieces": [["square", "180", "0 0"], ["small triangle2", "180", "0 1"]], "size": "5 5"}
        # dif 1
        # {"pieces": [["square", "180", "0 0"], ["small triangle2", "180", "0 1"]], "size": "5 5"}
        # ...
        #
        path_i = -1
        self.paths = []
        # with open('./game_facilitator/tangram_levels_' + filename + '.txt','r') as fp:
        with open(directory +'/game_facilitator/tangram_paths_'+ world + '.txt', 'r') as fp:
            for line in fp:
                if 'path' in line:
                    path_i += 1
                    self.paths.append([])
                else:
                    self.paths[path_i].append(line.strip('\n'))
        self.N_paths = path_i + 1
        self.path_indexes = np.zeros([self.N_paths], dtype=np.int)
        self.seen_puzzles = np.zeros([self.N_paths, max([len(self.paths[i]) for i in range(self.N_paths)])])


    def get_current_selection(self, player):
        # return three json_strings
        temp_task = Task()
        all_pieces_task = Task()
        all_pieces_task.create_from_json('{"pieces": [["square", "0", "0 0"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"], ["medium triangle", "0", "3 1"], ["large triangle2", "180", "1 1"]], "size": "5 5"}')
        all_pieces_init_pos = all_pieces_task.transfer_json_to_json_initial_pos('{"pieces": [["square", "0", "0 0"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"], ["medium triangle", "0", "3 1"], ["large triangle2", "180", "1 1"]], "size": "5 5"}')
        all_pieces_init_pos = all_pieces_task.transfer_json_to_json_initial_pos(
            '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}')

        if player == 'Child':
            T1 = self.paths[self.child_path_idx][self.path_indexes[self.child_path_idx]]
            T1_init_pos = temp_task.transfer_json_to_json_initial_pos(T1)
            T2 = self.paths[self.child_path_idx][self.path_indexes[self.child_path_idx]+1]
            T2_init_pos = temp_task.transfer_json_to_json_initial_pos(T2)
            T3 = self.paths[self.other_path_idx][self.path_indexes[self.other_path_idx]]
            T3_init_pos = temp_task.transfer_json_to_json_initial_pos(T3)
            return [[T1, all_pieces_init_pos], [T2, all_pieces_init_pos], [T3, all_pieces_init_pos]]
        elif player == 'Robot':
            T1 = self.paths[self.robot_path_idx][self.path_indexes[self.robot_path_idx]]
            T1_init_pos = temp_task.transfer_json_to_json_initial_pos(T1)
            T2 = self.paths[self.robot_path_idx][self.path_indexes[self.robot_path_idx]+1]
            T2_init_pos = temp_task.transfer_json_to_json_initial_pos(T2)
            T3 = self.paths[self.other_rot_path_idx][self.path_indexes[self.other_rot_path_idx]]
            T3_init_pos = temp_task.transfer_json_to_json_initial_pos(T3)
            return [[T1, all_pieces_init_pos], [T2, all_pieces_init_pos], [T3, all_pieces_init_pos]]

    def update_game_result(self, player, user_selection, game_result):
        # update the selection generator according to user selection and game result
        # player is 'Robot' or 'Child'
        # user_selection is 0/1/2
        # game_result is 'S' or 'F'

        if not self.is_last_challenge: # if last round was not a challenge
            if player == 'Child':
                if user_selection == 2: # unknown puzzle
                    # self.seen_puzzles[self.child_path_idx, self.path_indexes[self.child_path_idx]] += 1
                    self.seen_puzzles[self.other_path_idx, self.path_indexes[self.other_path_idx]] += 1
                    self.path_indexes[self.child_path_idx] += 1
                    self.path_indexes[self.other_path_idx] += 1
                elif user_selection == 0: # current puzzle
                    self.seen_puzzles[self.child_path_idx, self.path_indexes[self.child_path_idx]] += 1
                    if self.seen_puzzles[self.child_path_idx, self.path_indexes[self.child_path_idx]] == 2:
                        self.path_indexes[self.child_path_idx] += 1
                    self.path_indexes[self.other_path_idx] += 1
                elif user_selection == 1: # next puzzle
                    self.seen_puzzles[self.child_path_idx, self.path_indexes[self.child_path_idx] + 1] += 1
                    self.path_indexes[self.child_path_idx] += 1
                    self.path_indexes[self.other_path_idx] += 1
            elif player == 'Robot':
                if user_selection == 2:  # unknown puzzle
                    # self.seen_puzzles[self.child_path_idx, self.path_indexes[self.child_path_idx]] += 1
                    self.seen_puzzles[self.other_rot_path_idx, self.path_indexes[self.other_rot_path_idx]] += 1
                    self.path_indexes[self.robot_path_idx] += 1
                    self.path_indexes[self.other_rot_path_idx] += 1
                elif user_selection == 0:  # current puzzle
                    self.seen_puzzles[self.robot_path_idx, self.path_indexes[self.robot_path_idx]] += 1
                    if self.seen_puzzles[self.robot_path_idx, self.path_indexes[self.robot_path_idx]] == 2:
                        self.path_indexes[self.robot_path_idx] += 1
                    self.path_indexes[self.other_rot_path_idx] += 1
                elif user_selection == 1:  # next puzzle
                    self.seen_puzzles[self.robot_path_idx, self.path_indexes[self.robot_path_idx] + 1] += 1
                    self.path_indexes[self.robot_path_idx] += 1
                    self.path_indexes[self.other_rot_path_idx] += 1
        else: # if last game was a challenge then don't progress the paths.
            self.is_last_challenge = False


    def get_challenge_selection(self):
        # return three json_strings of special challenge level.
        self.is_last_challenge = True
        if self.challenge_counter == 0: # first challenge is hard tangrams, short time
            self.challenge_counter += 1
            temp_task = Task()
            all_pieces_task = Task()
            all_pieces_task.create_from_json(
                '{"pieces": [["square", "0", "0 0"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"], ["medium triangle", "0", "3 1"], ["large triangle2", "180", "1 1"]], "size": "5 5"}')
            # all_pieces_init_pos = all_pieces_task.transfer_json_to_json_initial_pos(
            #     '{"pieces": [["square", "0", "0 0"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"], ["medium triangle", "0", "3 1"], ["large triangle2", "180", "1 1"]], "size": "5 5"}')
            all_pieces_init_pos = all_pieces_task.transfer_json_to_json_initial_pos(
                '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}')

            T1 = self.paths[self.hard_path_idx][0]
            T1_init_pos = temp_task.transfer_json_to_json_initial_pos(T1)
            T2 = self.paths[self.hard_path_idx][1]
            T2_init_pos = temp_task.transfer_json_to_json_initial_pos(T2)
            T3 = self.paths[self.hard_path_idx][2]
            T3_init_pos = temp_task.transfer_json_to_json_initial_pos(T3)
            return [[T1, all_pieces_init_pos], [T2, all_pieces_init_pos], [T3, all_pieces_init_pos]]
        else:       # second challenge is impossible tangrams, long time
            self.challenge_counter += 1
            temp_task = Task()
            all_pieces_task = Task()
            all_pieces_task.create_from_json(
                '{"pieces": [["square", "0", "0 0"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"], ["medium triangle", "0", "3 1"], ["large triangle2", "180", "1 1"]], "size": "5 5"}')
            # all_pieces_init_pos = all_pieces_task.transfer_json_to_json_initial_pos(
            #     '{"pieces": [["square", "0", "0 0"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"], ["medium triangle", "0", "3 1"], ["large triangle2", "180", "1 1"]], "size": "5 5"}')
            all_pieces_init_pos = all_pieces_task.transfer_json_to_json_initial_pos(
                '{"pieces": [["large triangle2", "180", "1 1"], ["large triangle1", "0", "1 1"], ["parrallelogram", "0", "2 0"],  ["medium triangle", "0", "3 1"], ["small triangle2", "0", "0 1"], ["small triangle1", "90", "1 0"], ["square", "0", "0 0"]], "size": "5 5"}')

            T1 = self.paths[self.impossible_path_idx][0]
            T1_init_pos = temp_task.transfer_json_to_json_initial_pos(T1)
            T2 = self.paths[self.impossible_path_idx][1]
            T2_init_pos = temp_task.transfer_json_to_json_initial_pos(T2)
            T3 = self.paths[self.impossible_path_idx][2]
            T3_init_pos = temp_task.transfer_json_to_json_initial_pos(T3)
            return [[T1, all_pieces_init_pos], [T2, all_pieces_init_pos], [T3, all_pieces_init_pos]]

    #
    # def display(self):
    #     plt.figure()
    #     task = Task()
    #     num_of_rows = max(len(l) for l in self.paths)  # find the maximal length of all sub lists in paths
    #     for n in range(self.N_paths):
    #         for k in range(len(self.paths[n])):
    #             plt.subplot(num_of_rows,self.N_paths, n+1+k*self.N_paths)
    #             task.create_from_json(self.paths[n][k])
    #             if k < self.path_indexes[n]:
    #                 plt.imshow(task.x*-1, interpolation='none')
    #                 plt.axis('off')
    #             else:
    #                 plt.imshow(task.x, interpolation='none')
    #                 plt.axis('off')
        # plt.subplot(num_of_rows,self.N_paths,
        #             self.current_level-1+1+(self.path_indexes[self.current_level - 1])*self.N_paths)
        # task.create_from_json(self.paths[self.current_level-1][self.path_indexes[self.current_level - 1]])
        # plt.imshow(np.sin(task.x), interpolation='none')
        # plt.axis('off')
        # plt.subplot(num_of_rows, self.N_paths,
        #             self.current_level  + 1 + (self.path_indexes[self.current_level]) * self.N_paths)
        # task.create_from_json(self.paths[self.current_level ][self.path_indexes[self.current_level ]])
        # plt.imshow(np.sin(task.x), interpolation='none')
        # plt.axis('off')
        # plt.subplot(num_of_rows, self.N_paths,
        #             self.current_level + 1 + 1 + (self.path_indexes[self.current_level + 1]) * self.N_paths)
        # task.create_from_json(self.paths[self.current_level + 1][self.path_indexes[self.current_level + 1]])
        # plt.imshow(np.sin(task.x), interpolation='none')
        # plt.axis('off')
                # T1 = self.paths[self.current_level - 1][self.path_indexes[self.current_level - 1]]
                # T2 = self.paths[self.current_level][self.path_indexes[self.current_level]]
                # T3 = self.paths[self.current_level + 1][self.dif_indexes[self.current_level + 1]]



# task_dic =  {'size': '5 5', 'pieces': [('square', '90', '1 1'), ('small triangle2', '180', '0 1')]}
# json_str = json.dumps(task_dic)
# paths[0].append(json_str)
#
# task_dic =  {'size': '5 5', 'pieces': [('square', '90', '1 1'), ('small triangle2', '180', '0 1')]}
# json_str = json.dumps(task_dic)
# paths[0].append(json_str)



#
# task = Task()
# task.create_from_json(json_str)
#
#
# task_dic =  {'size': '5 5', 'pieces': [('square', '0', '1 1'), ('small triangle2', '0', '0 1')]}
# json_str = json.dumps(task_dic)
# task = Task()
# task.create_from_json(json_str)
