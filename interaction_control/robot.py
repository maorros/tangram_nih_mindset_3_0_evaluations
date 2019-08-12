from component import *
import time
from agent import *
import json
from random import choice
from tablet_app.tangram_game import *
import os

is_logged = False   #Solves the slow problem
try:
    from kivy_communication import *
except:
    print('no logging')
    is_logged = False


class RobotComponent(Component):
    whos_playing = None
    app = None
    expression = None
    agent = Agent()
    current_tangram = None
    robot_name = 'tega'
    animation = None
    gender = None
    study_world = "w"

    def load_text(self, session_filename='', general_filename='', participant_name=''):  #robot_text_revised3
        self.animation={}
        with open(session_filename) as data_file:
            self.animation.update(json.load(data_file))

        if general_filename != '':
            with open(general_filename) as general_f:
                self.animation.update(json.load(general_f))

        for item in self.animation:
            if item in ["robot-position", "prize-name"]:
                continue
            if isinstance(self.animation[item], dict):
                for condition in self.animation[item]: #all, c-g-, ...
                    for script in self.animation[item][condition]:
                        if "<prize-name>" in script[0]:
                            script[0] = script[0].replace("<prize-name>", self.animation['prize-name'])
                        if "<participant-name>" in script[0]:
                            script[0] = script[0].replace("<participant-name>", participant_name)
            elif isinstance(self.animation[item], list):
                for script in self.animation[item]:
                    if "<prize-name>" in script[0]:
                        script[0] = script[0].replace("<prize-name>", self.animation['prize-name'])
                    if "<participant-name>" in script[0]:
                        script[0] = script[0].replace("<participant-name>", participant_name)

    def run_function(self, action):
        print(self.name, 'run_function', action[0], action[1:])
        if action[0] == action[1:]:
            print('weird')
            return False
        try:
            if action[1] is not None:
                getattr(self, action[0])(action[1])
            else:
                getattr(self, action[0])()
            return True
        except Exception as e:
            print e
            if not isinstance(sys.exc_info()[1], AttributeError):
                print ("unexpected error:",sys.exc_info())
            try:
                self.express(action)
            except Exception as e:
                print e
                print ("except except: could not except in run_function:",sys.exc_info())
        return False

    def express(self, action):
        print("rinat express", action, action[0])
        print("TangramGame.cog_tangram_selection=", TangramGame.cog_tangram_selection)
        print ('question=', self.agent.current_round)
        self.current_state = 'express'
        if len(action) > 1:
            self.current_param = action[1:]

        if self.animation is None:
            self.expression = action[0]
            if KC.client.connection:
                print('if KC.client.connection:')
                data = [action[0], self.expression]
                data = {self.robot_name: data}
                KC.client.send_message(str(json.dumps(data)))

            if self.app:
                print("if self.app:")
                self.app.robot_express(action[0], self.expression)

        elif 'idle' not in action[0]:
            # select the animation
            print ("robot express idle not in action[0]",action[0])
            # print("self.animation=", self.animation)
            # print("self.animation[action[0]]=", self.animation[action[0]])
            the_options = copy.deepcopy(self.animation[action[0]])
            the_expressions = []
            what = action[0]  # Rinat added
            if isinstance(the_options, list):
                if what == "ask_question_party": #Rinat added
                    if self.agent.condition=='c+g-' or self.agent.condition == 'c+g+': #Rinat added
                        the_expressions = self.add_expression(the_expressions, choice(the_options)) #Rinat added
                else: #Rinat added
                    the_expressions = self.add_expression(the_expressions, choice(the_options))
            elif isinstance(the_options, dict):
                # Rinat added
                if 'all' in the_options:
                    the_expressions = self.add_expression(the_expressions, choice(the_options['all']))

                if what == "ask_question_robot_play":
                    if self.agent.condition == 'c+g-' or self.agent.condition == 'c+g+':
                        the_expressions = self.add_expression(the_expressions, the_options['question' + str(self.agent.current_round)])
                elif what == "my_turn":
                    do_selection_speech = choice([0,1,2])
                    if do_selection_speech == 0 and TangramGame.cog_tangram_selection in [0,1,2]:
                            the_expressions = self.add_expression(the_expressions,the_options[self.agent.condition][TangramGame.cog_tangram_selection])
                    else:
                        the_expressions = self.add_expression(the_expressions,
                                                              choice(the_options[self.agent.condition][3:]))

                # Rinat end
                else:
                    if self.agent.condition in the_options:
                        the_expressions = self.add_expression(the_expressions, choice(the_options[self.agent.condition]))


            self.expression = self.process_expression (the_expressions)

            self.expression = the_expressions

            if KC.client.connection:
                data = [action[0], self.expression]
                data = {self.robot_name: data}
                KC.client.send_message(str(json.dumps(data)))

            if self.app:
                self.app.robot_express(action[0], self.expression)


    def process_expression (self, expression):
        # check and change accordingly sound file names to match Gender and World if needed
        print("robot process_expression ",expression, self.gender, self.study_world)

        sound_list = os.listdir("./tablet_app/sounds/wav_tangram")

        for i in range(0,len(expression)):
            if (expression[i].lower()==expression[i]):
                name = expression[i]
                try_sound = name
                try_sound_gender = name + '_' + self.gender
                try_sound_world = name + "_" + self.study_world
                try_sound_world_gender = name + "_" + self.study_world + '_' + self.gender
                if (try_sound + '.wav' in sound_list):
                    expression[i] = try_sound
                elif (try_sound_gender + '.wav' in sound_list):
                    expression[i] = try_sound_gender
                elif (try_sound_world + '.wav' in sound_list):
                    expression[i] = try_sound_world
                elif (try_sound_world_gender +'.wav' in sound_list):
                    expression[i] = try_sound_world_gender
        return expression


    def add_expression(self, base, add):
        if len(base) == 0:
            base = add
        else:
            base[0] += add[0]
            for b in add[1:]:
                base.append(b)
        return base

    def after_called(self):
        if self.current_param:
            if isinstance(self.current_param, list):
                if 'done' in self.current_param:
                    self.current_state = 'idle'

    def set_playing(self, action):
        self.current_param = action[1:]
        self.whos_playing = action[0]
        print(self.whos_playing, self.current_param)

    def select_treasure(self):
        print("robot select_treasure")
        if not isinstance(self.current_param, int):  # tangram already selected
            the_selection = self.agent.set_selection()
            print("finished set_selection")
            print(self.name, 'select_treasure', the_selection, self.current_param)
            self.current_tangram = self.current_param[0][the_selection]
            self.current_state = 'select_treasure'
            self.current_param = the_selection
        self.agent.finish_moves()  # indication to the agent that the last game is finished. agent clears the last solution

    def select_move(self, x):
        print(self.name, 'select_move', x)
        self.current_state = 'select_move'
        self.current_param = self.current_tangram[0]
        move = self.agent.play_move(x)
        self.current_param = move

    # def set_selection(self, action):
    #     print('robot set selection', action)
    #     # self.current_param = action[1:]
    #     # set the possible treasures to select from
    #     # select 1 for demo, 2 for robot
    #     # waiting for Maor's algorithm
    #     if self.whos_playing == 'demo':se
    #         self.current_param = 1
    #         self.current_state = 'idle'
    #     if self.whos_playing == 'robot':
    #         self.current_state = 'select_treasure'
    #         self.current_param = 2

    def win(self):  # called only in tutorial
        print(self.name, self.whos_playing, 'wins!')
        if self.whos_playing == 'child':
            self.run_function(['child_win', None])
        else:
            self.run_function(['robot_win', None])

    def after_child_win(self):
        print(self.name, self.whos_playing, 'after_child_win')
        self.agent.record_child_result('S')
        self.current_state = 'after_child_win'

    def after_child_lose(self):
        print(self.name, self.whos_playing, 'after_child_lose')
        self.agent.record_child_result('F')
        self.current_state = 'after_child_lose'

    def play_game(self, action):
        print(self.whos_playing, 'playing the game', action)
        self.current_state = 'play_game'
        self.agent = Agent()
        seq = self.agent.solve_task(action[1][0]) #  solve the selected task and return a seq of moves in json string
        #self.current_param = action[1]
        self.current_param = seq

    # def comment_selection(self, action):
    #     if self.whos_playing == "child":
    #         print(self.name, 'commenting on selection ', action)

    # def comment_move(self, action):
    #     if self.whos_playing == "child":
    #         print(self.name, 'commenting on move ', action)

    # def comment_turn(self, action):
    #     if self.whos_playing == "child":
    #         print(self.name, 'commenting on turn ', action)

    def finished_expression(self, action):
        # self.current_param = None
        self.current_state = action
        print('finished expression:', self.name, action, self.current_state)

    def data_received(self, data):
        # if data signals end of speech
        # call: self.finished_expression(action)
        if "pid" in data and "condition" in data: #message for tangram app
            return

        idx = data.find("}{")
        while idx >= 0:
            print(self.name, data[:(idx+1)])
            the_data = json.loads(data[:(idx+1)])
            self.finished_expression(the_data[self.robot_name][1])
            data = data[(idx+1):]
            idx = data.find("}{")

        the_data = json.loads(data)
        self.finished_expression(the_data[self.robot_name][1])
        # if the_data[self.robot_name][1] == "robot_select":
        #     self.app.enable_tablet()


    def child_selection(self, x):
        print(self.name, 'child selected', x)
        self.agent.record_child_selection(x)
