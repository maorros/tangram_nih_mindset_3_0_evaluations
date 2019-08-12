from interaction_control import *
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import Layout
from kivy.uix.image import Image
from kivy.lang import Builder
from kivy.base import runTouchApp
from kivy.clock import Clock
from kivy.app import App
from kivy.animation import Animation
from kivy.core.window import Window

from kivy.app import App
from kivy_communication import *
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.audio import SoundLoader

class FirstScreenRoom(Screen):
    the_tablet = None

    # def __init__(self, **kwargs):
    #     print("init first", kwargs)
    #     super(Screen, self).__init__(**kwargs)

    def __init__(self, the_tablet):
        self.the_tablet = the_tablet
        super(Screen, self).__init__()

    def init_first_screen_room(self,the_app):
        print('init_first_screen_room')
        world = the_app.study_world
        robot_char = the_app.robot_character
        print(world)
        self.ids['background_image'].source = './tablet_app/images/worlds/' + world + '/TangramGame_Open.png'
        self.ids['yes_button'].background_normal =  './tablet_app/images/worlds/' + world + '/PriceBtn.png'
        self.ids['yes_button'].background_down = './tablet_app/images/worlds/' + world + '/PriceBtn_on.png'
        self.robot_character.source = './tablet_app/images/worlds/' + robot_char +'.png'
        pos = the_app.interaction.components['robot'].animation['robot-position']
        self.robot_character.x = the_app.root.size[0] * pos[0]
        self.robot_character.y = the_app.root.size[1] * pos[1]

    def on_enter(self, *args):
        print("on_enter first_screen_room")
        self.the_tablet.change_state('first_screen')
        #self.load_sounds()
        #self.play_sound("TangramOpen_myFriend")

    def disable_widgets(self):
        pass

    def enable_widgets(self):
        pass



                #def load_sounds(self):
    #    self.sounds = {}
    #    self.sounds[0] = SoundLoader.load("sounds\TangramOpen_myFriend.m4a")
    #    self.sounds[1] = SoundLoader.load("sounds\TangramOpen_click.m4a")

    #def play_sound(self, soundName):
    #    if soundName == "TangramOpen_myFriend":
    #        sound = self.sounds.get(0)
    #        sound.bind(on_stop=self.finish_tangram_intro)
    #        # Clock.schedule_once(self.callback(), 0)
    #    elif soundName == "TangramOpen_click":
    #        sound = self.sounds.get(1)
    #    if sound is not None:
    #        sound.volume = 0.5
    #        sound.play()

    #def finish_intro(self, dt):
        #now present the yes button
        #self.ids['yes_button'].opacity = 1
        #print("finish_tangram_intro")
        #self.play_sound("TangramOpen_click")

    #def press_yes_button(self):
        #print("press_yes_button")
        #app.sm.enter_solve_tangram_room()
        #App.action('press_yes_button')


