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

class PartyScreenRoom(Screen):
    the_tablet = None

    def __init__(self, the_tablet):
        self.the_tablet = the_tablet
        super(Screen, self).__init__()

    def on_enter(self, *args):
        print("on_enter first_screen_room")
        self.the_tablet.change_state('party_screen')


    def init_party(self, the_app, tangrams_solved):
        # display the number of tangrams solved (by the Robot and Child).

        self.world = the_app.study_world
        robot_char = the_app.robot_character
        self.ids['party_screen_background'].ids['background_image'].source = './tablet_app/images/worlds/' + self.world + '/TangramGame_Open.png'
        self.ids['party_screen_background'].robot_character.source = './tablet_app/images/worlds/' + robot_char + '.png'
        pos = the_app.interaction.components['robot'].animation['robot-position']
        self.ids['party_screen_background'].robot_character.x = the_app.root.size[0] * pos[0]
        self.ids['party_screen_background'].robot_character.y = the_app.root.size[1] * pos[1]

        i = 1
        while i < the_app.tangrams_solved + 1:

            self.ids['party_screen_prices_widget'].ids["price" + str(i)].source = './tablet_app/images/worlds/' + self.world + '/Price_' + str(i) + '.png'
            self.ids['party_screen_prices_widget'].ids["price" + str(i)].opacity = 1
            print("visible", i)

            i += 1

        while i < 13:
            self.ids['party_screen_prices_widget'].ids["price" + str(i)].opacity = 0
            print("invisible", i)

            i += 1




class PartyScreenBackground(Widget):
    pass

class PartyScreenPricesWidget(Widget):
    pass

