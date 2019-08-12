import os
from kivy.core.audio import SoundLoader
import time

#sound = SoundLoader.load('mytest.wav')
sound1 = SoundLoader.load('./tablet_app/sounds/wav_tangram/introduction_c-g-_0_w1_f.wav')


sounds_list = os.listdir("./tablet_app/sounds/wav_tangram")
sounds = {}
print(sounds_list)
for sound in sounds_list:
    sounds[sound] = SoundLoader.load("./tablet_app/sounds/wav_tangram/" + sound)
 #   sounds[sound].play()
#print(sounds_list[1])
#sound1 = SoundLoader.load("./tablet_app/sounds/wav_tangram/" + "selection_tutorial_c+g-_0.wav")
#sound2 = SoundLoader.load("./tablet_app/sounds/wav_tangram/" + sounds_list[2])
#sound1.play()

if sound1.length> 0:
    print("Sound found at %s" % sound1.source)
    print(sound1.length)
#    print("Sound is %.3f seconds" % sound.length)
    sound1.play()
    # for i in range(0,1000):
    #     time.sleep(1)

import pygame
pygame.init()
pygame.mixer.pre_init(44100, 16, 2, 4096) # Frequency, size, channels and buffersize
shield_on = pygame.mixer.Sound('/home/maor/test_audio2.wav')

