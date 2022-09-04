import manim as man

from utils import TemplateMathTex
import matplotlib.pyplot as plt


class TestScene(man.Scene):
    def construct(self):
        logo = plt.imread("Logo_OPIS.png")*255
        # logo[logo[..., -1] == 0] = 255
        # logo[..., -1] = 1
        
        img = man.ImageMobject(logo)
        self.play(man.FadeIn(img))
        self.wait(3)