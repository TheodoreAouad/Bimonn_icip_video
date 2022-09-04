import manim as man

import matplotlib.pyplot as plt


class ResultsAnimation(man.Scene):
    def construct(self):
        erodila = man.ImageMobject(plt.imread("res_erodila.png")*255)
        erodila.height = 4
        opeclos = man.ImageMobject(plt.imread("res_opeclos.png")*255)
        opeclos.height = 5

        rect_ero = man.Rectangle(color=man.RED, height=1.1, width=1.2).move_to(3.4*man.RIGHT + 1.4*man.DOWN)
        rect_ope = man.Rectangle(color=man.RED, height=3.7, width=1.1).move_to(3.8*man.LEFT+.7*man.DOWN)

        self.wait(3)
        self.play(man.FadeIn(erodila))
        self.wait(18)
        self.play(man.Create(rect_ero))
        self.wait(6)
        self.play(man.FadeOut(erodila, rect_ero))

        self.play(man.FadeIn(opeclos))
        self.wait(6)
        self.play(man.Create(rect_ope))

        self.wait(15)
