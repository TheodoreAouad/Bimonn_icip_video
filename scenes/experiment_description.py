from pipes import Template
import manim as man
import skimage

from utils import TemplateMathTex, TemplateTex
from mobjects import ArrayImage
from bise.structuring_elements import *


class ExperimentDescriptionAnimation(man.Scene):
    def construct(self):
        texs = [
            TemplateTex(r"Structuring Elements: "),
            TemplateTex(r"$\oplus$ / $\ominus$"),
            TemplateTex(r"Opening $\circ =  (\cdot \ominus S) \oplus S$"),
            TemplateTex(r"Closing $\bullet = (\cdot \oplus S) \ominus S$"),
            TemplateTex(r"$\longrightarrow$"),
            TemplateTex(r"DiceLoss for Diskorect, MSE Loss for MNIST"),
        ]

        selem1 = disk(3).astype(int)
        selem2 = hstick(7)
        selem3 = dcross(7)

        selem_mob1 = ArrayImage(selem1, show_value=False, cmap='Blues', shape_target=1*man.UP + 1 * man.RIGHT)
        selem_mob2 = ArrayImage(selem2, show_value=False, cmap='Blues', shape_target=1*man.UP + 1 * man.RIGHT)
        selem_mob3 = ArrayImage(selem3, show_value=False, cmap='Blues', shape_target=1*man.UP + 1 * man.RIGHT)

        selem_mob1.next_to(texs[0], man.RIGHT)
        selem_mob2.next_to(selem_mob1, man.RIGHT)
        selem_mob3.next_to(selem_mob2, man.RIGHT)

        man.VGroup(texs[0], selem_mob1, selem_mob2, selem_mob3).move_to(man.ORIGIN + 3 * man.UP)

        bise_mob = man.Circle(radius=.5, color=man.GREEN, fill_opacity=.5)
        bise_mob_label = TemplateMathTex(r"\bise_{\omega, b, p}", font_size=man.DEFAULT_FONT_SIZE*.5).add_updater(lambda x: x.move_to(bise_mob))
        bise_mob1 = man.Circle(radius=.5, color=man.GREEN, fill_opacity=.5)
        bise_mob1_label = TemplateMathTex(r"\bise_{\omega_1, b_1, p_1}", font_size=man.DEFAULT_FONT_SIZE*.5).add_updater(lambda x: x.move_to(bise_mob1))
        bise_mob2 = man.Circle(radius=.5, color=man.GREEN, fill_opacity=.5)
        bise_mob2_label = TemplateMathTex(r"\bise_{\omega_2, b_2, p_2}", font_size=man.DEFAULT_FONT_SIZE*.5).add_updater(lambda x: x.move_to(bise_mob2))

        bise_mob.next_to(texs[1], 2*man.DOWN)
        man.VGroup(texs[1], bise_mob).move_to(man.ORIGIN + 3 * man.LEFT)

        texs[3].next_to(texs[2], man.DOWN)
        bise_mob1.next_to(texs[3], man.DOWN)
        texs[4].next_to(bise_mob1, man.RIGHT)
        bise_mob2.next_to(texs[4], man.RIGHT)

        man.VGroup(texs[4], bise_mob2, bise_mob1).next_to(texs[3], 2*man.DOWN)
        man.VGroup(texs[4], bise_mob2, bise_mob1, texs[2], texs[3]).move_to(man.ORIGIN + 3 * man.RIGHT)

        self.wait(2)
        self.play(man.FadeIn(texs[1]))
        self.wait(2)
        self.play(man.FadeIn(bise_mob, bise_mob_label))
        self.wait(4)
        self.play(man.FadeIn(texs[2], texs[3]))
        self.wait(2)
        self.play(man.FadeIn(texs[4], bise_mob2, bise_mob1, bise_mob1_label, bise_mob2_label))
        self.wait(5)
        self.play(man.FadeIn(texs[0], selem_mob1, selem_mob2, selem_mob3))
        self.wait(10)
        self.play(man.FadeIn(texs[-1].shift(3*man.DOWN)))

        self.wait(10)
