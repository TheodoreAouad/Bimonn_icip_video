import manim as man
import numpy as np

from mobjects import ArrayImage
from utils import TemplateMathTex
from example_array import example1


class NotationAnimation(man.Scene):
    def construct(self):
        texs = [
            TemplateMathTex(r"\mathbb{Z}^d").shift(2*man.UP),  # 0
            TemplateMathTex(r"~,~ d = 2").shift(2*man.UP),  # 1
            TemplateMathTex(r"~,~ d = 3").shift(2*man.UP),  # 2
            TemplateMathTex(r"I \in", r"\mathbb{R}", r"^{\mathbb{Z}^d}: (i, j) \in \mathbb{Z}^d \mapsto I(i, j)").shift(2*man.UP),  # 3
            TemplateMathTex(r"I \in", r"[0, 1]", r"^{\mathbb{Z}^d}: (i, j) \in \mathbb{Z}^d \mapsto I(i, j)").shift(2*man.UP),  # 4
            TemplateMathTex(r"X", r"\subset \mathbb{Z}^d").shift(2*man.UP),  # 5
            TemplateMathTex(r"\mathbbm{1}_{", "X", "}", r"\in \{0, 1\}^{\mathbb{Z}^d}").shift(2*man.UP),  # 6
            TemplateMathTex(r"S", r"\subset \mathbb{Z}^d").shift(2*man.UP),  # 7
            TemplateMathTex(r"\mathbbm{1}_{", "S", "}", r"\in \{0, 1\}^{\mathbb{Z}^d}").shift(2*man.UP),  # 8
            TemplateMathTex(r"I_1, I_2 \in \mathbb{R}^{\mathbb{Z}^d}", r"~,~ I_1 \circledast I_2").shift(2*man.UP),  # 9
        ]

        einstein_img = np.load("einstein.npy")
        X_shape = example1
        selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

        shape_mob = ArrayImage(X_shape, mask=X_shape, show_value=False)
        shape_im_mob = ArrayImage(X_shape, show_value=True)

        selem_mob = ArrayImage(selem, mask=selem, show_value=False, cmap='Blues')
        selem_im_mob = ArrayImage(selem, show_value=True, cmap='Blues')

        einstein_mob = man.ImageMobject(einstein_img,)
        einstein_mob.height=3

        self.wait(4)
        self.play(man.FadeIn(texs[0]))
        self.wait(1)
        self.play(man.FadeIn(texs[1].next_to(texs[0], man.RIGHT)))
        self.wait(4)
        self.play(man.TransformMatchingTex(texs[1], texs[2].move_to(texs[1])))
        self.play(man.FadeOut(texs[0], texs[2]))


        self.play(man.FadeIn(texs[3], einstein_mob.next_to(texs[3], man.DOWN)))
        self.wait(5)
        self.play(man.TransformMatchingTex(texs[3], texs[4]))
        self.wait(2)
        self.play(man.FadeOut(texs[4], einstein_mob))

        shape_im_mob.next_to(texs[5], man.DOWN)
        self.play(man.FadeIn(texs[5], shape_mob.move_to(shape_im_mob)))
        self.wait(2)
        self.play(man.TransformMatchingTex(texs[5], texs[6]), man.FadeOut(shape_mob), man.FadeIn(shape_im_mob))
        self.wait(4)
        self.play(man.FadeOut(texs[6], shape_im_mob))

        selem_im_mob.next_to(texs[7], man.DOWN)
        self.play(man.FadeIn(texs[7], selem_mob.move_to(selem_im_mob)))
        self.wait(3)
        self.play(man.TransformMatchingTex(texs[7], texs[8]), man.FadeOut(selem_mob), man.FadeIn(selem_im_mob))
        self.wait(2)
        self.play(man.FadeOut(texs[8], selem_im_mob))

        tex_conv = man.Text("Convolution:", font_size=man.DEFAULT_FONT_SIZE*.8)
        texs[9].next_to(tex_conv, man.RIGHT)
        man.VGroup(tex_conv, texs[9]).move_to(man.ORIGIN)
        self.play(man.FadeIn(tex_conv, texs[9]))


        self.wait(6)
