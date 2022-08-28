import os, sys

from mobjects.array_image import Pixel

print(os.getcwd())
print(sys.path)

from functools import partial

import numpy as np
import manim as man
from skimage.morphology import dilation

from tex.latex_templates import latex_template
from mobjects import ArrayImage, DilationOperationMob, ConvolutionOperationMob
from utils import play_horizontal_sequence, play_transforming_tex


TemplateTex = partial(man.MathTex, tex_template=latex_template)


class BiseEqQuestionAnimation(man.Scene):
    def construct(self):
        eq_texts = [
            TemplateTex(r"\dil{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"\indicator{S}", r"\geq", r"1", r"\Big)",),
            TemplateTex(r"\dil{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"W", r"\geq", r"b", r"\Big)",),
            TemplateTex(r"\dil{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"W", r"\geq", r"b", r"\Big)", r"~,~ W \in [0, 1]^{\Omega}"),
        ]

        comment_texts = [
            man.Tex(r"What is the relationship between $W$ and $b$ ?")
        ]


        self.play(man.Create(eq_texts[0]))
        self.wait()
        self.play(man.TransformMatchingTex(eq_texts[0], eq_texts[1]))
        self.play(man.TransformMatchingTex(eq_texts[1], eq_texts[2]))
        self.play(man.Create(comment_texts[0].next_to(eq_texts[1], man.DOWN)))
        self.wait(3)


class BiseEq1DerivationAnimation(man.Scene):
    def construct(self):
        ex1 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])

        dil_ex1 = dilation(ex1, selem)

        W = np.array([
            ["$w_{" + f"{i}{j}" + "}$" for i in range(3)] for j in range(3)
        ])

        ex1_mob1 = ArrayImage(ex1, show_value=True)
        sq1 = man.Square(side_length=ex1_mob1.horizontal_stretch, color='red').move_to(ex1_mob1.get_center())
        grp1 = man.VGroup(ex1_mob1, sq1)
        selem_mob = ArrayImage(selem, mask=selem, cmap='Blues')
        dil_ex1_mob1 = Pixel(value=dil_ex1[1, 1], color=man.YELLOW, show_value=True, height=ex1_mob1.horizontal_stretch)

        ex1_mob2 = ArrayImage(ex1, show_value=True)
        sq2 = man.Square(side_length=ex1_mob2.horizontal_stretch, color='red').move_to(ex1_mob2.get_center())
        grp2 = man.VGroup(ex1_mob2, sq2)
        W_mob = ArrayImage(W, show_value=True)

        texs = [
            man.Tex(r"As~~"),
            man.Tex(r"~~, we have"),
            man.MathTex(r">", r"b", r"\Rightarrow", ),
            man.MathTex(r"w_{1, 2}", r"> b")
        ]

        dil_mob = DilationOperationMob(grp1, selem_mob, dil_ex1_mob1, show_braces=False)

        conv_mob = ConvolutionOperationMob(grp2, W_mob, show_braces=False)

        play_horizontal_sequence(self, [texs[0], dil_mob, texs[1]], origin=man.ORIGIN + 2 * man.UP + 3*man.LEFT)
        play_horizontal_sequence(self, [conv_mob, texs[2], texs[3]], origin=dil_mob.get_left() + 3*man.DOWN, aligned_edge=man.LEFT)

        for (i, j) in [(0, 1), (1, 1), (1, 2), (1, 0)]:
            self.play(man.FadeOut(grp1), man.FadeOut(grp2))

            ex1 = np.zeros((3, 3), dtype=int); ex1[i, j] = 1
            ex1_mob1.update_array(ex1)
            ex1_mob2.update_array(ex1)
            new_tex = man.MathTex(r"w_{" + f"{i}, {j}" + r"}", r"> b").move_to(texs[3].get_center())

            self.play(man.FadeIn(grp1), man.FadeIn(grp2), man.TransformMatchingTex(texs[3], new_tex))
            texs[3] = new_tex

        self.play(man.FadeOut(texs[0], dil_mob, texs[1], texs[2], conv_mob))
        self.play(man.TransformMatchingTex(texs[3], man.MathTex(r"\min_{i, j \in S}", r"w_{i,j}", r"> b").move_to(man.ORIGIN)))

        self.wait(3)


class BiseEq2DerivationAnimation(man.Scene):
    def construct(self):
        ex1 = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])

        dil_ex1 = dilation(ex1, selem)

        W = np.array([
            ["$w_{" + f"{i}{j}" + "}$" for i in range(3)] for j in range(3)
        ])

        ex1_mob1 = ArrayImage(ex1, show_value=True)
        sq1 = man.Square(side_length=ex1_mob1.horizontal_stretch, color='red').move_to(ex1_mob1.get_center())
        grp1 = man.VGroup(ex1_mob1, sq1)
        selem_mob = ArrayImage(selem, mask=selem, cmap='Blues')
        dil_ex1_mob1 = Pixel(value=dil_ex1[1, 1], color=man.PURPLE, show_value=True, height=ex1_mob1.horizontal_stretch)

        ex1_mob2 = ArrayImage(ex1, show_value=True)
        sq2 = man.Square(side_length=ex1_mob2.horizontal_stretch, color='red').move_to(ex1_mob2.get_center())
        grp2 = man.VGroup(ex1_mob2, sq2)
        W_mob = ArrayImage(W, show_value=True)

        texs = [
            man.Tex(r"As~~"),
            man.Tex(r"~~, we have"),
            man.MathTex(r"\leq", r"b", r"\Rightarrow", ),
            man.MathTex(r"w_{0, 1}", r"+", r"w_{1, 2}", r"+", r"w_{2, 1}", r"+", r"w_{1, 0}" r"\leq b")
        ]

        dil_mob = DilationOperationMob(grp1, selem_mob, dil_ex1_mob1, show_braces=False)

        conv_mob = ConvolutionOperationMob(grp2, W_mob, show_braces=False)

        play_horizontal_sequence(self, [texs[0], dil_mob, texs[1]], origin=man.ORIGIN + 2 * man.UP + 6*man.LEFT)
        play_horizontal_sequence(self, [conv_mob, texs[2], texs[3]], origin=dil_mob.get_left() + 3*man.DOWN, aligned_edge=man.LEFT)


        self.play(man.FadeOut(texs[0], dil_mob, texs[1], texs[2], conv_mob))
        self.play(man.TransformMatchingTex(texs[3], man.MathTex(r"\sum_{i, j \in S}", r"w_{i,j}", r"\leq b").move_to(man.ORIGIN)))

        self.wait(3)


class BiseEq3DerivationAnimation(man.Scene):
    def construct(self):
        texs = [
            # TemplateTex(r'\ero{X}{S}')
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{",
            r"\indicator{", r"X", r"}", r"}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\ero{X}{S}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{",
            r"\indicator{", r"\bar{X}", r"}", r"}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\ero{\bar{X}}{S}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{",
            r"\indicator{", r"\bar{X}", r"}", r"}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\overline{", r"\dil{\bar{X}}{S}", r"}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{",
            r"\indicator{", r"\bar{X}", r"}", r"}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\overline{", r"\dil{\bar{X}}{S}", r"}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\sum_{i \in \Omega}", r"w_i", r"-", r"\conv{(",
            r"\indicator{", r"{X}", r"}", r")}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\overline{", r"\dil{\bar{X}}{S}", r"}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\sum_{i \in \Omega}", r"w_i", r"-", r"\sup_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{(",
            r"\indicator{", r"{X}", r"}", r")}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\overline{", r"\dil{\bar{X}}{S}", r"}", r"\}"),
            TemplateTex(r"U_{erosion} =", r"\sum_{i \in \Omega}", r"w_i", r"-", r"L_{dilation}")
        ]

        play_transforming_tex(self, texs, origin=man.ORIGIN, run_time=2)
