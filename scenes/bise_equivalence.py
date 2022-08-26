from functools import partial

import numpy as np
import manim as man

from tex.latex_templates import latex_template
from mobjects import ArrayImage, DilationOperationMob


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


class BiseEqDerivationAnimation(man.Scene):
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

        W = np.array([
            ["$w_{" + f"{i}{j}" + "}$" for i in range(3)] for j in range(3)
        ])

        ex1_mob1 = ArrayImage(ex1, show_value=True)
        W_mob = ArrayImage(W, show_value=True)

        ex1_mob2 = ArrayImage(ex1, show_value=True).move_to(man.ORIGIN - 3 * man.LEFT)
        selem_mob = ArrayImage(selem, mask=selem).next_to(ex1_mob2, man.RIGHT)

        dil_mob = DilationOperationMob(ex1_mob2, selem_mob)

        self.play(man.Create(dil_mob))
        # self.play(man.Create(W_mob.next_to(ex1_mob1, man.RIGHT)))
        self.wait(3)
