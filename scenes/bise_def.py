from functools import partial

import manim as man

from tex.latex_templates import latex_template


TemplateTex = partial(man.MathTex, tex_template=latex_template)


class BiseDefAnimation(man.Scene):
    def construct(self):
        dil_mob = TemplateTex(
            r"\dil{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"\indicator{S}", r"\geq", r"1", r"\Big)",
        ).shift(.5*man.UP)
        ero_mob = TemplateTex(
            r"\ero{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"\indicator{S}", r"\geq", r"\card{S}", r"\Big)"
        ).next_to(dil_mob, man.DOWN)

        all_texts = [
            TemplateTex(r"\Big(", r"\indicator{X}", r"\circledast", r"\indicator{S}", r"\geq", r"b", r"\Big)"),
            TemplateTex(r"\Big(", r"I", r"\circledast", r"\indicator{S}", r"\geq", r"b", r"\Big)"),
            TemplateTex(r"\Big(", r"I", r"\circledast", r"W", r"\geq", r"b", r"\Big)"),
            TemplateTex(r"\Big(", r"I", r"\circledast", r"\xi(W)", r"\geq", r"b", r"\Big)"),
            TemplateTex(r"\Big(", r"I", r"\circledast", r"\xi(W)", r"-", r"b", r"\geq", r"0", r"\Big)"),
            TemplateTex(r"\xi", r"\Big(", r"I", r"\circledast", r"W", r"-", r"b", r"\Big)"),
            TemplateTex(r"\xi", r"\Bigg(", r"p", r"\Big(", r"I", r"\circledast", r"W", r"-", r"b", r"\Big)", r"\Bigg)"),
        ]

        xi_expr = TemplateTex(r"\xi(u) = \frac{1}{2} \cdot \tanh(u) + \frac{1}{2}")

        self.play(man.Create(dil_mob), man.Create(ero_mob))
        self.wait()
        self.play(man.FadeOut(ero_mob), man.TransformMatchingTex(dil_mob, all_texts[0]))
        self.wait()

        for i in range(1, len(all_texts)):
            self.play(man.TransformMatchingTex(all_texts[i-1], all_texts[i]))
            if r"\xi" in all_texts[i].tex_string and (xi_expr not in self.mobjects):
                self.play(man.Create(xi_expr.next_to(all_texts[i], man.DOWN)))
            self.wait()

        self.wait(3)
