import manim as man

from tex.latex_templates import latex_template


class CnnToBimonn(man.Scene):
    def construct(self):
        texs = [
            man.MathTex("Input"),  # 0
            man.MathTex("Conv"),  # 1
            man.MathTex("Conv"),  # 2
            man.MathTex("Morpho"),  # 3
            man.MathTex("Morpho"),  # 4
            man.MathTex("Output"),  # 5
            # man.MathTex(r"\cancel{", r"Conv", r"}", tex_template=latex_template),  # 6
            # man.MathTex(r"\cancel{", r"Conv", r"}", tex_template=latex_template),  # 7
            man.MathTex(r"\cancel{Conv}", tex_template=latex_template),  # 6
            man.MathTex(r"\cancel{Conv}", tex_template=latex_template),  # 7
        ]
        side_length = 2.2
        input_mob = man.Square(side_length=side_length, color=man.BLUE)
        input_mob = man.VGroup(input_mob, texs[0].move_to(input_mob))

        arrow_mob1 = man.Arrow(start=input_mob.get_right(), end=input_mob.get_right() + man.RIGHT)

        conv_mob1 = man.Square(side_length=side_length, color=man.YELLOW)
        conv_mob1 = man.VGroup(conv_mob1, texs[1].move_to(conv_mob1)).next_to(arrow_mob1, man.RIGHT)

        dot_mob1 = man.Dot().next_to(conv_mob1, man.RIGHT)
        dot_mob2 = man.Dot().next_to(dot_mob1, man.RIGHT)
        dot_mob3 = man.Dot().next_to(dot_mob2, man.RIGHT)

        conv_mob2 = man.Square(side_length=side_length, color=man.YELLOW)
        conv_mob2 = man.VGroup(conv_mob2, texs[2].move_to(conv_mob2)).next_to(dot_mob3, man.RIGHT)

        arrow_mob2 = man.Arrow(start=conv_mob2.get_right(), end=conv_mob2.get_right() + man.RIGHT)

        output_mob = man.Square(side_length=side_length, color=man.BLUE)
        output_mob = man.VGroup(output_mob, texs[5].move_to(output_mob)).next_to(arrow_mob2, man.RIGHT)

        man.VGroup(input_mob, arrow_mob1, conv_mob1, dot_mob1, dot_mob2, dot_mob3, conv_mob2, arrow_mob2, output_mob).move_to(man.ORIGIN)

        self.play(man.Create(input_mob), run_time=2)
        self.wait(3)
        self.play(man.Create(arrow_mob1))
        self.play(man.Create(conv_mob1))
        self.play(man.Create(dot_mob1), man.Create(dot_mob2), man.Create(dot_mob3))
        self.play(man.Create(conv_mob2))
        self.play(man.Create(arrow_mob2))
        self.play(man.Create(output_mob))

        texs[3].next_to(texs[1], man.DOWN)
        texs[4].next_to(texs[2], man.DOWN)
        texs[6].move_to(texs[1])
        texs[7].move_to(texs[2])
        self.play(man.TransformMatchingTex(texs[1], texs[6]), man.TransformMatchingTex(texs[2], texs[7]), man.FadeIn(texs[3], texs[4]), run_time=3)
        self.wait(13)
