import manim as man


class TitleSlideAnimation(man.Scene):
    def construct(self):
        texs = [
            man.Text("Binary Morphological Neural", font_size=man.DEFAULT_FONT_SIZE),
            man.Text("Network", font_size=man.DEFAULT_FONT_SIZE),
            man.Text("Theodore Aouad, Hugues Talbot", slant=man.ITALIC, font_size=man.DEFAULT_FONT_SIZE*.6),
            man.Text("University of Paris-Saclay, CentraleSupelec, Inria, CVN, France", font_size=man.DEFAULT_FONT_SIZE*.6),
            man.Text("ICIP 2022, Bordeaux, France"),
        ]

        texs[1].next_to(texs[0], man.DOWN)
        texs[2].next_to(texs[1], man.DOWN)
        texs[3].next_to(texs[2], man.DOWN)
        texs[4].next_to(texs[3], 2*man.DOWN)

        man.VGroup(*texs).move_to(man.ORIGIN)

        self.add(*texs)
        self.wait(3)
        self.play(man.FadeOut(*texs))
        self.wait(3)
