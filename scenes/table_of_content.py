import manim as man


class TableOfContentAnimation(man.Scene):
    def construct(self):
        tex_title = man.Text("Outline")

        tex_enum = [
            man.Tex(r"I.   Introduction: from convolution to morphology"),
            man.Tex(r"II.  The Binary Structuring Element (BiSE) Neuron"),
            man.Tex(r"II.1  The morphology convolution equivalence", font_size=man.DEFAULT_FONT_SIZE*.8),
            man.Tex(r"II.2  The BiSE Neuron", font_size=man.DEFAULT_FONT_SIZE*.8),
            man.Tex(r"II.3  The Binary Morphological Neural Network (BiMoNN)", font_size=man.DEFAULT_FONT_SIZE*.8),
            man.Tex(r"III. Experiments"),
            man.Tex(r"IV.  Future work"),
        ]

        for i in range(1, len(tex_enum)):
            tex_enum[i].next_to(tex_enum[i - 1], man.DOWN, aligned_edge=man.LEFT)

        man.VGroup(*tex_enum).move_to(man.ORIGIN)

        tex_title.next_to(tex_enum[0], 2*man.UP)

        self.play(man.FadeIn(tex_title))

        self.play(man.FadeIn(tex_enum[0]))
        self.play(man.FadeIn(tex_enum[1]))
        self.play(man.FadeIn(
            tex_enum[2].shift(man.RIGHT),
            tex_enum[3].shift(man.RIGHT),
            tex_enum[4].shift(man.RIGHT)
        ))
        self.wait(2)
        self.play(man.FadeIn(tex_enum[5]))
        self.wait()
        self.play(man.FadeIn(tex_enum[6]))
        self.wait(5)
