import manim as man


class FutureWorkAnimation(man.Scene):
    def construct(self):
        tex_title = man.Text("Future Work")

        tex_enum = [
            man.Tex(r"$\bullet$ BiSE init: identical dual training, avoid gradient vanishing"),
            man.Tex(r"$\bullet$ Allow a binarization even if BiSE is not activated"),
            man.Tex(r"$\bullet$ Add intersection / union of operators"),
            man.Tex(r"$\bullet$ Extend to gray scale morphology"),
        ]

        for i in range(1, len(tex_enum)):
            tex_enum[i].next_to(tex_enum[i - 1], man.DOWN, aligned_edge=man.LEFT)

        man.VGroup(*tex_enum).move_to(man.ORIGIN)

        tex_title.next_to(tex_enum[0], 2*man.UP)

        self.wait(3)
        self.play(man.FadeIn(tex_title))
        self.wait(4)

        self.play(man.FadeIn(tex_enum[0]))
        self.wait(8)
        self.play(man.FadeIn(tex_enum[1]))
        self.wait(4)
        self.play(man.FadeIn(tex_enum[2]))
        self.wait(8)
        self.play(man.FadeIn(tex_enum[3]))
        self.wait(3)

        self.play(man.FadeOut(*tex_enum, tex_title))

        self.play(man.FadeIn(man.Tex("theodore.aouad@centralesupelec.fr")))
        self.wait(7)
