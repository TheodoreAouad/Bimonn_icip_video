import numpy as np
import manim as man

from mobjects import ArrayImage
from example_array import W1


class ThresholdingAnimation(man.Scene):
    def construct(self):
        W = W1

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]).astype(bool)

        Xs, Ys = np.where(selem)
        barXs, barYs = np.where(~selem)

        b_dila = (W[selem].min() + W[~selem].sum()) / 2
        b_ero = W.sum() - b_dila
        tau_dila = "b"
        tau_ero = r"\sum_{i, j \in \Omega}w_{i, j} - b"

        W_thresh = (W > b_dila).astype(int)

        W_mob = ArrayImage(W, show_value=True)
        W_thresh = ArrayImage(W_thresh, show_value=True)
        selem_mob = ArrayImage(selem.astype(int), mask=selem, show_value=False, cmap='Blues')

        texs = [
            man.MathTex(r"S", r"="),  # 0
            man.MathTex(r"W", r"="),  # 1
            man.MathTex(r"b", r"=", f"{b_dila}"),  # 2
            man.MathTex(r"\sum_", r"{i, j \notin S}", r"w_{i,j}", r"=", f"{W[~selem].sum()}"),  # 3
            man.MathTex(r"\min_", r"{i, j \in S}", r"w_{i,j}", r"=", f"{W[selem].min()}"),  # 4
            man.MathTex(r"\sum_", r"{i, j \notin S}", r"w_{i,j}", r"\leq", r"b", r"<", r"\min_", r"{i, j \in S}", r"w_{i,j}"),  # 5
            man.Tex(r"$\Leftrightarrow$~", r"$\oplus$~", r"by~", r"$S$"),  # 6
            man.MathTex(",", r"\tau_", r"{\oplus}", r"=", f"{tau_dila}"),  # 7
            man.MathTex(r"W"),  # 8
            man.MathTex(r"W", ">", r"\tau_", r"{\oplus}"),  # 9
            man.MathTex(r"="),  # 10
            man.MathTex(r"b", r"=", f"{b_ero}", tex_to_color_map={f"{b_ero}": man.GREEN}),  # 11
            man.MathTex(
                r"\sum_", r"{i, j \in \Omega}", r"w_{i,j}", r"-", r"\min_", r"{i, j \in S}", r"{w_{i, j}}", r"\leq", r"b", r"<", r"\sum_", r"{i, j \in S}", r"w_{i,j}",
                tex_to_color_map={
                    r"{i, j \in \Omega}": man.GREEN,
                    r"-": man.GREEN,
                    r"\min_": man.GREEN,
                    r"{w_{i, j}}": man.GREEN,
                    r"\sum_": man.GREEN,
                    r"{i, j \in S}": man.GREEN,
                    r"w_{i,j}": man.GREEN,
                }
            ),  # 12
            man.MathTex(",", r"\tau_", r"{\ominus}", r"=", f"{tau_ero}", tex_to_color_map={f"{tau_ero}": man.GREEN, r"{\ominus}": man.GREEN}),  # 13
            man.MathTex(r"W", ">", r"\tau_", r"{\ominus}", tex_to_color_map={r"{\ominus}": man.GREEN}),  # 14
            man.Tex(r"$\Leftrightarrow$~", r"$\ominus$~", r"by~", r"$S$", tex_to_color_map={r"$\ominus$": man.GREEN}),  # 15
            man.MathTex(f"= {b_dila:.2f}"),  # 16
            man.MathTex(f"= {W.sum() - b_ero:.2f}", color=man.GREEN),  # 17
        ]

        selem_mob.next_to(texs[0], man.RIGHT)
        W_mob.next_to(texs[1], man.RIGHT)
        man.VGroup(W_mob, texs[1]).next_to(man.VGroup(selem_mob, texs[0]), man.DOWN)
        texs[2].next_to(man.VGroup(W_mob, texs[1]), man.DOWN)
        man.VGroup(selem_mob, texs[0], W_mob, texs[1], texs[2]).move_to(man.ORIGIN)

        texs[4].next_to(texs[3], man.DOWN)
        texs[5].next_to(texs[4], man.DOWN)
        man.VGroup(texs[3], texs[4], texs[5]).move_to(man.ORIGIN + man.RIGHT)

        self.play(man.FadeIn(texs[0], selem_mob, texs[1], W_mob, texs[2]))

        self.play(man.VGroup(texs[0], selem_mob, texs[1], W_mob, texs[2]).animate.shift(5.5*man.LEFT))

        grp_in = man.VGroup(*[W_mob.get_pixel(i, j).copy() for (i, j) in zip(Xs, Ys)])
        grp_out = man.VGroup(*[W_mob.get_pixel(i, j).copy() for (i, j) in zip(barXs, barYs)])

        self.play(man.Transform(grp_out, texs[3]))
        self.play(man.Transform(grp_in, texs[4]))

        self.play(man.TransformMatchingTex(texs[3].copy(), texs[5]), man.TransformMatchingTex(texs[4].copy(), texs[5]))
        # self.play(man.Create(texs[5]))
        self.play(man.FadeOut(grp_in, grp_out))
        self.play(texs[5].animate.shift(4*man.UP))
        self.play(man.FadeIn(texs[6].next_to(texs[5], man.DOWN)))

        W_mob2 = W_mob.copy().shift(5 * man.RIGHT)

        self.play(man.FadeIn(W_mob2, texs[8].next_to(W_mob2, man.DOWN)))
        texs[7].next_to(W_mob2, 2*man.RIGHT)
        self.play(man.FadeIn(texs[7], texs[16].next_to(texs[7], man.RIGHT)))

        texs[9].move_to(texs[8])
        self.play(man.FadeIn(W_thresh.move_to(W_mob2)), man.FadeOut(W_mob2), man.TransformMatchingTex(texs[8], texs[9]))
        tex0 = texs[0].copy()
        self.play(man.FadeIn(tex0.next_to(W_thresh, man.LEFT)))

        self.wait(3)

        texs[11].move_to(texs[2])
        texs[12].move_to(texs[5])
        texs[13].move_to(texs[7], aligned_edge=man.LEFT)
        texs[14].move_to(texs[9])
        texs[15].move_to(texs[6])
        self.play(
            man.TransformMatchingTex(texs[2], texs[11]),
            man.TransformMatchingTex(texs[5], texs[12]),
            man.TransformMatchingTex(texs[7], texs[13]),
            man.TransformMatchingTex(texs[9], texs[14]),
            man.TransformMatchingTex(texs[6], texs[15]),
            man.TransformMatchingTex(texs[16], texs[17].next_to(texs[13], man.RIGHT)),
        )

        self.wait(10)
