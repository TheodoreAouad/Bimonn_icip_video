from string import Template
import numpy as np
import manim as man
from scipy.signal import convolve2d

from example_array import example2, W1
from mobjects import ArrayImage
from bise.threshold_fn import tanh_threshold
from utils import TemplateMathTex, TemplateTex


class BiseRecapAnimation(man.Scene):
    def construct(self):
        texs = [
            TemplateMathTex(r"\forall I \in [0, 1]^{\mathbb{Z^d}} ~,~ \bise_{\omega, b, p} = "
                            r"\xi\Big(p(I \circledast \xi(\omega) - b)\Big) ~,~ \omega \in \mathbb{R}^{\Omega} ~,~ b, p \in \mathbb{R}"),  # 0
            TemplateTex("Is the BiSE activated ?"),  # 1
            TemplateTex(r"The activation depends on the input. For an almost binary input $I \in \mathcal{I}(v_1, v_2)$, being activated means", font_size=man.DEFAULT_FONT_SIZE*.7),  # 2
            TemplateMathTex(r"\exists S \subset \mathbb{Z}^d"),  # 3
            TemplateMathTex(r"\sum_{i, j \in \Omega}{\xi(\omega_{i,j}})", r"-", "(1 - v_1)", r"\min_{i, j \in S}", r"\xi(\omega_{i,j})",
                            r"\leq b", "<", "v_2", r"\sum_{i, j \in S}", r"\xi(\omega_{i,j})", font_size=man.DEFAULT_FONT_SIZE*.7),  # 4
            TemplateMathTex(r"\sum_{i, j \notin S}", r"\xi(\omega_{i, j})", r"+ v_1\sum_{i, j \in S}{\xi(\omega_{i, j})}",
                            r"\leq", "b", "<", r"v_2\min_{i, j \in S}", r"\xi(\omega_{i, j})", font_size=man.DEFAULT_FONT_SIZE*.7),  # 5
            TemplateTex(r"We check the thresholds"),  # 6
            TemplateMathTex(r"S_{\oplus} = \xi(\omega) > \tau{\oplus} ~,~ \tau_{\oplus} = \frac{b}{v_2}"),  # 7
            TemplateMathTex(r"S_{\ominus} = \xi(\omega) > \tau{\ominus} ~,~ \tau_{\ominus} = \frac{\sum_{i, j \in \Omega}\xi(\omega_{i,j}) - b}{1 - v_1}"),  # 8
            TemplateTex(r"It is activated by $S_{\oplus}$ for dilation"),  # 9
            TemplateMathTex(r"S = W > \tau_{\oplus}"),  # 10
            TemplateMathTex(r"\forall I \in \mathcal{I}(v_1, v_2) ~,~", r"\Big(", r"I \circledast W > b ", r"\Big)", "=", r"\Big(", r"X_I \oplus S", r"\Big)"),  # 11
            TemplateMathTex(r"\forall I \in \mathcal{I}(v_1, v_2) ~,~", r"\Big(", r"\bise_{\omega, b, p}(I) > v_2 ", r"\Big)", "=", r"\Big(", r"X_I \oplus S", r"\Big)"),  # 12
            TemplateMathTex(r"I "),  # 13
            TemplateMathTex(r"\bise_{\omega, b, p}"),  # 14
            TemplateMathTex(r"\bise_{\omega, b, p}(I)"),  # 15
            TemplateMathTex(r"X_I"),  # 16
            TemplateMathTex(r"\cdot \oplus S"),  # 17
            TemplateMathTex(r"X_{\bise_{\omega, b, p}(I)}"),  # 18
            TemplateTex(r"or"),  # 19
            TemplateMathTex(r"\Leftrightarrow"),  # 20
            TemplateMathTex(r"\longrightarrow"),  # 21
            TemplateMathTex(r"\longrightarrow"),  # 22
            TemplateMathTex(r"\longrightarrow"),  # 23
            TemplateMathTex(r"\longrightarrow"),  # 24
        ]

        texs[0].shift(3*man.UP)
        self.play(man.FadeIn(texs[0]))
        self.play(man.FadeIn(texs[1].next_to(texs[0], 2*man.DOWN)))
        self.play(man.TransformMatchingTex(texs[1], texs[2].next_to(texs[0], man.DOWN)))
        texs[3].next_to(texs[2], man.DOWN)
        texs[4].next_to(texs[3], man.DOWN)
        texs[19].next_to(texs[4], man.DOWN)
        texs[5].next_to(texs[19], man.DOWN)
        self.play(man.FadeIn(texs[3], texs[4], texs[5], texs[19]))
        self.wait()
        self.play(man.FadeOut(texs[2], texs[3], texs[4], texs[5], texs[19]))

        texs[7].next_to(texs[6], man.DOWN)
        texs[8].next_to(texs[7], man.DOWN)
        self.play(man.FadeIn(*texs[6:9]))

        self.wait()
        self.play(man.FadeOut(*texs[6:9]))

        self.play(man.FadeIn(texs[9], texs[10].next_to(texs[9], man.DOWN)))
        self.wait()
        self.play(man.FadeIn(texs[11].next_to(texs[10], man.DOWN)))
        self.wait()
        self.play(man.TransformMatchingTex(texs[11], texs[12].move_to(texs[11])))
        self.wait()

        self.play(man.FadeOut(texs[9], texs[10]), texs[12].animate.shift(3*man.UP))

        texs[21].next_to(texs[13], 2*man.RIGHT)
        texs[14].next_to(texs[21], 2*man.RIGHT)
        texs[22].next_to(texs[14], 2*man.RIGHT)
        texs[15].next_to(texs[22], 2*man.RIGHT)
        man.VGroup(*texs[13:16], *texs[21:23]).move_to(man.ORIGIN)
        self.play(man.FadeIn(*texs[13:16], *texs[21:23]))

        texs[20].next_to(man.VGroup(*texs[13:16], *texs[21:23]), man.DOWN)
        self.play(man.FadeIn(texs[20]))
        # texs[16].next_to(texs[19], man.DOWN)
        texs[23].next_to(texs[16], 2*man.RIGHT)
        texs[17].next_to(texs[23], 2*man.RIGHT)
        texs[24].next_to(texs[17], 2*man.RIGHT)
        texs[18].next_to(texs[24], 2*man.RIGHT)
        man.VGroup(*texs[16:19], *texs[23:25]).next_to(texs[20], man.DOWN)
        self.play(man.FadeIn(*texs[16:19], *texs[23:25]))

        self.wait(3)


# class BiseRecapAnimation(man.Scene):
#     def construct(self):
#         texts = [
#             man.Text("We define the BiSE operator"),  # 0
#             man.Text("We apply a classical convolution on normalized weights"),  # 1
#             man.Tex("We multiply by a scaling scaling factor $p$"), # 2
#             man.Text("We apply a smooth thresholding"),  # 3
#         ]

#         texs = [
#             man.MathTex("X", r"\circledast", r"\xi(", "W", ")", "- b"),  # 0
#             man.MathTex("p(", "X", r"\circledast", r"\xi(", "W", ")", "- b", ")"),  # 1
#             man.MathTex(r"\xi", r"\Big(", "p(", "X", r"\circledast", r"\xi(", "W", ")", "- b", ")", r"\Big)"),  # 2
#             man.MathTex("X"),  # 3
#         ]

#         X = example2
#         W = W1
#         selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
#         b = (W[selem].min() + W[~selem].sum()) / 2
#         p = 3
#         xi = tanh_threshold

#         XconvW = convolve2d(X, W, mode="same") - b
#         pXconvW = p * X
#         thresh_pXconvW = xi(pXconvW)

#         mobs = {
#             "X": ArrayImage(X, show_value=True),
#             "W": ArrayImage(W, show_value=True),
#             "selem": ArrayImage(selem, show_value=True),
#             "b": man.MathTex("b"),
#             "p": man.MathTex("p"),
#             "xi": man.MathTex(r"\xi"),
#             "XconvW": ArrayImage(XconvW, show_value=True),
#             "pXconvW": ArrayImage(pXconvW, show_value=True),
#             "thresh_pXconvW": ArrayImage(thresh_pXconvW, show_value=True),
#         }


#         self.play(mobs[X])
