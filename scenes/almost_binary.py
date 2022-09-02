import manim as man

from utils import TemplateMathTex, TemplateTex


class AlmostBinaryAnimation(man.Scene):
    def construct(self):
        texs = [
            man.Tex(r"We suppose the BiSE is activated: $L_S(\xi(\omega)) \leq b < U_S(\xi(\omega))$"),  # 0
            TemplateMathTex(r"\forall X \in \mathbb{Z}^d ~,~", r"\mathbbm{1}_{X} \circledast \xi(\omega)", r"\notin", r"]L, U["),  # 1
            TemplateMathTex(r"\forall X \in \mathbb{Z}^d ~,~", r"\bise_{\omega, b, p}(\mathbbm{1}_X)", r"\notin \bigg]", r"\xi\Big(p(L - b)\Big)", r", \xi\Big(p(U - b)\Big)", r"\bigg["),  # 2
            TemplateMathTex(r"\forall X \in \mathbb{Z}^d ~,~", r"\bise_{\omega, b, p}(\mathbbm{1}_X)", r"\in \bigg[", r"0, \xi\Big(p(L - b)\Big)\bigg]", r"\cup \bigg[\xi\Big(p(U - b)\Big)", r", 1\bigg]"),  # 3
            TemplateTex(r"Then $\bise_{\omega, b, p}(\mathbbm{1}_X) \in \mathcal{I}(u ,v) = \{I \in [0, u] \cup [v, 1]\}$"),  # 4
            man.Tex(r"the set of almost binary images with parameters "),  # 5
            man.MathTex(r"u = \xi\Big(p(L - b)\Big) ~,~ v = \xi\Big(p(U - b)\Big)"),  # 6
        ]

        self.play(man.FadeIn(texs[0].shift(2*man.UP)))
        self.play(man.FadeIn(texs[1]))
        self.wait()
        self.play(man.TransformMatchingTex(texs[1], texs[2]))
        self.wait()
        self.play(man.TransformMatchingTex(texs[2], texs[3]))
        self.wait()
        self.play(texs[3].animate.shift(man.UP))

        texs[4].next_to(texs[3], man.DOWN)
        texs[5].next_to(texs[4], man.DOWN)
        texs[6].next_to(texs[5], man.DOWN)

        self.play(man.Create(texs[4]))
        self.play(man.Create(texs[5]))
        self.play(man.Create(texs[6]))

        self.wait(3)
