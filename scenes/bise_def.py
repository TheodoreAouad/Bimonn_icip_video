from functools import partial

import manim as man

from bise.threshold_fn import tanh_threshold
from utils import TemplateMathTex


class BiseDefAnimation(man.Scene):
    def construct(self):
        # dil_mob = TemplateMathTex(
        #     r"\gamma{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"W    ", r">", r"0", r"\Big)",
        # ).shift(.5*man.UP)
        # ero_mob = TemplateMathTex(
        #     r"\ero{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"W    ", r">", r"\card{S} - 1", r"\Big)"
        # ).next_to(dil_mob, man.DOWN)

        texs = [
            man.Text("Binary Structuring Element (BiSE) Neuron", font_size=man.DEFAULT_FONT_SIZE * .7),  # 0
            TemplateMathTex(r"X \subset \mathbb{Z}^d", "~,~", r"\gamma_S(X)", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"W", r">", r"b", r"\Big)", "~,~", r"W \in [0, 1]^{\Omega}", "~,~", r"b \in \mathbb{R}"),  # 1
            TemplateMathTex(r"X \subset \mathbb{Z}^d", "~,~", r"\Big(", r"\indicator{X}", r"\circledast", r"W", r">", r"b", r"\Big)", "~,~", r"W \in [0, 1]^{\Omega}", "~,~", r"b \in \mathbb{R}"),  # 2
            TemplateMathTex(r"I \in [0, 1]^{Z^d}", "~,~", r"\Big(", r"I", r"\circledast", r"W", r">", r"b", r"\Big)", "~,~", r"W \in [0, 1]^{\Omega}", "~,~", r"b \in \mathbb{R}"),  # 3
            TemplateMathTex(r"I \in [0, 1]^{Z^d}", "~,~", r"\Big(", r"I", r"\circledast", r"\xi(\omega)", r">", r"b", r"\Big)", "~,~", r"\omega \in \mathbb{R}^{\Omega}", "~,~", r"b \in \mathbb{R}"),  # 4
            TemplateMathTex(r"I \in [0, 1]^{Z^d}", "~,~", r"\Big(", r"I", r"\circledast", r"\xi(\omega)", r"-", r"b", r">", r"0", r"\Big)", "~,~", r"\omega \in \mathbb{R}^{\Omega}", "~,~", r"b \in \mathbb{R}"),  # 5
            TemplateMathTex(r"I \in [0, 1]^{Z^d}", "~,~", r"\xi", r"\Big(", r"I", r"\circledast", r"\xi(\omega)", r"-", r"b", r"\Big)", "~,~", r"\omega \in \mathbb{R}^{\Omega}", "~,~", r"b \in \mathbb{R}"),  # 6
            TemplateMathTex(r"I \in [0, 1]^{Z^d}", "~,~", r"\xi", r"\Bigg(", r"p", r"\Big(", r"I", r"\circledast", r"\xi(\omega)", r"-", r"b", r"\Big)", r"\Bigg)", "~,~", r"\omega \in \mathbb{R}^{\Omega}", "~,~", r"b \in \mathbb{R}", r"~,~ p \in \mathbb{R}"),  # 7
            TemplateMathTex(r"\bise_{\omega, b, p}(I) = ", r"\xi", r"\Bigg(", r"p", r"\Big(", r"I", r"\circledast", r"\xi(\omega)", r"-", r"b", r"\Big)", r"\Bigg)"),  # 8
            man.Tex(r"$\gamma_S$ is erosion / dilation by $S$"),  # 9
            man.Tex(r"If $L_S(\xi(\omega)) \leq b < U_S(\xi(\omega))$, the BiSE is activated for $S$."),  # 10
        ]

        axes = man.Axes(x_range=(-3, 3), y_range=(-.1, 1.1), y_axis_config={"numbers_to_include": [0, 1]}, x_length=5, y_length=2, tips=False)

        # xi_expr = TemplateMathTex(r"\xi(u) = \frac{1}{2} \cdot \tanh(u) + \frac{1}{2}", font_size=man.DEFAULT_FONT_SIZE*.5)

        self.play(man.Create(texs[0].move_to(man.ORIGIN + 2 * man.UP)))
        # self.play(man.FadeIn(dil_mob, ero_mob))
        # self.wait()
        # self.play(man.FadeOut(ero_mob), man.TransformMatchingTex(dil_mob, texs[1]))
        self.wait()
        self.play(man.FadeIn(texs[1], texs[9].next_to(texs[1], man.DOWN)))
        self.wait()

        self.play(man.TransformMatchingTex(texs[1], texs[2]), man.FadeOut(texs[9]))
        self.wait()

        do_show = True

        for i in range(3, len(texs) - 2):
            self.play(man.TransformMatchingTex(texs[i-1], texs[i]))
            if r"\xi" in texs[i].tex_string and do_show:
                axes.next_to(texs[i], 2*man.DOWN)
                thresh_graph = axes.plot(tanh_threshold, color=man.BLUE)
                label_graph = axes.get_graph_label(thresh_graph, man.MathTex(r"\xi(u) = \frac{1}{2} \cdot \tanh(u) + \frac{1}{2}", font_size=man.DEFAULT_FONT_SIZE*.7))
                self.play(man.FadeIn(axes, thresh_graph, label_graph.next_to(thresh_graph, man.RIGHT)))
                do_show = False
            self.wait()

        self.play(man.FadeOut(axes, thresh_graph, label_graph))
        self.play(man.Create(texs[10].next_to(texs[8], man.DOWN)))

        self.wait(3)
