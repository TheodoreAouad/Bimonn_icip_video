import manim as man
import numpy as np

from utils import TemplateMathTex, TemplateTex
from example_array import W1
from bise.threshold_fn import tanh_threshold


def get_rectangle_corners(bottom_left, top_right):
    return [
        (top_right[0], top_right[1]),
        (bottom_left[0], top_right[1]),
        (bottom_left[0], bottom_left[1]),
        (top_right[0], bottom_left[1]),
    ]


class AlmostBinaryAnimation(man.Scene):
    def construct(self):
        W = W1

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]).astype(bool)

        ub = W[selem].min()
        lb = W[~selem].sum()

        b = (lb + ub) / 2

        p = man.ValueTracker(3)

        def v2_fn(p):
            return tanh_threshold(p.get_value() * (ub - b))

        def v1_fn(p):
            return tanh_threshold(p.get_value() * (lb - b))

        # v2 = man.always_redraw(lambda: man.ValueTracker(tanh_threshold(p.get_value() * (ub - b))))
        # v1 = man.always_redraw(lambda: man.ValueTracker(tanh_threshold(p.get_value() * (lb - b))))

        tex_labels = {
            "b": man.MathTex("b", color=man.RED),
            "lb": man.MathTex("L", color=man.PURPLE),
            "ub": man.MathTex("U", color=man.YELLOW),
            "lb2": man.MathTex("L - b", color=man.PURPLE),
            "ub2": man.MathTex("U - b", color=man.YELLOW),
            "v2": man.MathTex("v_2", color=man.YELLOW),
            "v1": man.MathTex("v_1", color=man.PURPLE),
        }


        texs = [
            man.Tex(r"We suppose the BiSE is activated: $L_S(\xi(\omega)) \leq b < U_S(\xi(\omega))$"),  # 0
            man.always_redraw(lambda: man.MathTex(f"p = {p.get_value():.2f}", font_size=man.DEFAULT_FONT_SIZE*.8).move_to(2 * man.UP + 5 * man.LEFT)),  # 1
            # TemplateMathTex(r"\forall X \in \mathbb{Z}^d ~,~", r"\mathbbm{1}_{X} \circledast \xi(\omega)", r"\notin", r"]L, U["),  # 1
            # TemplateMathTex(r"\forall X \in \mathbb{Z}^d ~,~", r"\bise_{\omega, b, p}(\mathbbm{1}_X)", r"\notin \bigg]", r"\xi\Big(p(L - b)\Big)", r", \xi\Big(p(U - b)\Big)", r"\bigg["),  # 2
            # TemplateMathTex(r"\forall X \in \mathbb{Z}^d ~,~", r"\bise_{\omega, b, p}(\mathbbm{1}_X)", r"\in \bigg[", r"0, \xi\Big(p(L - b)\Big)\bigg]", r"\cup \bigg[\xi\Big(p(U - b)\Big)", r", 1\bigg]"),  # 3
            # TemplateTex(r"Then $\bise_{\omega, b, p}(\mathbbm{1}_X) \in \mathcal{I}(u ,v) = \{I \in [0, u] \cup [v, 1]\}$"),  # 4
            # man.Tex(r"the set of almost binary images with parameters "),  # 5
            # man.MathTex(r"u = \xi\Big(p(L - b)\Big) ~,~ v = \xi\Big(p(U - b)\Big)"),  # 6
        ]

        texs.append(man.always_redraw(lambda: man.MathTex(
            r"v_2 = \xi\Big(p(U - b)\Big) =" + f"{v2_fn(p):.2f}", color=man.YELLOW, font_size=man.DEFAULT_FONT_SIZE*.8
        ).next_to(texs[1], man.DOWN)))  # 2

        texs.append(man.always_redraw(lambda: man.MathTex(
            r"v_1 = \xi\Big(p(L - b)\Big) =" + f"{v1_fn(p):.2f}", color=man.PURPLE, font_size=man.DEFAULT_FONT_SIZE*.8
        ).next_to(texs[2], man.DOWN)))  # 3

        texs += [
            man.always_redraw(lambda: TemplateMathTex(
                r"\bise_{\omega, w, p}(\mathbbm{1}_X) \in ", f"[0, {v1_fn(p):.2f}]", r"\cup", f"[{v2_fn(p):.2f}, 1]",
                font_size=man.DEFAULT_FONT_SIZE*.8, tex_to_color_map={f"{v2_fn(p):.2f}": man.YELLOW, f"{v1_fn(p):.2f}": man.PURPLE}
            ).next_to(texs[3], man.DOWN, aligned_edge=man.LEFT))  # 4
        ]

        axes = man.Axes(
            x_range=[-1, 1, .25],
            y_range=[-.1, 1.3, .25],
            x_length=12,
            y_length=4,
            tips=True,
            y_axis_config={"numbers_to_include": [0, 1]},
            x_axis_config={"numbers_to_include": [-1, 0, 1]},
        ).shift(man.DOWN)

        tex_labels["v2"].add_updater(lambda x: x.move_to(axes.c2p(0, v2_fn(p)), aligned_edge=man.RIGHT))
        tex_labels["v1"].add_updater(lambda x: x.move_to(axes.c2p(0, v1_fn(p)), aligned_edge=man.LEFT))


        def get_rectangle_lb():
            polygon_lb = man.Polygon(
                *[
                    axes.c2p(*i)
                    for i in get_rectangle_corners(
                        (lb-b, 0), (0, v1_fn(p))
                    )
                ], color=man.PURPLE
            )
            polygon_lb.stroke_width = 1
            return polygon_lb

        def get_rectangle_ub():
            polygon_ub = man.Polygon(
                *[
                    axes.c2p(*i)
                    for i in get_rectangle_corners(
                        (0, 0), (ub - b, v2_fn(p))
                    )
                ], color=man.YELLOW
            )
            polygon_ub.stroke_width = 1
            return polygon_ub

        def get_rectangle_lb_ub():
            polygon_lb_ub = man.Polygon(
                *[
                    axes.c2p(*i)
                    for i in get_rectangle_corners(
                        (lb - b, v1_fn(p)), (ub - b, v2_fn(p))
                    )
                ], color=man.RED, fill_opacity=.3
            )
            polygon_lb_ub.stroke_width = 1
            return polygon_lb_ub


        polygon_lb = man.always_redraw(get_rectangle_lb)
        polygon_ub = man.always_redraw(get_rectangle_ub)
        polygon_lb_ub = man.always_redraw(get_rectangle_lb_ub)

        x_label = axes.get_x_axis_label(TemplateMathTex(r"\mathbbm{1}_X \circledast W - b"))
        y_label = axes.get_y_axis_label(TemplateMathTex(r"\bise_{\omega, b, p}(\mathbbm{1}_X) = \xi\Big(p(\mathbbm{1}_X \circledast W - b)\Big)"))

        bise_graph = man.always_redraw(lambda: axes.plot(lambda x: tanh_threshold(p.get_value() * x), color=man.BLUE))

        b_mob = man.Dot(color=man.RED).move_to(axes.c2p(b, 0))
        tex_labels["b"].add_updater(lambda x: x.next_to(b_mob, man.DOWN))

        lb_mob = man.Dot(color=man.PURPLE).move_to(axes.c2p(lb, 0))
        tex_labels["lb"].add_updater(lambda x: x.next_to(lb_mob, man.DOWN))
        tex_labels["lb2"].add_updater(lambda x: x.next_to(lb_mob, man.DOWN))

        ub_mob = man.Dot(color=man.YELLOW).move_to(axes.c2p(ub, 0))
        tex_labels["ub"].add_updater(lambda x: x.next_to(ub_mob, man.DOWN))
        tex_labels["ub2"].add_updater(lambda x: x.next_to(ub_mob, man.DOWN))

        self.play(man.Create(texs[0].move_to(3 * man.UP)))
        self.wait()
        self.play(man.FadeIn(axes))
        self.play(man.FadeIn(b_mob, lb_mob, ub_mob, tex_labels["b"], tex_labels["lb"], tex_labels["ub"], ))

        tex_labels["lb2"].next_to(lb_mob.copy().move_to(axes.c2p(lb - b, 0)), man.DOWN)
        tex_labels["ub2"].next_to(ub_mob.copy().move_to(axes.c2p(ub - b, 0)), man.DOWN)

        self.play(
            man.FadeOut(b_mob, tex_labels["b"]),
            lb_mob.animate.move_to(axes.c2p(lb - b, 0)), man.TransformMatchingTex(tex_labels["lb"], tex_labels["lb2"]),
            ub_mob.animate.move_to(axes.c2p(ub - b, 0)), man.TransformMatchingTex(tex_labels["ub"], tex_labels["ub2"]),
            man.FadeIn(x_label)
        )

        self.play(man.FadeIn(y_label))
        self.play(man.Create(bise_graph), man.Create(texs[1]))
        self.wait(2)
        self.play(man.Create(polygon_lb), man.Create(polygon_ub), man.FadeIn(tex_labels["v2"], tex_labels["v1"], texs[2], texs[3]))
        self.wait(5)
        self.play(man.Create(polygon_lb_ub))
        self.play(man.ReplacementTransform(polygon_lb_ub.copy(), texs[4]))

        self.wait(5)

        for i in range(3):
            if i % 2:
                value = 1
            else:
                value = 10
            self.play(p.animate.set_value(value), run_time=3)

        self.wait(10)


class AlmostBinaryDefAnimation(man.Scene):
    def construct(self):
        texs = [
            TemplateMathTex(r"\mathcal{I}(v_1, v_2) = \Big([0, v_1] \cup [v_2, 1]\Big)^{\mathbb{Z}^d}"),
            TemplateTex(r"If $I \in \mathcal{I}(v_1, v_2)$ then $X_I = (I \geq v_2) \subset \mathbb{Z}^d$"),
            TemplateTex(r"$\bise_{\omega, b, p}$ activated"),
            TemplateMathTex(r"\Rightarrow \forall X \subset \mathbb{Z}^d ~,~", r"\bise_{\omega_, p, b}(", r"\mathbbm{1}_X", r")", r"\in \mathcal{I}(v_1, v_2)"),
            TemplateMathTex(r"I \in \mathcal{I}(v_1, v_2) ~,~ ", r"\bise_{\omega_, p, b}(", "I", ")", r"~~ ?"),
        ]

        self.wait(1)
        self.play(man.Create(texs[0]))
        self.wait(5)
        self.play(texs[0].animate.shift(2*man.UP))
        self.play(man.Create(texs[1]))
        self.wait(12)

        texs[3].next_to(texs[2], man.RIGHT)
        man.VGroup(texs[2], texs[3]).move_to(man.ORIGIN)
        self.play(man.FadeOut(texs[1]))
        self.play(man.Create(texs[2]))
        self.play(man.Create(texs[3]))
        self.wait(8)
        self.play(man.FadeOut(texs[2]), man.TransformMatchingTex(texs[3], texs[4]))

        self.wait(14)


class AlmostBinaryAdaptAnimation(man.Scene):
    def construct(self):
        title_texs = [
            man.Tex("$-$ Equivalence between morphology"),
            man.Tex("and convolution"),
            man.Tex("$-$ Structuring element recovery"),
            man.Tex("by thresholding"),
            man.Tex("$-$ BiSE Output"),
        ]



        texs_eq = [
            TemplateMathTex(r"\forall X \subset \mathbb{Z}^d", "~,~", "X", r'\oplus', "S", "=", r"\mathbbm{1}_{X}", r"\circledast", r"W", r"> b"),  # 0
            # TemplateMathTex(r"\Leftrightarrow"),  # 1
            TemplateMathTex(r"\Leftrightarrow", r"\sum_{i, j \notin S}", "w_{i, j}", r"\leq", "b", "<", r"\min_{i, j \in S}", "w_{i, j}"),  # 2
            TemplateMathTex(r"\forall I \in \mathcal{I}(v_1, v_2)", "~,~", "X_I", r'\oplus', "S", "=", r"I", r"\circledast", r"W", r"> b"),  # 3
            TemplateMathTex(r"\Leftrightarrow", r"\sum_{i, j \notin S}", "w_{i, j}", r"+ v_1\sum_{i, j \in S}{w_{i, j}}", r"\leq", "b", "<", r"v_2\min_{i, j \in S}", "w_{i, j}"),  # 4
            TemplateMathTex(r"\forall I \in \mathcal{I}(v_1, v_2)", "~,~", "X_I", r'\ominus', "S", "=", r"I", r"\circledast", r"W", r"> b"),  # 5
            TemplateMathTex(r"\Leftrightarrow", r"\sum_{i, j \in \Omega}{w_{i,j}}", r"-", "(1 - v_1)", r"\min_{i, j \in S}", r"w_{i,j}", r"\leq b", "<", "v_2", r"\sum_{i, j \in S}", r"w_{i,j}"),  # 6
        ]

        texs_thresh = [
            TemplateMathTex(r"\sum_{i, j \notin S}", "w_{i, j}", r"\leq", "b", "<", r"\min_{i, j \in S}", "w_{i, j}"),  # 0
            TemplateMathTex(r"\Rightarrow"),  # 1
            TemplateMathTex(r"S = \Big(W > \tau_{\oplus}\Big)"),  # 2
            TemplateMathTex(r"\tau_{\oplus}", "=", r"b"),  # 3
            TemplateMathTex(r"\sum_{i, j \notin S}", "w_{i, j}", r"+ v_1\sum_{i, j \in S}{w_{i, j}}", r"\leq", "b", "<", r"v_2\min_{i, j \in S}", "w_{i, j}"),  # 4
            TemplateMathTex(r"\tau_{\oplus}", "=", r"\frac{b}{v_2}"),  # 5
            TemplateMathTex(r"S = \Big(W > \tau_{\ominus}\Big)"),  # 6
            TemplateMathTex(r"\sum_{i, j \in \Omega}{w_{i,j}}", r"-", "(1 - v_1)", r"\min_{i, j \in S}", r"w_{i,j}", r"\leq b", "<", "v_2", r"\sum_{i, j \in S}", r"w_{i,j}", font_size=man.DEFAULT_FONT_SIZE*.9),  # 7
            TemplateMathTex(r"\tau_{\ominus} =", r"\frac{\sum_{i, j \in \Omega}w_{i, j} - b}{1 - v_1}"),  # 8
        ]

        texs_output = [
            TemplateMathTex(r"\forall X \subset \mathbb{Z}^{d} ~,~ \bise_{\omega, b, p}(\mathbbm{1}_X) \in \mathcal{I}(v_1, v_2)"),
            TemplateMathTex(r"v_1=\xi\Big(p(", r"L", r" - b)\Big)"),
            TemplateMathTex(r"v_2=\xi\Big(p(", r"U", r" - b)\Big)"),
            TemplateMathTex(r"\forall I \in \mathcal{I}(z_1, z_2) ~,~ \bise_{\omega, b, p}(I) \in \mathcal{I}(v_1, v_2)"),
            TemplateMathTex(r"v_1=\xi\Big(p(", r"L^{new}", r"- b)\Big)"),
            TemplateMathTex(r"v_2=\xi\Big(p(", r"U^{new}", r"- b)\Big)"),
        ]


        title_texs[1].next_to(title_texs[0], man.DOWN)
        title_texs[2].next_to(title_texs[1], 2*man.DOWN)
        title_texs[3].next_to(title_texs[2], man.DOWN)
        title_texs[4].next_to(title_texs[3], 2*man.DOWN)

        man.VGroup(*title_texs).move_to(man.ORIGIN)

        self.play(man.FadeIn(title_texs[0], title_texs[1]))
        self.play(man.FadeIn(title_texs[2], title_texs[3]))
        self.play(man.FadeIn(title_texs[4]))
        self.wait()
        self.play(man.FadeOut(*title_texs))

        title_texs[0].move_to(man.ORIGIN + 3 * man.UP)
        title_texs[1].next_to(title_texs[0], man.DOWN)

        title_texs[2].move_to(title_texs[0])
        title_texs[3].next_to(title_texs[2], man.DOWN)

        title_texs[4].move_to(title_texs[0])

        # Conv Morpho Equivalence
        texs = texs_eq
        texs[1].next_to(texs[0], man.DOWN)


        self.play(man.Create(title_texs[0]))
        self.play(man.Create(title_texs[1]))
        self.play(man.Create(texs[0]))
        self.play(man.Create(texs[1]))

        self.wait()
        self.play(man.TransformMatchingTex(texs[0], texs[2].move_to(texs[0])), texs[1].animate.set_opacity(.3))
        self.wait(4)
        self.play(man.TransformMatchingTex(texs[1], texs[3].move_to(texs[1])))
        # erosion
        self.wait(2)
        self.play(man.TransformMatchingTex(texs[2], texs[4].move_to(texs[2])), man.TransformMatchingTex(texs[3], texs[5].move_to(texs[3])))

        self.wait(3)
        self.play(man.FadeOut(title_texs[0], title_texs[1], texs[5], texs[4]))

        # Threshold
        self.play(man.FadeIn(title_texs[2], title_texs[3]))
        texs = texs_thresh
        texs[1].next_to(texs[0], man.RIGHT)
        texs[2].next_to(texs[1], man.RIGHT)
        grp1 = man.VGroup(*texs[:3]).move_to(man.ORIGIN + 2*man.RIGHT)
        texs[3].next_to(grp1, man.DOWN).shift(man.DOWN*.2)
        self.play(man.Create(grp1), man.Create(texs[3]))
        self.wait(1)
        self.play(man.TransformMatchingTex(texs[0], texs[4].move_to(texs[0], aligned_edge=man.RIGHT)), texs[3].animate.set_opacity(.3))
        self.wait(1)
        self.play(man.TransformMatchingTex(texs[3], texs[5].move_to(texs[3])))
        self.wait()
        # # erosion
        self.play(
            man.TransformMatchingTex(texs[5], texs[8].move_to(texs[5], aligned_edge=man.UP)),
            man.TransformMatchingTex(texs[2], texs[6].move_to(texs[2])),
            man.TransformMatchingTex(texs[4], texs[7].move_to(texs[4], aligned_edge=man.RIGHT))
        )


        self.wait()
        self.play(man.FadeOut(title_texs[2], title_texs[3], texs[8], texs[6], texs[7], texs[1]))


        # Bise Output
        self.play(man.FadeIn(title_texs[4]))
        texs = texs_output
        texs[1].next_to(texs[0], man.DOWN)
        texs[2].next_to(texs[1], man.DOWN)

        self.play(man.Create(texs[0]))
        self.play(man.Create(texs[1]), man.Create(texs[2]))
        self.wait()
        self.play(man.TransformMatchingTex(texs[0], texs[3]), man.TransformMatchingTex(texs[1], texs[4].move_to(texs[1])), man.TransformMatchingTex(texs[2], texs[5].move_to(texs[2])))

        self.wait(3)
