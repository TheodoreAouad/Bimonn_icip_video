from functools import partial
from string import Template
import numpy as np
import manim as man
from scipy.signal import convolve2d

from example_array import example2, W1, W2
from mobjects import ArrayImage
from bise.threshold_fn import tanh_threshold
from utils import TemplateMathTex, TemplateTex
from run_times import tanh_run_times


CurTemplateMathTex = partial(TemplateMathTex, font_size=man.DEFAULT_FONT_SIZE*.5)


def ero_bounds(W, S, v1=0, v2=1):
    if S.sum() == 0:
        minS = 0
    else:
        minS = W[S].min()
    return W[W > 0].sum() - (1 - v1) * minS, v2 * W[S & (W > 0)].sum() + W[W < 0].sum()


def dila_bounds(W, S, v1=0, v2=1):
    if S.sum() == 0:
        minS = 0
    else:
        minS = W[S].min()
    return W[(~S) & (W > 0)].sum() + v1 * W[S & (W > 0)].sum(), v2 * minS + W[W < 0].sum()


def ero_thresh(W, b, v1=0, v2=1):
    return (W.sum() - b) / (1 - v1)


def dila_thresh(W, b, v1=0, v2=1):
    return b / v2


class BiseRecapAnimation(man.Scene):
    def construct(self):
        texs = {
            "bise_def": TemplateMathTex(r"\forall I \in [0, 1]^{\mathbb{Z^d}} ~,~ \bise_{\omega, b, p} = "
                            r"\xi\Big(p(I \circledast \xi(\omega) - b)\Big) ~,~ \omega \in \mathbb{R}^{\Omega} ~,~ b, p \in \mathbb{R}"),
            "bin_to_ab": TemplateTex(r"Current input: $\indicator{X} \in \mathcal{I}(v_1^1, v_2^1) ~,~ (v_1^1, v_2^1) = (0, 1)$", font_size=man.DEFAULT_FONT_SIZE*.7),
            "w_label": TemplateMathTex(r"\xi(\omega)="),
            "all_candidate_selem": TemplateTex(r"Candidate $S \subset \Omega$"),
            "thresh_candidate_selem": TemplateTex(r"$S_{\oplus} = \xi(\omega) > \tau_{\oplus}$ or $S_{\ominus} = \xi(\omega) > \tau_{\ominus}$", font_size=man.DEFAULT_FONT_SIZE*.7),
        }


        selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(bool)
        W_no = W2

        bise1_mob = man.Circle(radius=.5, color=man.GREEN, fill_opacity=.5)
        bise1_label = CurTemplateMathTex(r"\bise_{\omega_1, p_1, b_1}").add_updater(lambda x: x.move_to(bise1_mob))

        self.W_mob = ArrayImage(W_no, show_value=True, vmin_cmap=0, vmax_cmap=1)
        self.selem_mob = ArrayImage(selem.astype(int), show_value=True, vmin_cmap=0, vmax_cmap=1, cmap='Blues')
        texs["all_candidate_selem"].add_updater(lambda x: x.next_to(self.selem_mob, man.UP))


        axis = man.Line(start=man.ORIGIN, end=man.ORIGIN + 4 * man.RIGHT).shift(man.UP)
        axis_label = man.always_redraw(lambda:
            man.MathTex(r"(L \leq b < U)?", tex_to_color_map={"L": man.PURPLE, "b": man.RED, "U": man.YELLOW}).next_to(axis, man.RIGHT)
        )
        # axis_ero = man.Line(start=man.ORIGIN, end=man.ORIGIN + 4 * man.RIGHT).shift(man.DOWN)

        Ldila, Udila = dila_bounds(W_no, selem, v1=0, v2=1)
        Lero, Uero = ero_bounds(W_no, selem, v1=0, v2=1)


        self.b_mob = man.ValueTracker(.5)
        p_mob = man.ValueTracker(1)
        self.Ldila_mob, self.Udila_mob = man.ValueTracker(Ldila), man.ValueTracker(Udila)
        self.Lero_mob, self.Uero_mob = man.ValueTracker(Lero), man.ValueTracker(Uero)

        self.OK_symbol = man.MathTex("KO", color=man.RED)

        Ldila_dot = man.Dot(color=man.PURPLE).add_updater(lambda x: x.move_to(self.c2p(axis, self.Ldila_mob.get_value())))
        Ldila_dot_label = man.MathTex(r"L_{\oplus}(S)", font_size=man.DEFAULT_FONT_SIZE*.5, color=man.PURPLE).add_updater(lambda x: x.next_to(Ldila_dot, man.DOWN))

        Udila_dot = man.Dot(color=man.YELLOW).add_updater(lambda x: x.move_to(self.c2p(axis, self.Udila_mob.get_value())))
        Udila_dot_label = man.MathTex(r"U_{\oplus}(S)", font_size=man.DEFAULT_FONT_SIZE*.5, color=man.YELLOW).add_updater(lambda x: x.next_to(Udila_dot, man.UP))

        b_dot = man.Dot(color=man.RED).add_updater(lambda x: x.move_to(self.c2p(axis, self.b_mob.get_value())))

        Lero_dot = man.Dot(color=man.PURPLE).add_updater(lambda x: x.move_to(self.c2p(axis, self.Lero_mob.get_value())))
        Lero_dot_label = man.MathTex(r"L_{\ominus}(S)", font_size=man.DEFAULT_FONT_SIZE*.5, color=man.PURPLE).add_updater(lambda x: x.next_to(Lero_dot, man.DOWN))

        Uero_dot = man.Dot(color=man.YELLOW).add_updater(lambda x: x.move_to(self.c2p(axis, self.Uero_mob.get_value())))
        Uero_dot_label = man.MathTex(r"U_{\ominus}(S)", font_size=man.DEFAULT_FONT_SIZE*.5, color=man.YELLOW).add_updater(lambda x: x.next_to(Uero_dot, man.UP))

        texs["bise_def"].move_to(3 * man.UP)
        texs["bin_to_ab"].next_to(texs["bise_def"], man.DOWN, aligned_edge=man.LEFT)
        bise1_mob.move_to(5*man.LEFT + .5*man.UP)
        # bise1_mob.next_to(texs["bin_to_ab"], man.DOWN)

        self.W_mob.next_to(texs["w_label"], man.RIGHT)
        grpW = man.VGroup(self.W_mob, texs["w_label"]).next_to(bise1_mob, man.DOWN)

        texs["b_label"] = man.always_redraw(lambda: TemplateMathTex(f"b = {self.b_mob.get_value():.2f}").next_to(grpW, man.DOWN))
        texs["p_label"] = man.always_redraw(lambda: TemplateMathTex(f"p = {p_mob.get_value()}").next_to(texs["b_label"], man.DOWN))


        self.play(man.FadeIn(texs["bise_def"]))
        self.wait(10)
        self.play(man.FadeIn(texs["bin_to_ab"]))
        self.play(man.FadeIn(bise1_mob, bise1_label))
        self.wait(10)
        self.play(man.FadeIn(
            texs["w_label"], self.W_mob,
            texs["b_label"],
        ))

        self.wait(7)
        self.selem_mob.next_to(axis, 8*man.DOWN)

        self.play(man.FadeIn(axis,))
        self.play(man.FadeIn(
            Ldila_dot, Ldila_dot_label,
            Udila_dot, Udila_dot_label,
            Lero_dot, Lero_dot_label,
            Uero_dot, Uero_dot_label,
            axis_label, b_dot,
            self.selem_mob, texs["all_candidate_selem"],
            self.OK_symbol.next_to(axis_label, man.DOWN),
        ))


        run_time_fn = tanh_run_times(start_value=.5, end_value=.1, fade_start=3)
        for i in range(10):
            cur_selem = np.random.randint(0, 2, (3, 3))
            self.update_axis_selem(cur_selem, do_fadeout=True, run_time=run_time_fn(i))


        tau_dila = dila_thresh(self.W_mob.array, self.b_mob.get_value(), v1=0, v2=1)
        tau_ero = ero_thresh(self.W_mob.array, self.b_mob.get_value(), v1=0, v2=1)

        # tau_dila = min(self.W_mob.array.max(), tau_dila)
        # tau_ero = min(self.W_mob.array.max(), tau_ero)

        selem_dila = self.W_mob.array >= tau_dila
        selem_ero = self.W_mob.array >= tau_ero

        selem_dila_label = TemplateMathTex(r"\mathbbm{1}_{S_{\oplus}} = ", font_size=man.DEFAULT_FONT_SIZE*.7).next_to(self.selem_mob, man.LEFT)
        selem_ero_label = TemplateMathTex(r"\mathbbm{1}_{S_{\ominus}} = ", font_size=man.DEFAULT_FONT_SIZE*.7).next_to(self.selem_mob, man.LEFT)


        self.play(man.TransformMatchingTex(texs["all_candidate_selem"], texs["thresh_candidate_selem"].move_to(texs["all_candidate_selem"])))

        self.play(man.FadeOut(self.selem_mob))
        self.wait(8)
        self.play(man.FadeIn(selem_ero_label))
        self.update_axis_selem(selem_ero)
        self.play(man.FadeOut(selem_ero_label, self.selem_mob))
        self.play(man.FadeIn(selem_dila_label))
        self.update_axis_selem(selem_dila)
        self.play(man.FadeOut(selem_dila_label))

        self.wait(5)

        self.play(man.FadeOut(self.W_mob))
        self.W_mob.update_array(W1)
        best_ldila, best_udila = dila_bounds(self.W_mob.array, selem.astype(bool))
        self.play(man.FadeIn(self.W_mob), self.b_mob.animate.set_value((best_ldila + best_udila) / 2))

        tau_dila = dila_thresh(self.W_mob.array, self.b_mob.get_value(), v1=0, v2=1)
        tau_ero = ero_thresh(self.W_mob.array, self.b_mob.get_value(), v1=0, v2=1)

        # tau_dila = min(self.W_mob.array.max(), tau_dila)
        # tau_ero = min(self.W_mob.array.max(), tau_ero)


        selem_dila = self.W_mob.array >= tau_dila
        selem_ero = self.W_mob.array >= tau_ero

        selem_dila_label = TemplateMathTex(r"\mathbbm{1}_{S_{\oplus}} = ", font_size=man.DEFAULT_FONT_SIZE*.7).next_to(self.selem_mob, man.LEFT)
        selem_ero_label = TemplateMathTex(r"\mathbbm{1}_{S_{\ominus}} = ", font_size=man.DEFAULT_FONT_SIZE*.7).next_to(self.selem_mob, man.LEFT)

        self.play(man.FadeOut(self.selem_mob))
        self.play(man.FadeIn(selem_ero_label))
        self.update_axis_selem(selem_ero)
        self.wait(2)
    
        self.play(man.FadeOut(selem_ero_label, self.selem_mob))
        self.play(man.FadeIn(selem_dila_label))
        self.update_axis_selem(selem_dila)
        self.wait(3)

        self.play(man.FadeOut(selem_dila_label))
        self.play(man.FadeOut(texs["thresh_candidate_selem"], self.selem_mob, selem_dila_label))

        texs["almost_bin_output"] = TemplateTex("The BiSE output is almost binary with", font_size=man.DEFAULT_FONT_SIZE*.7).shift(man.DOWN + 1.5*man.RIGHT)
        texs["output_v1"] = TemplateMathTex(r"v_1^{2}=\xi\Big(p(", r"L_{\oplus}(S)", r" - b)\Big) < .5", font_size=man.DEFAULT_FONT_SIZE*.7).next_to(texs["almost_bin_output"], man.DOWN)
        texs["output_v2"] = TemplateMathTex(r"v_2^{2}=\xi\Big(p(", r"U_{\oplus}(S)", r" - b)\Big) > .5", font_size=man.DEFAULT_FONT_SIZE*.7).next_to(texs["output_v1"], man.DOWN, aligned_edge=man.LEFT)

        self.play(man.FadeIn(texs["almost_bin_output"], texs["output_v1"], texs["output_v2"]))

        self.wait(15)

    def c2p(self, axis, value):
        return axis.point_from_proportion(value / 7)

    def update_axis_selem(self, new_selem, do_fadeout=False, run_time=1):
        if do_fadeout:
             self.play(man.FadeOut(self.selem_mob), run_time=run_time)

        self.play(man.FadeOut(self.OK_symbol), run_time=run_time)
        self.selem_mob.update_array(new_selem.astype(int))
        self.play(man.FadeIn(self.selem_mob), run_time=run_time)

        Ldila, Udila = dila_bounds(self.W_mob.array, new_selem.astype(bool), v1=0, v2=1)
        Lero, Uero = ero_bounds(self.W_mob.array, new_selem.astype(bool), v1=0, v2=1)

        bval = self.b_mob.get_value()
        if (Lero <= bval < Uero) or (Ldila <= bval < Udila):
            self.OK_symbol = man.MathTex("OK", color=man.GREEN).move_to(self.OK_symbol)
        else:
            self.OK_symbol = man.MathTex("KO", color=man.RED).move_to(self.OK_symbol)

        self.play(
            self.Ldila_mob.animate.set_value(Ldila),
            self.Udila_mob.animate.set_value(Udila),
            self.Lero_mob.animate.set_value(Lero),
            self.Uero_mob.animate.set_value(Uero),
            man.FadeIn(self.OK_symbol),
            run_time=run_time
        )


class BiseRecapAnimation2(man.Scene):
    def construct(self):
        texs = [
            TemplateMathTex(r"\forall I \in \mathcal{I}(v_1^1, v_2^1) ~,~", r"\Big(", r"I \circledast W > b ", r"\Big)", "=", r"\Big(", r"X_I \oplus S", r"\Big)"),  # 0
            TemplateMathTex(r"\forall I \in \mathcal{I}(v_1^1, v_2^1) ~,~", r"\Big(", r"\bise_{\omega, b, p}(I) \geq v_2^2 ", r"\Big)", "=", r"\Big(", r"X_I \oplus S", r"\Big)"),  # 1
            TemplateMathTex(r"I "),  # 2
            TemplateMathTex(r"\bise_{\omega, b, p}"),  # 3
            TemplateMathTex(r"\bise_{\omega, b, p}(I)"),  # 4
            TemplateMathTex(r"X_I"),  # 5
            TemplateMathTex(r"\cdot \oplus S"),  # 6
            TemplateMathTex(r"X_{\bise_{\omega, b, p}(I)}"),  # 7
            TemplateTex(r"or"),  # 8
            TemplateMathTex(r"\Leftrightarrow"),  # 9
            TemplateMathTex(r"\longrightarrow"),  # 10
            TemplateMathTex(r"\longrightarrow"),  # 11
            TemplateMathTex(r"\longrightarrow"),  # 12
            TemplateMathTex(r"\longrightarrow"),  # 13
            TemplateMathTex(r"\forall I \in \mathcal{I}(v_1^1, v_2^1) ~,~", r"\Big(", r"\bise_{\omega, b, p}(I) > 0.5 ", r"\Big)", "=", r"\Big(", r"X_I \oplus S", r"\Big)"),  # 14
        ]

        tex0 = TemplateTex(r"The Bise is activated and $\bise_{\omega, b, p}(I)\in \mathcal{I}(v_1^2, v_2^2)$").shift(3 * man.UP)
        bise_mob = man.Circle(radius=.5, color=man.GREEN, fill_opacity=.5)
        texs[3].set_font_size(man.DEFAULT_FONT_SIZE*.5).add_updater(lambda x: x.move_to(bise_mob))
        self.play(man.FadeIn(tex0))
        self.play(man.FadeIn(texs[0]))
        self.wait(5)
        self.play(man.TransformMatchingTex(texs[0], texs[14].move_to(texs[0])))
        self.wait(5)
        self.play(man.TransformMatchingTex(texs[14], texs[1].move_to(texs[0])))
        self.wait()

        self.play(texs[1].animate.next_to(tex0, man.DOWN))

        texs[10].next_to(texs[2], 2*man.RIGHT)
        bise_mob.next_to(texs[10], 2*man.RIGHT)
        texs[11].next_to(bise_mob, 2*man.RIGHT)
        texs[4].next_to(texs[11], 2*man.RIGHT)
        man.VGroup(*texs[2:5], *texs[10:12], bise_mob).move_to(man.ORIGIN)
        self.play(man.FadeIn(*texs[2:5], *texs[10:12], bise_mob))

        texs[9].next_to(man.VGroup(*texs[2:5], *texs[10:12], bise_mob), man.DOWN)
        self.play(man.FadeIn(texs[9]))
        # texs[5].next_to(texs[19], man.DOWN)
        texs[12].next_to(texs[5], 2*man.RIGHT)
        texs[6].next_to(texs[12], 2*man.RIGHT)
        texs[13].next_to(texs[6], 2*man.RIGHT)
        texs[7].next_to(texs[13], 2*man.RIGHT)
        man.VGroup(*texs[5:8], *texs[12:14]).next_to(texs[9], man.DOWN)
        self.play(man.FadeIn(*texs[5:8], *texs[12:14]))

        self.wait(18)


# class BiseRecapAnimation(man.Scene):
#     def construct(self):
#         texs = [
            # TemplateMathTex(r"\forall I \in [0, 1]^{\mathbb{Z^d}} ~,~ \bise_{\omega, b, p} = "
            #                 r"\xi\Big(p(I \circledast \xi(\omega) - b)\Big) ~,~ \omega \in \mathbb{R}^{\Omega} ~,~ b, p \in \mathbb{R}"),  # 0
#             TemplateTex("Is the BiSE activated ?"),  # 1
#             TemplateTex(r"The activation depends on the input. For an almost binary input $I \in \mathcal{I}(v_1, v_2)$, being activated means", font_size=man.DEFAULT_FONT_SIZE*.7),  # 2
#             TemplateMathTex(r"\exists S \subset \mathbb{Z}^d"),  # 3
#             TemplateMathTex(r"\sum_{i, j \in \Omega}{\xi(\omega_{i,j}})", r"-", "(1 - v_1)", r"\min_{i, j \in S}", r"\xi(\omega_{i,j})",
#                             r"\leq b", "<", "v_2", r"\sum_{i, j \in S}", r"\xi(\omega_{i,j})", font_size=man.DEFAULT_FONT_SIZE*.7),  # 4
#             TemplateMathTex(r"\sum_{i, j \notin S}", r"\xi(\omega_{i, j})", r"+ v_1\sum_{i, j \in S}{\xi(\omega_{i, j})}",
#                             r"\leq", "b", "<", r"v_2\min_{i, j \in S}", r"\xi(\omega_{i, j})", font_size=man.DEFAULT_FONT_SIZE*.7),  # 5
#             TemplateTex(r"We check the thresholds"),  # 6
#             TemplateMathTex(r"S_{\oplus} = \xi(\omega) > \tau{\oplus} ~,~ \tau_{\oplus} = \frac{b}{v_2}"),  # 7
#             TemplateMathTex(r"S_{\ominus} = \xi(\omega) > \tau{\ominus} ~,~ \tau_{\ominus} = \frac{\sum_{i, j \in \Omega}\xi(\omega_{i,j}) - b}{1 - v_1}"),  # 8
#             TemplateTex(r"It is activated by $S_{\oplus}$ for dilation"),  # 9
#             TemplateMathTex(r"S = W > \tau_{\oplus}"),  # 10
#             TemplateMathTex(r"\forall I \in \mathcal{I}(v_1, v_2) ~,~", r"\Big(", r"I \circledast W > b ", r"\Big)", "=", r"\Big(", r"X_I \oplus S", r"\Big)"),  # 11
#             TemplateMathTex(r"\forall I \in \mathcal{I}(v_1, v_2) ~,~", r"\Big(", r"\bise_{\omega, b, p}(I) > v_2 ", r"\Big)", "=", r"\Big(", r"X_I \oplus S", r"\Big)"),  # 12
#             TemplateMathTex(r"I "),  # 13
#             TemplateMathTex(r"\bise_{\omega, b, p}"),  # 14
#             TemplateMathTex(r"\bise_{\omega, b, p}(I)"),  # 15
#             TemplateMathTex(r"X_I"),  # 16
#             TemplateMathTex(r"\cdot \oplus S"),  # 17
#             TemplateMathTex(r"X_{\bise_{\omega, b, p}(I)}"),  # 18
#             TemplateTex(r"or"),  # 19
#             TemplateMathTex(r"\Leftrightarrow"),  # 20
#             TemplateMathTex(r"\longrightarrow"),  # 21
#             TemplateMathTex(r"\longrightarrow"),  # 22
#             TemplateMathTex(r"\longrightarrow"),  # 23
#             TemplateMathTex(r"\longrightarrow"),  # 24
#         ]

#         texs[0].shift(3*man.UP)
#         self.play(man.FadeIn(texs[0]))
#         self.play(man.FadeIn(texs[1].next_to(texs[0], 2*man.DOWN)))
#         self.play(man.TransformMatchingTex(texs[1], texs[2].next_to(texs[0], man.DOWN)))
#         texs[3].next_to(texs[2], man.DOWN)
#         texs[4].next_to(texs[3], man.DOWN)
#         texs[19].next_to(texs[4], man.DOWN)
#         texs[5].next_to(texs[19], man.DOWN)
#         self.play(man.FadeIn(texs[3], texs[4], texs[5], texs[19]))
#         self.wait()
#         self.play(man.FadeOut(texs[2], texs[3], texs[4], texs[5], texs[19]))

#         texs[7].next_to(texs[6], man.DOWN)
#         texs[8].next_to(texs[7], man.DOWN)
#         self.play(man.FadeIn(*texs[6:9]))

#         self.wait()
#         self.play(man.FadeOut(*texs[6:9]))

#         self.play(man.FadeIn(texs[9], texs[10].next_to(texs[9], man.DOWN)))
#         self.wait()
#         self.play(man.FadeIn(texs[11].next_to(texs[10], man.DOWN)))
#         self.wait()
#         self.play(man.TransformMatchingTex(texs[11], texs[12].move_to(texs[11])))
#         self.wait()

#         self.play(man.FadeOut(texs[9], texs[10]), texs[12].animate.shift(3*man.UP))

#         texs[21].next_to(texs[13], 2*man.RIGHT)
#         texs[14].next_to(texs[21], 2*man.RIGHT)
#         texs[22].next_to(texs[14], 2*man.RIGHT)
#         texs[15].next_to(texs[22], 2*man.RIGHT)
#         man.VGroup(*texs[13:16], *texs[21:23]).move_to(man.ORIGIN)
#         self.play(man.FadeIn(*texs[13:16], *texs[21:23]))

#         texs[20].next_to(man.VGroup(*texs[13:16], *texs[21:23]), man.DOWN)
#         self.play(man.FadeIn(texs[20]))
#         # texs[16].next_to(texs[19], man.DOWN)
#         texs[23].next_to(texs[16], 2*man.RIGHT)
#         texs[17].next_to(texs[23], 2*man.RIGHT)
#         texs[24].next_to(texs[17], 2*man.RIGHT)
#         texs[18].next_to(texs[24], 2*man.RIGHT)
#         man.VGroup(*texs[16:19], *texs[23:25]).next_to(texs[20], man.DOWN)
#         self.play(man.FadeIn(*texs[16:19], *texs[23:25]))

#         self.wait(3)


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
