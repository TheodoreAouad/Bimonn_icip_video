import random
from functools import partial


import numpy as np
import manim as man
from skimage.morphology import dilation
from scipy.signal import convolve2d

from mobjects.array_image import Pixel
from tex.latex_templates import latex_template
from mobjects import ArrayImage, DilationOperationMob, ConvolutionOperationMob
from utils import play_horizontal_sequence, play_transforming_tex, euclidean_division, animation_update_array_mob
from example_array import example1


TemplateTex = partial(man.MathTex, tex_template=latex_template)


class OnePixelEnoughAnimation(man.Scene):
    def construct(self):
        array = example1

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        Xs, Ys = np.where(selem.astype(bool))
        Xs = Xs - selem.shape[0] // 2
        Ys = Ys - selem.shape[1] // 2

        coords = (2, 1)

        dil_array = dilation(array, selem)
        conv_array = convolve2d(array, selem, mode='same')
        unknown = np.array([["?" for _ in range(array.shape[1])] for _ in range(array.shape[0])])

        texs = [
            man.MathTex("X"),
            man.MathTex("S"),
            man.MathTex(r"X", r"\circledast", r"S"),
            man.MathTex("[2, 1]", "=", tex_to_color_map={"[2, 1]": man.RED}),
            man.MathTex("="),
            # man.MathTex("(", r"X", r"\circledast", r"S", ")", "(2, 1)", "=", tex_to_color_map={"(2, 1)": man.RED}),
        ]

        array_mob = ArrayImage(array, show_value=True).move_to(man.ORIGIN + 2*man.LEFT)
        conv_mob = ArrayImage(conv_array, show_value=True)
        selem_mob = ArrayImage(selem, show_value=True, cmap='Blues')
        unknown_mob = ArrayImage(unknown).next_to(array_mob, 4*man.RIGHT)

        conv_op_mob = ConvolutionOperationMob(array_mob, selem_mob, unknown_mob, subscripts=(texs[0], texs[1], texs[2]), show_braces=True).move_to(man.ORIGIN + man.DOWN)

        center_pixel = unknown_mob.get_pixel(*coords)
        conv_pixel = conv_mob.get_pixel(*coords)

        group_pixel = man.VGroup(*[array_mob.get_pixel(*(coords - np.array([i, j]))).copy() for i in range(-1, 2) for j in range(-1, 2)])


        cadre_input = Pixel(value=None, show_value=False, color=man.RED, width=array_mob.hscale, height=array_mob.vscale).move_to(array_mob.get_pixel(*coords).get_center())
        cadre_conv = Pixel(value=None, show_value=False, color=man.RED, width=array_mob.hscale, height=array_mob.vscale).move_to(center_pixel.get_center())

        conv_op_equal = conv_op_mob.submobjects[5]
        conv_op_circledast = conv_op_mob.submobjects[1]
        conv_op_brace2 = conv_op_mob.submobjects[4]
        conv_op_brace1 = conv_op_mob.submobjects[3]

        # self.play(man.FadeIn(array_mob), man.Create(texs[0].next_to(array_mob, man.DOWN)))
        # self.play(man.FadeIn(selem_mob), man.Create(texs[1].next_to(selem_mob, man.DOWN)))
        self.play(man.Create(conv_op_mob))
        self.wait(1.5)
        self.play(man.Create(cadre_conv), man.Create(cadre_input))
        self.wait(1.5)
        self.play(man.FadeOut(unknown_mob, cadre_conv, conv_op_mob.subscripts[-1]), man.FadeIn(center_pixel))
        texs[3].next_to(conv_op_brace2, man.RIGHT)
        self.play(man.Create(texs[3]), center_pixel.animate.next_to(texs[3], man.RIGHT), man.FadeOut(conv_op_equal))
        self.wait(1.5)
        selem_mob2 = selem_mob.copy()
        self.play(man.FadeOut(texs[1]), group_pixel.animate.next_to(array_mob, man.UP), selem_mob2.animate.next_to(group_pixel.copy().next_to(array_mob, man.UP), man.RIGHT))
        self.wait(1.5)

        grp_conv_pixel = man.VGroup(group_pixel, selem_mob2)
        self.play(man.Transform(grp_conv_pixel, conv_pixel.next_to(texs[3], man.RIGHT)), man.FadeOut(center_pixel))

        self.wait(1.5)

        group_pixel = man.VGroup(*[array_mob.get_pixel(*(coords - np.array([i, j]))).copy() for i in range(-1, 2) for j in range(-1, 2)])
        sq = man.Square(side_length=selem.shape[0] * selem_mob.horizontal_stretch, color=man.RED).move_to(group_pixel)

        self.play(man.Create(sq), man.FadeOut(cadre_input))
        self.wait(1)

        # tex_copy = texs[3].copy()
        # self.remove(texs[3])
        # self.add(tex_copy)
        texs[4].next_to(selem_mob, man.RIGHT)
        self.play(
            man.FadeOut(array_mob, sq, conv_op_brace1, conv_op_brace2, texs[0]),
            group_pixel.animate.next_to(conv_op_circledast, man.LEFT),
            man.TransformMatchingTex(texs[3], texs[4]),
            grp_conv_pixel.animate.next_to(texs[4], man.RIGHT)
        )

        selem_mob2 = selem_mob.copy()
        group_pixel2 = group_pixel.copy()

        self.play(man.FadeOut(man.VGroup(group_pixel2, selem_mob2), target_position=grp_conv_pixel, scale=0))

        self.wait(3)


class BiseEqQuestionAnimation(man.Scene):
    def construct(self):
        eq_texts = [
            TemplateTex(r"\dil{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"\indicator{S}", r"\geq", r"1", r"\Big)",),
            TemplateTex(r"\dil{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"W", r"\geq", r"b", r"\Big)",),
            TemplateTex(r"\dil{X}{S}", r"=", r"\Big(", r"\indicator{X}", r"\circledast", r"W", r"\geq", r"b", r"\Big)", r"~,~ W \in [0, 1]^{\Omega}"),
        ]

        comment_texts = [
            man.Tex(r"What is the relationship between $W$ and $b$ ?")
        ]


        self.play(man.Create(eq_texts[0]))
        self.wait()
        self.play(man.TransformMatchingTex(eq_texts[0], eq_texts[1]))
        self.play(man.TransformMatchingTex(eq_texts[1], eq_texts[2]))
        self.play(man.Create(comment_texts[0].next_to(eq_texts[1], man.DOWN)))
        self.wait(3)


class BiseEq1DerivationAnimation(man.Scene):
    def construct(self):
        ex1 = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])

        dil_ex1 = dilation(ex1, selem)

        W = np.array([
            ["$w_{" + f"{i}{j}" + "}$" for i in range(3)] for j in range(3)
        ])

        ex1_mob1 = ArrayImage(ex1, show_value=True)
        sq1 = man.Square(side_length=ex1_mob1.horizontal_stretch, color='red').move_to(ex1_mob1.get_center())
        grp1 = man.VGroup(ex1_mob1, sq1)
        selem_mob = ArrayImage(selem, mask=selem, cmap='Blues')
        dil_ex1_mob1 = Pixel(value=dil_ex1[1, 1], color=man.YELLOW, show_value=True, height=ex1_mob1.horizontal_stretch)

        ex1_mob2 = ArrayImage(ex1, show_value=True)
        sq2 = man.Square(side_length=ex1_mob2.horizontal_stretch, color='red').move_to(ex1_mob2.get_center())
        grp2 = man.VGroup(ex1_mob2, sq2)
        W_mob = ArrayImage(W, show_value=True)

        texs = [
            man.Tex(r"As~~"),
            man.Tex(r"~~, we have"),
            man.MathTex(r">", r"b", r"\Rightarrow", ),
            man.MathTex(r"w_{1, 2}", r"> b")
        ]

        dil_mob = DilationOperationMob(grp1, selem_mob, dil_ex1_mob1, show_braces=False, subscripts=[man.MathTex("X"), man.MathTex("S"), None])

        conv_mob = ConvolutionOperationMob(grp2, W_mob, show_braces=False, subscripts=[man.MathTex("X"), man.MathTex("W"), None])

        play_horizontal_sequence(self, [texs[0], dil_mob, texs[1]], origin=man.ORIGIN + 2 * man.UP + 3*man.LEFT)
        play_horizontal_sequence(self, [conv_mob, texs[2], texs[3]], origin=dil_mob.get_left() + 3*man.DOWN, aligned_edge=man.LEFT)

        for (i, j) in [(0, 1), (1, 1), (1, 2), (1, 0)]:
            self.play(man.FadeOut(grp1), man.FadeOut(grp2))

            ex1 = np.zeros((3, 3), dtype=int); ex1[i, j] = 1
            ex1_mob1.update_array(ex1)
            ex1_mob2.update_array(ex1)
            new_tex = man.MathTex(r"w_{" + f"{i}, {j}" + r"}", r"> b").move_to(texs[3].get_center())

            self.play(man.FadeIn(grp1), man.FadeIn(grp2), man.TransformMatchingTex(texs[3], new_tex))
            texs[3] = new_tex

        self.play(man.FadeOut(texs[0], dil_mob, texs[1], texs[2], conv_mob))
        self.play(man.TransformMatchingTex(texs[3], man.MathTex(r"\min_{i, j \in S}", r"w_{i,j}", r"> b").move_to(man.ORIGIN)))

        self.wait(3)


class BiseEq2DerivationAnimation(man.Scene):
    def construct(self):
        ex1 = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])

        dil_ex1 = dilation(ex1, selem)

        W = np.array([
            ["$w_{" + f"{i}{j}" + "}$" for i in range(3)] for j in range(3)
        ])

        ex1_mob1 = ArrayImage(ex1, show_value=True)
        sq1 = man.Square(side_length=ex1_mob1.horizontal_stretch, color='red').move_to(ex1_mob1.get_center())
        grp1 = man.VGroup(ex1_mob1, sq1)
        selem_mob = ArrayImage(selem, mask=selem, cmap='Blues')
        dil_ex1_mob1 = Pixel(value=dil_ex1[1, 1], color=man.PURPLE, show_value=True, height=ex1_mob1.horizontal_stretch)

        ex1_mob2 = ArrayImage(ex1, show_value=True)
        sq2 = man.Square(side_length=ex1_mob2.horizontal_stretch, color='red').move_to(ex1_mob2.get_center())
        grp2 = man.VGroup(ex1_mob2, sq2)
        W_mob = ArrayImage(W, show_value=True)

        texs = [
            man.Tex(r"As~~"),
            man.Tex(r"~~, we have"),
            man.MathTex(r"\leq", r"b", r"\Rightarrow", ),
            man.MathTex(r"w_{0, 1}", r"+", r"w_{1, 2}", r"+", r"w_{2, 1}", r"+", r"w_{1, 0}" r"\leq b")
        ]

        dil_mob = DilationOperationMob(grp1, selem_mob, dil_ex1_mob1, show_braces=False, subscripts=[man.MathTex("X"), man.MathTex("S"), None])

        conv_mob = ConvolutionOperationMob(grp2, W_mob, show_braces=False, subscripts=[man.MathTex("X"), man.MathTex("W"), None])

        play_horizontal_sequence(self, [texs[0], dil_mob, texs[1]], origin=man.ORIGIN + 2 * man.UP + 6*man.LEFT)
        play_horizontal_sequence(self, [conv_mob, texs[2], texs[3]], origin=dil_mob.get_left() + 3*man.DOWN, aligned_edge=man.LEFT)


        self.play(man.FadeOut(texs[0], dil_mob, texs[1], texs[2], conv_mob))
        self.play(man.TransformMatchingTex(texs[3], man.MathTex(r"\sum_{i, j \in S}", r"w_{i,j}", r"\leq b").move_to(man.ORIGIN)))

        self.wait(3)


class BiseConvBoundsAnimation(man.Scene):
    def construct(self):
        def generate_input(selem: np.ndarray, nb: int, outside_selem: bool = False):
            """ This function generates a binary input given a number between 0 and selem.shape.prod() - 1. We can give
            the option to choose between generating only outside the selem.
            If nb < (~selem).sum() then all 1s are outside the selem. If nb >= (~selem).sum(), at least one pixel of selem
            if 1.
            """
            nbits_in = selem.sum()
            nbits_out = np.prod(selem.shape) - nbits_in

            nb_in, nb_out = euclidean_division(nb, 2**nbits_out)

            # First we compute outside the selem
            base_2_out = format(nb_out, f"#0{nbits_out + 2}b")[2:]

            res = np.zeros(selem.shape)
            idxes = np.array([s == '1' for s in base_2_out])

            for idx, i in enumerate(zip(*np.where(1 - selem))):
                res[i] = idxes[idx]

            if outside_selem:
                return res

            # If we want to compute inside the selem
            # nb_in = max(nb_in, 1)  # We force at least one pixel positive
            base_2_in = format(nb_in, f"#0{nbits_in + 2}b")[2:]

            idxes = np.array([s == '1' for s in base_2_in])

            for idx, i in enumerate(zip(*np.where(selem))):
                res[i] = idxes[idx]

            return res

        ex1 = np.array([
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
        ])

        W = np.array([
            [0.12, 0.8, 0.13],
            [0.81, 0.83, 0.89],
            [0.1, 0.86, 0.14],
        ])

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]).astype(bool)

        lb = np.zeros(W.shape, dtype=int)
        lb[~selem] = 1

        ub = np.zeros(W.shape, dtype=int)
        W_tmp = W.copy()
        W_tmp[~selem] = np.infty
        Xs, Ys = np.where(W == W_tmp.min())
        ub[Xs[0], Ys[0]] = 1


        b = (W[selem].min() + W[~selem].sum()) / 2

        def play_conv_and_axis(conv_mob: ConvolutionOperationMob, array_mob: ArrayImage, array_mob_dil: ArrayImage, dil_array_mob: ArrayImage, new_array: np.ndarray, run_time: float = 1) -> man.Dot:
            new_dil_array = np.array([[dilation(new_array, selem)[1, 1]]])

            animation_update_array_mob(self, [array_mob, dil_array_mob, array_mob_dil], [new_array, new_dil_array, new_array], run_time=run_time)

            val = (new_array * W).sum()
            color = man.YELLOW if val > b else man.PURPLE
            point = man.Dot(color=man.WHITE).next_to(axis_mob.get_left(), aligned_edge=man.LEFT).shift(val * man.RIGHT)

            self.play(man.Transform(conv_mob.copy(), point), run_time=run_time)
            self.play(man.FadeOut(dil_array_mob.copy(), target_position=point.get_center()), point.animate.set_color(color), run_time=run_time)
            return point

        ex1 = lb

        dil_array = np.array([[dilation(ex1, selem)[1, 1]]])

        array_mob_conv = ArrayImage(ex1, show_value=True, vmin_cmap=0, vmax_cmap=1)
        array_mob_dil = ArrayImage(ex1, show_value=True, vmin_cmap=0, vmax_cmap=1)
        dil_array_mob = ArrayImage(dil_array, show_value=True, vmin_cmap=0, vmax_cmap=1)
        W_mob = ArrayImage(W, cmap=lambda x: [1, 1, 1, 1])
        selem_mob = ArrayImage(selem.astype(int), cmap='Blues', mask=selem)
        axis_mob = man.NumberLine(x_range=[0, W.sum(), .5], length=12, include_tip=True)

        # Create axis and bias
        self.play(man.Create(axis_mob.move_to(man.ORIGIN + 2*man.DOWN)))
        b_mob = man.Dot(color=man.RED).next_to(axis_mob.get_left(), aligned_edge=man.LEFT).shift(b * man.RIGHT)
        self.play(man.Create(b_mob), man.Create(man.MathTex("b", color=man.RED).next_to(b_mob, man.UP)))

        # Initialize convolution
        conv_mob = ConvolutionOperationMob(
            array_mob_conv, W_mob, show_braces=False, subscripts=[man.MathTex("X"), man.MathTex("W"), None]
        ).move_to(man.ORIGIN + 2*man.UP + 4*man.LEFT)

        # Initalize dilation
        dil_op_mob = DilationOperationMob(
            array_mob_dil, selem_mob, dil_array_mob, show_braces=False, subscripts=[man.MathTex("X"), man.MathTex("W"), None]
        ).move_to(man.ORIGIN + 2*man.UP + 4*man.RIGHT)

        self.play(man.Create(conv_mob), man.Create(dil_op_mob))

        # Compute first conv point for lower bound
        val = (ex1 * W).sum()
        color = man.YELLOW if val > b else man.PURPLE
        point = man.Dot(color=man.WHITE).next_to(axis_mob.get_left(), aligned_edge=man.LEFT).shift(val * man.RIGHT)
        self.play(man.Transform(conv_mob.copy(), point), run_time=1)
        self.play(man.FadeOut(dil_array_mob.copy(), target_position=point.get_center()), point.animate.set_color(color), run_time=1)
        self.play(man.Create(man.MathTex("L", height=.3, color=color).next_to(point, man.DOWN)), run_time=.5)

        # Compute upper bound
        point_ub = play_conv_and_axis(conv_mob, array_mob_conv, array_mob_dil, dil_array_mob, ub, run_time=.5)
        self.play(man.Create(man.MathTex("U", height=.3, color=man.YELLOW).next_to(point_ub, man.DOWN)), run_time=.5)

        # Sample of example matrices for convolution
        lim_inout = 2**(np.prod(selem.shape) - selem.sum())

        N_examples = 3
        nb_in = random.sample(range(lim_inout), N_examples)
        nb_out = random.sample(range(lim_inout, 2**np.prod(selem.shape)), N_examples)

        nbs = nb_in + nb_out
        random.shuffle(nbs)

        # Animate example matrices
        for nb in nbs:
            ex1 = generate_input(selem, nb).astype(int)
            play_conv_and_axis(conv_mob, array_mob_conv, array_mob_dil, dil_array_mob, ex1, run_time=.5)

        self.wait(3)


class BiseEq3DerivationAnimation(man.Scene):
    def construct(self):
        texs = [
            # TemplateTex(r'\ero{X}{S}')
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{",
            r"\indicator{", r"X", r"}", r"}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\ero{X}{S}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{",
            r"\indicator{", r"\bar{X}", r"}", r"}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\ero{\bar{X}}{S}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{",
            r"\indicator{", r"\bar{X}", r"}", r"}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\overline{", r"\dil{\bar{X}}{S}", r"}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{",
            r"\indicator{", r"\bar{X}", r"}", r"}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\overline{", r"\dil{\bar{X}}{S}", r"}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\inf_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\sum_{i \in \Omega}", r"w_i", r"-", r"\conv{(",
            r"\indicator{", r"{X}", r"}", r")}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\overline{", r"\dil{\bar{X}}{S}", r"}", r"\}"),
            TemplateTex(r"U_{erosion} = ", r"\sum_{i \in \Omega}", r"w_i", r"-", r"\sup_{", r"X \in \{0, 1\}, i \in \Omega", r"}", r"\{" r"\conv{(",
            r"\indicator{", r"{X}", r"}", r")}", r"{", r"W", r"}", r"(i)", r"~\|~", r"i \in", r"\overline{", r"\dil{\bar{X}}{S}", r"}", r"\}"),
            TemplateTex(r"U_{erosion} =", r"\sum_{i \in \Omega}", r"w_i", r"-", r"L_{dilation}")
        ]

        play_transforming_tex(self, texs, origin=man.ORIGIN, run_time=2)
