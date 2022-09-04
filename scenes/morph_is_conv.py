from functools import partial

import numpy as np
import manim as man
from skimage.morphology import dilation
from scipy.signal import convolve2d

from example_array import example2
from tex.latex_templates import latex_template
from mobjects import ArrayImage
from run_times import tanh_run_times
from utils import TemplateMathTex





class ConvMorphoAnimation(man.Scene):
    def construct(self):
        array = example2

        selem = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        Xs, Ys = np.where(selem.astype(bool))
        Xs = Xs - selem.shape[0] // 2
        Ys = Ys - selem.shape[1] // 2

        dil_array = dilation(array, selem)
        conv_array = convolve2d(array, selem, mode='same')

        texs = [
            TemplateMathTex(r"X"),  # 0
            TemplateMathTex(r"\mathbbm{1}_{", r"X", r"}"),  # 1
            TemplateMathTex(r"\mathbbm{1}_{", r"X", r"}", r"\circledast", r"\mathbbm{1}_{", r"S", r"}"),  # 2
            TemplateMathTex(r"\mathbbm{1}_{", r"X", r"}", r"\circledast", r"\mathbbm{1}_{", r"S", r"}", r"\geq 1"),  # 3
            TemplateMathTex(r"X", r"\oplus", r"S"),  # 4
            TemplateMathTex(r"\mathbbm{1}_{", r"X", r"\oplus", r"S", r"}"),  # 5
            TemplateMathTex("S"),  # 6
            TemplateMathTex(r"\mathbbm{1}_{", "S", "}"),  # 7
            TemplateMathTex(r'\circledast'),  # 8
            TemplateMathTex(r'\oplus'),  # 9
            TemplateMathTex("X", r'\oplus', "S"),  # 10
            TemplateMathTex("X", r'\oplus', "S", "=", r"\mathbbm{1}_{", "X", "}", r"\mathbbm{1}_{", "S", "}", r"\geq", "1"),  # 11
            TemplateMathTex("X", r'\ominus', "S", "=", r"\mathbbm{1}_{", "X", "}", r"\circledast", r"\mathbbm{1}_{", "S", "}", r"\geq", r"\card{S}"),  # 12
            TemplateMathTex("="),  # 13
        ]

        array_mob1 = ArrayImage(array, mask=array, show_value=False)
        array_mob2 = ArrayImage(array, show_value=True)
        selem_mob0 = ArrayImage(selem, mask=selem, cmap='Blues', show_value=False)
        selem_mob1 = ArrayImage(selem, cmap='Blues', show_value=True)
        
        conv_array_mob = ArrayImage(conv_array, show_value=True)
        conv_thresh_mob = ArrayImage((conv_array > 0).astype(int), mask=(conv_array > 0).astype(int), show_value=False)
        dil_array_mob1 = ArrayImage(dil_array, mask=dil_array, show_value=False)
        dil_array_mob2 = ArrayImage(dil_array, show_value=True)

        conv_array_mob.move_to(man.ORIGIN + 2*man.RIGHT + 1*man.DOWN)
        array_mob1.move_to(man.ORIGIN + 1*man.DOWN)

        # appearance of X
        self.play(man.FadeIn(array_mob1, texs[0].next_to(array_mob1, man.DOWN)))
        self.wait(8) 

        texs[9].next_to(array_mob1, man.RIGHT)
        selem_mob0.next_to(texs[9], man.RIGHT)
        texs[6].next_to(selem_mob0, man.DOWN)

        # compute the dilation
        self.play(man.FadeIn(texs[9], selem_mob0, texs[6]))  

        dil_array_mob1.next_to(array_mob1).shift(3*man.UP + 2 * man.RIGHT)
        texs[10].next_to(dil_array_mob1, man.DOWN)

        ar_tmp = array_mob1.copy()
        grp_dil = man.VGroup(ar_tmp, texs[9], selem_mob0, texs[6])
        # move the dilation up north
        self.play(man.Transform(grp_dil, dil_array_mob1), man.FadeIn(texs[10]))

        array_mob2.move_to(array_mob1)
        # see X as a binary image
        self.play(man.FadeOut(array_mob1), man.FadeIn(array_mob2), man.TransformMatchingTex(texs[0], texs[1].next_to(array_mob2, man.DOWN)))
        self.wait(2)

        # init compute convolution
        self.play(array_mob2.animate.shift(2*man.LEFT), texs[1].animate.shift(2*man.LEFT))
        arrow_mob = man.Arrow(start=array_mob2.get_right(), end=conv_array_mob.get_left())
        texs[8].next_to(array_mob2, man.LEFT)
        selem_mob1.next_to(texs[8], man.LEFT) 

        self.play(man.FadeIn(selem_mob1, texs[7].next_to(selem_mob1, man.DOWN), texs[8]), man.Create(arrow_mob))
        self.play(man.FadeIn(texs[2].next_to(conv_array_mob, man.DOWN)))
        self.wait()
        
        # compute pixel by pixel
        self.play(man.FadeOut(texs[7], texs[8]))

        conv_pixels = man.VGroup()
        run_time_fn = tanh_run_times(start_value=.5, end_value=.1, fade_start=4)
        t = 0
        for i in range(array_mob2.shape[0]):
            for j in range(array_mob2.shape[1]):
                pixel = array_mob2.get_pixel(i, j)
                self.play(selem_mob1.animate.move_to(pixel.get_center()), run_time=run_time_fn(t))

                local_pixels = [
                    array_mob2.get_pixel(
                        max(0, min(array_mob2.shape[0] - 1, i + i1)),
                        max(0, min(array_mob2.shape[1] - 1, j + j1))
                    ).copy() for i1, j1 in zip(Xs, Ys)
                ]

                grp = man.VGroup(*local_pixels)
                conv_pixels.add(grp)

                self.play(man.Transform(grp, conv_array_mob.get_pixel(i, j)), run_time=run_time_fn(t))
                t += 1

        self.play(man.FadeOut(arrow_mob, array_mob2, selem_mob1, texs[1]))
        conv_thresh_mob.move_to(conv_array_mob)
        self.wait(2)

        # apply threshold
        self.play(man.FadeOut(conv_pixels), man.FadeIn(conv_thresh_mob), man.TransformMatchingTex(texs[2], texs[3].next_to(conv_thresh_mob, man.DOWN)), run_time=2)
        self.wait(2)
        self.remove(grp_dil)

        # move dilation group to the threshold
        self.play(dil_array_mob1.animate.move_to(array_mob2), texs[10].animate.next_to(dil_array_mob1.copy().move_to(array_mob2), man.DOWN))
        self.play(man.FadeIn(texs[13].move_to(.5*(dil_array_mob1.get_center() + conv_thresh_mob.get_center()))))

        # erosion
        self.wait(3)
        self.play(man.FadeOut(conv_thresh_mob, dil_array_mob1), texs[10].animate.next_to(texs[13], man.LEFT), texs[3].animate.next_to(texs[13], man.RIGHT))
        self.play(man.VGroup(texs[13], texs[10], texs[3]).animate.move_to(man.ORIGIN + .5 * man.UP))
        self.play(man.FadeIn(texs[12].move_to(man.ORIGIN + .5 * man.DOWN)))
        self.play(texs[12].animate.set_color_by_tex(tex=r"\card{S}", color=man.GREEN), texs[3][-1][-1].animate.set_color(man.GREEN))

        self.wait(10)
