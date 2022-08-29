from functools import partial

import numpy as np
import manim as man
from skimage.morphology import dilation
from scipy.signal import convolve2d

from example_array import example2
from tex.latex_templates import latex_template
from mobjects import ArrayImage


TemplateTex = partial(man.MathTex, tex_template=latex_template)



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
            TemplateTex(r"X"),
            TemplateTex(r"\mathbbm{1}_{", r"X", r"}"),
            # TemplateTex(r"\indicator{X}"),
            # TemplateTex(r"\conv{\indicator{X}}{\indicator{S}}"),
            TemplateTex(r"\mathbbm{1}_{", r"X", r"}", r"\circledast", r"\mathbbm{1}_{", r"S", r"}"),
            TemplateTex(r"\mathbbm{1}_{", r"X", r"}", r"\circledast", r"\mathbbm{1}_{", r"S", r"}", r"\geq 1"),
            # TemplateTex(r"\conv{\indicator{X}}{\indicator{S}}", r"\geq 0"),
            TemplateTex(r"X", r"\oplus", r"S"),
            # TemplateTex(r"\dil{X}{S}"),
            TemplateTex(r"\mathbbm{1}_{", r"X", r"\oplus", r"S", r"}"),
            # TemplateTex(r"\indicator{\dil{X}{S}}"),
        ]

        array_mob1 = ArrayImage(array, mask=array, show_value=False)
        array_mob2 = ArrayImage(array, show_value=True)
        selem_mob = ArrayImage(selem, mask=selem, cmap='Blues', show_value=False)
        conv_array_mob = ArrayImage(conv_array, show_value=True)
        conv_thresh_mob = ArrayImage((conv_array > 0).astype(int), show_value=True)
        dil_array_mob1 = ArrayImage(dil_array, mask=dil_array, show_value=False)
        dil_array_mob2 = ArrayImage(dil_array, show_value=True)

        conv_array_mob.move_to(man.ORIGIN + 2*man.RIGHT)

        self.play(man.FadeIn(array_mob1, texs[0].next_to(array_mob1, man.DOWN)))
        self.play(man.FadeOut(array_mob1), man.FadeIn(array_mob2), man.TransformMatchingTex(texs[0], texs[1].next_to(array_mob2, man.DOWN)))

        self.play(array_mob2.animate.shift(2*man.LEFT), texs[1].animate.shift(2*man.LEFT))
        arrow_mob = man.Arrow(start=array_mob2.get_right(), end=conv_array_mob.get_left())

        self.play(man.FadeIn(selem_mob.next_to(array_mob2, man.LEFT)), man.Create(arrow_mob), man.FadeIn(texs[2].next_to(conv_array_mob, man.DOWN)))

        conv_pixels = man.VGroup()
        for i in range(array_mob2.shape[0]):
            for j in range(array_mob2.shape[1]):
                pixel = array_mob2.get_pixel(i, j)
                self.play(selem_mob.animate.move_to(pixel.get_center()), run_time=.3)

                local_pixels = [
                    array_mob2.get_pixel(
                        max(0, min(array_mob2.shape[0] - 1, i + i1)),
                        max(0, min(array_mob2.shape[1] - 1, j + j1))
                    ).copy() for i1, j1 in zip(Xs, Ys)
                ]

                grp = man.VGroup(*local_pixels)
                conv_pixels.add(grp)

                self.play(man.Transform(grp, conv_array_mob.get_pixel(i, j)), run_time=.5)

        self.play(man.FadeOut(arrow_mob), man.FadeOut(array_mob2), man.FadeOut(selem_mob), man.FadeIn(array_mob1.move_to(array_mob2)), man.TransformMatchingTex(texs[1], texs[0].next_to(array_mob2, man.DOWN)))
        # self.play(man.FadeOut(*conv_pixels))


        dil_array_mob1.move_to(array_mob1)
        dil_array_mob2.move_to(array_mob1)
        self.play(man.FadeOut(array_mob1), man.FadeIn(dil_array_mob1), man.TransformMatchingTex(texs[0], texs[4].next_to(dil_array_mob1, man.DOWN)))
        # self.play(man.FadeOut(dil_array_mob1), man.FadeIn(dil_array_mob2), man.TransformMatchingTex(texs[4], texs[5].next_to(dil_array_mob2, man.DOWN)))
        conv_thresh_mob.move_to(conv_array_mob)
        self.play(man.FadeOut(conv_pixels), man.FadeIn(conv_thresh_mob), man.TransformMatchingTex(texs[2], texs[3].next_to(conv_thresh_mob, man.DOWN)))
        self.play(man.FadeIn(man.MathTex("=").move_to(.5*(dil_array_mob1.get_center() + conv_thresh_mob.get_center()))))

        self.wait(3)
