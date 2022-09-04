from functools import partial
import manim as man
from skimage.morphology import opening, disk

from utils import TemplateMathTex, TemplateTex
from bise.generate_forms import get_random_diskorect_channels


CurTemplateMathTex = partial(TemplateMathTex, font_size=man.DEFAULT_FONT_SIZE*.5)


class BimonnIntroAnimation(man.Scene):
    def construct(self):
        texs = [
            CurTemplateMathTex(r"\omega_1, b_1, p_1"),
            CurTemplateMathTex(r"\omega_2, b_2, p_2"),
            CurTemplateMathTex(r"\mathcal{L}(Y, Y^*)")
        ]

        bise1_mob = man.Circle(radius=.5, fill_opacity=.5, color=man.GREEN)
        bise1_label = CurTemplateMathTex(r"\bise_{\omega_1, p_1, b_1}").add_updater(lambda x: x.move_to(bise1_mob))
        grp1 = man.VGroup(*texs[:1]).add_updater(lambda x: x.next_to(bise1_mob, man.DOWN))

        bise2_mob = man.Circle(radius=.5, fill_opacity=.5, color=man.GREEN)
        bise2_label = CurTemplateMathTex(r"\bise_{\omega_2, p_2, b_2}").add_updater(lambda x: x.move_to(bise2_mob))
        grp2 = man.VGroup(*texs[1:2]).add_updater(lambda x: x.next_to(bise2_mob, man.DOWN))

        generation_args = {
            'size': (50, 50, 1),
            'n_shapes': 20,
            'max_shape': (20, 20),
            'p_invert': 0.5,
            'n_holes': 10,
            'max_shape_holes': (10, 10),
            'noise_proba': 0.02,
            "border": (4, 4)
        }

        input_array = get_random_diskorect_channels(**generation_args)[..., 0]
        target_array = opening(input_array, disk(3))

        input_mob = man.ImageMobject(input_array*255)
        input_mob.height = 2
        input_mob_label = CurTemplateMathTex(r"X").add_updater(lambda x: x.next_to(input_mob, man.DOWN))

        target_mob = man.ImageMobject(target_array * 255)
        target_mob.height = 2
        target_mob_label = CurTemplateMathTex(r"Y^*").add_updater(lambda x: x.next_to(target_mob, man.DOWN))


        pred_label = CurTemplateMathTex(r"Y^*").add_updater(lambda x: x.next_to(pred_mob_output, man.DOWN))
        pred_mob_output = man.Square(side_length=2, color=man.RED, fill_opacity=.5)

        bise1_mob.next_to(input_mob, 2*man.RIGHT)
        bise2_mob.next_to(bise1_mob, 2*man.RIGHT)
        pred_mob_output.next_to(bise2_mob, 2*man.RIGHT)
        target_mob.next_to(pred_mob_output, man.RIGHT)

        texs[2].add_updater(lambda x: x.next_to(target_mob, man.UP))

        man.Group(input_mob, bise1_mob, bise2_mob, pred_mob_output, target_mob).move_to(man.ORIGIN)

        pred_mob_input = man.Square(side_length=2, color=man.RED, fill_opacity=.5).move_to(input_mob)
        pred_mob_bise1 = man.Circle(radius=.5, fill_opacity=.5, color=man.RED).move_to(bise1_mob)
        pred_mob_bise2 = man.Circle(radius=.5, fill_opacity=.5, color=man.RED).move_to(bise2_mob)
        square_output = man.Square(side_length=2, color=man.BLUE).move_to(target_mob)

        self.play(man.FadeIn(TemplateTex("Binary Morphological Neural Network (BiMoNN)").shift(3*man.UP)))
        self.play(man.FadeIn(bise1_mob, bise1_label, grp1))
        self.play(man.FadeIn(bise2_mob, bise2_label, grp2))
        self.play(man.FadeIn(input_mob, input_mob_label))
        self.play(man.FadeIn(pred_mob_input))
        self.play(man.ReplacementTransform(pred_mob_input, pred_mob_bise1))
        self.play(man.ReplacementTransform(pred_mob_bise1, pred_mob_bise2))
        self.play(man.ReplacementTransform(pred_mob_bise2, pred_mob_output), man.FadeIn(pred_label))

        self.wait(2)
        self.play(man.FadeIn(target_mob, target_mob_label))

        grp_temp = man.VGroup(pred_mob_output, square_output)
        self.play(man.ReplacementTransform(grp_temp, texs[2]), man.FadeOut(pred_label))

        self.play(man.ReplacementTransform(texs[2].copy(), pred_mob_bise1))
        self.play(man.TransformFromCopy(pred_mob_bise1, texs[1].copy()), texs[1].animate.set_font_size(man.DEFAULT_FONT_SIZE))
        self.play(texs[1].animate.set_font_size(man.DEFAULT_FONT_SIZE*.5))

        self.play(man.ReplacementTransform(pred_mob_bise1, pred_mob_input))
        self.play(man.TransformFromCopy(pred_mob_input, texs[0].copy()), texs[0].animate.set_font_size(man.DEFAULT_FONT_SIZE))
        self.play(texs[0].animate.set_font_size(man.DEFAULT_FONT_SIZE*.5))

        self.play(man.FadeOut(pred_mob_input))
        self.wait(3)
