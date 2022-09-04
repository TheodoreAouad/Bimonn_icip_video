from shutil import register_unpack_format
import manim as man
import numpy as np


from bise.generate_forms import get_random_diskorect_channels, set_borders_to
from run_times import tanh_run_times


class DatasetsAnimation(man.Scene):
    def construct(self):
        self.animate_diskorect()
        self.animate_mnist()
        self.wait(3)

    def animate_mnist(self):
        mnist_init = np.load("bise/mnist_init.npy")
        mnist_up = np.load("bise/mnist_up.npy")
        mnist_up_thresh = np.load("bise/mnist_up_thresh.npy")
        mnist_inv = np.load("bise/inv_mnist_up_thresh.npy")

        title_texs = [man.Tex("MNIST"), man.Tex("inverted MNIST")]

        texs = [
            man.Tex(r"(28", r"$\times$", r"28)", font_size=man.DEFAULT_FONT_SIZE*.7),
            man.Tex(r"Upsampled", r"(50", r"$\times$", "50)", font_size=man.DEFAULT_FONT_SIZE*.7),
            man.Tex(r"Thresholded", font_size=man.DEFAULT_FONT_SIZE*.7),
        ]

        init_mob = man.ImageMobject(mnist_init)
        init_mob.height = 3
        init_mob.set_resampling_algorithm(man.RESAMPLING_ALGORITHMS["nearest"])
        texs[0].next_to(init_mob, man.UP)

        up_mob = man.ImageMobject(mnist_up)
        up_mob.height = 3
        up_mob.set_resampling_algorithm(man.RESAMPLING_ALGORITHMS["nearest"])
        texs[1].next_to(up_mob, man.UP)

        up_thresh_mob = man.ImageMobject(mnist_up_thresh*255)
        up_thresh_mob.height = 3
        up_thresh_mob.set_resampling_algorithm(man.RESAMPLING_ALGORITHMS["nearest"])
        texs[2].next_to(up_mob, man.UP)

        inv_mob = man.ImageMobject(mnist_inv*255)
        inv_mob.height = 3
        inv_mob.set_resampling_algorithm(man.RESAMPLING_ALGORITHMS["nearest"])

        self.play(man.FadeIn(title_texs[0].shift(3 * man.UP)))

        self.play(man.FadeIn(init_mob, texs[0]), run_time=.5)
        self.wait()
        self.play(man.FadeOut(init_mob), man.FadeIn(up_mob), man.TransformMatchingTex(texs[0], texs[1]), run_time=.5)
        self.wait()
        self.play(man.FadeOut(up_mob), man.FadeIn(up_thresh_mob), man.TransformMatchingTex(texs[1], texs[2]), run_time=.5)

        self.play(title_texs[0].animate.shift(man.DOWN), man.FadeOut(texs[2]), run_time=.5)
        self.play(title_texs[0].animate.set_font_size(man.DEFAULT_FONT_SIZE * .7), run_time=.5)

        self.play(man.FadeIn(title_texs[1].move_to(4 * man.RIGHT + 2 * man.UP).set_font_size(man.DEFAULT_FONT_SIZE * .7)))
        self.play(man.TransformFromCopy(up_thresh_mob, inv_mob.shift(4*man.RIGHT)))


    def animate_diskorect(self):
        generation_args = {
            'size': (50, 50, 1),
            'n_shapes': 20,
            'max_shape': (20, 20),
            'p_invert': 0,
            'n_holes': 10,
            'max_shape_holes': (10, 10),
            'noise_proba': 0.05,
            "border": (4, 4)
        }

        title_texs = [
            man.Text("Diskorect"),
        ]

        all_forms = get_random_diskorect_channels(**generation_args, return_all_results=True)[0]
        # all_forms.append(set_borders_to(1 - all_forms[-1], (4, 4), value=0))
        all_forms.append(1-all_forms[-1])

        # import matplotlib.pyplot as plt
        # plt.imshow(all_forms[-2], interpolation="NEAREST");plt.savefig("ex.png")
        all_mobs = []
        for ar in all_forms:
            cur_mob = man.ImageMobject(ar*255)
            cur_mob.height = 2
            cur_mob.set_resampling_algorithm(man.RESAMPLING_ALGORITHMS["nearest"])
            all_mobs.append(cur_mob)

        tex1 = man.Tex("Adding shapes", font_size=man.DEFAULT_FONT_SIZE*.7).shift(2*man.UP)
        tex2 = man.Tex("Adding holes", font_size=man.DEFAULT_FONT_SIZE*.7).shift(2*man.UP)
        tex3 = man.Tex("Adding noise", font_size=man.DEFAULT_FONT_SIZE*.7).shift(2*man.UP)
        tex4 = man.Tex("Invert with proba $p=0.5$", font_size=man.DEFAULT_FONT_SIZE*.7).shift(2*man.UP)

        self.play(man.FadeIn(title_texs[0].shift(3*man.UP)))
        self.play(man.FadeIn(all_mobs[0]), run_time=.5)
        self.wait(8)

        self.play(man.FadeIn(tex1))
        run_time_fn = tanh_run_times(start_value=.5, end_value=.1, fade_start=5)
        for i in range(1, generation_args["n_shapes"] + 1):
            self.play(
                man.FadeOut(all_mobs[i - 1]),
                man.FadeIn(all_mobs[i]),
            run_time=run_time_fn(i))
            # self.remove(all_mobs[i - 1])
            # self.play(man.FadeIn(all_mobs[i]), run_time=.5)
            self.wait(.1)

        self.play(man.FadeOut(tex1), man.FadeIn(tex2))

        run_time_fn = tanh_run_times(start_value=.5, end_value=.1, fade_start=3)
        for t, i in enumerate(range(generation_args["n_shapes"] + 1, len(all_forms) - 2)):
            self.play(
                man.FadeOut(all_mobs[i - 1]),
                man.FadeIn(all_mobs[i]),
            run_time=run_time_fn(t))
            # self.remove(all_mobs[i - 1])
            # self.play(man.FadeIn(all_mobs[i]), run_time=.5)
            self.wait(.1)

        self.play(man.FadeOut(tex2), man.FadeIn(tex3))
        self.play(
            man.FadeOut(all_mobs[-3]),
            man.FadeIn(all_mobs[-2]),
        run_time=1)

        self.wait()

        self.play(man.FadeOut(tex3), man.FadeIn(tex4))
        self.play(man.FadeOut(all_mobs[-2]), man.FadeIn(all_mobs[-1]), run_time=.5)
        self.play(man.FadeOut(tex4))
        self.play(all_mobs[-1].animate.shift(4*man.LEFT), title_texs[0].animate.shift(4 * man.LEFT + man.DOWN))
        self.play(title_texs[0].animate.set_font_size(man.DEFAULT_FONT_SIZE * .7))

