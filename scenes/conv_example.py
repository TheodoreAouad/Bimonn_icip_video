import manim as man
import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from mobjects import ArrayImage


class ConvExampleAnimation(man.Scene):
    def construct(self):
        img = np.load("einstein.npy")
        selem1 = np.array([[-1, 1]])
        selem2 = np.array([[-1], [1]])

        selem = np.ones((9, 9))

        # shape = (100, 100)
        # shape = (20, 20)

        # img = cv2.resize(img, shape).astype(float)


        conv1 = convolve2d(img, selem1, mode="same")
        conv2 = convolve2d(img, selem2, mode="same")

        conv_tot = np.sqrt(conv1 ** 2 + conv2 ** 2)

        img_noisy = img + np.random.randn(*img.shape) * img.max() / 20
        sigma = .6
        # gaussian_fn = lambda x: 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-x**2/(2*sigma**2))
        # XX, YY = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))
        # gaussian_filter = gaussian_fn(XX, YY)
        # conv_denoise = convolve2d(img_noisy, gaussian_filter, mode="same")
        conv_denoise = gaussian_filter(img_noisy, sigma=sigma)
        # plt.imshow(img_noisy, cmap='gray'); plt.savefig("ex1.png")
        # plt.imshow(conv_denoise, cmap='gray'); plt.savefig("ex2.png")
        # assert False

        # img_mob = ArrayImage(img, show_value=False, shape_target=5*man.UP + 5*man.RIGHT,)
        # img_mob = ArrayImage(img, show_value=False, fill_opacity=1, shape_target=3*man.UP + 3*man.RIGHT, cmap='gray')#.move_to(man.ORIGIN + 3 * man.LEFT)
        # conv_mob = ArrayImage(conv_tot, show_value=False, fill_opacity=1, shape_target=3*man.UP + 3*man.RIGHT, cmap='gray')#.move_to(man.ORIGIN + 3 * man.RIGHT)
        img_mob = man.ImageMobject(img)
        img_mob.height = 3
        conv_mob = man.ImageMobject(conv_tot)
        conv_mob.height = 3
        # selem_mob = ArrayImage(selem, show_value=False, cmap='Blues', shape_target=.3*man.UP + .3*man.RIGHT)
        selem_mob = man.Square(side_length=img_mob.height / 20, color=man.BLUE, fill_opacity=.5)


        # img_noise_mob = ArrayImage(img_noisy, show_value=False, fill_opacity=1, shape_target=3*man.UP + 3*man.RIGHT, cmap='gray')#.move_to(man.ORIGIN + 3 * man.LEFT)
        # conv_noise_mob = ArrayImage(conv_denoise, show_value=False, fill_opacity=1, shape_target=3*man.UP + 3*man.RIGHT, cmap='gray')#.move_to(man.ORIGIN + 3 * man.RIGHT)

        img_noise_mob = man.ImageMobject(img_noisy)
        img_noise_mob.height = 3
        conv_noise_mob = man.ImageMobject(conv_denoise)
        conv_noise_mob.height = 3

        self.play(man.FadeIn(img_mob))
        self.play(man.FadeIn(selem_mob.next_to(img_noise_mob, man.LEFT)))  # 2s
        for j in [0, 1]:
            # for i in range(4, shape[0] - 4, 5):
            for i in range(int(img_mob.height / selem_mob.height)):
                self.play(selem_mob.animate.move_to(img_noise_mob.get_top() + img_noise_mob.get_left() + selem_mob.height * ((1 / 2 + j) * man.DOWN + (1 / 2 + i) * man.RIGHT)), run_time=.25)
        self.play(man.FadeOut(selem_mob))
        # ~12s

        self.play(man.FadeIn(img_noise_mob), man.FadeOut(img_mob))  # 1s
        self.wait()

        self.play(man.FadeIn(conv_noise_mob))  # 13s
        self.play(img_noise_mob.animate.shift(2 * man.UP + 2 * man.LEFT), conv_noise_mob.animate.shift(2 * man.UP + 2 * man.RIGHT))  # 14s

        self.play(man.FadeIn(img_mob.shift(2 * man.DOWN)))  # 15s
        self.play(man.FadeIn(conv_mob.shift(2 * man.DOWN)))  # 16s
        self.play(img_mob.animate.shift(2 * man.LEFT), conv_mob.animate.shift(2 * man.RIGHT))  # 17s

        self.wait(5)
