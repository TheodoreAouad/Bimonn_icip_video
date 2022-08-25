import numpy as np
import manim as man
from skimage.morphology import disk, dilation

from mobjects.array_image import ArrayImage, Pixel
from example_array import example1
from utils import sort_by_polar, play_all_submob


class DilationAnimation(man.Scene):
    def construct(self):

        array = example1
        selem = disk(1)
        mask = example1 != 0

        dil_image = dilation(array, selem)
        diff_dil = dil_image - array
        Xs, Ys = sort_by_polar(*np.where(diff_dil))

        array_image = ArrayImage(array, mask=mask, show_value=False)

        translation = man.ORIGIN - array_image.get_center()
        array_image.shift(translation)

        selem_image = ArrayImage(selem, mask=selem, show_value=False, cmap='Blues').next_to(array_image, man.RIGHT)
        dil_image = ArrayImage(dil_image, mask=dil_image, show_value=False).move_to(array_image)
        diff_dil_image = ArrayImage(diff_dil, mask=diff_dil, show_value=False, cmap='Reds').move_to(array_image)

        play_all_submob(self, array_image)
        play_all_submob(self, selem_image)

        pixel_group = man.VGroup(array_image, diff_dil_image)
        for x, y in zip(Xs, Ys):
            scale_x = array_image.vertical_stretch
            scale_y = array_image.horizontal_stretch

            pos = y * scale_y * man.RIGHT + x * scale_x * man.DOWN + translation
            self.play(selem_image.animate.move_to(pos))
            pixel = Pixel(1, color=man.BLUE, show_value=False, width=scale_x, height=scale_y).move_to(pos)
            self.add(pixel)
            pixel_group.add(pixel)

        self.play(man.FadeOut(selem_image))
        self.play(man.FadeIn(dil_image), man.FadeOut(pixel_group))

        array_image = ArrayImage(array, mask=mask, show_value=False).move_to(man.ORIGIN)
        self.play(
            dil_image.animate.shift(2*man.RIGHT),
            man.Create(array_image.shift(2*man.LEFT)),
        )
        self.play(man.Create(man.Arrow(start=array_image.get_right(), end=dil_image.get_left())))
        self.wait(3)
