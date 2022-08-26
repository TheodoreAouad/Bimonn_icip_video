from dataclasses import dataclass
from turtle import color
from typing import Callable, List
import numpy as np
import manim as man
from skimage.morphology import disk, dilation, erosion, opening, closing

from mobjects.array_image import ArrayImage, Pixel
from example_array import example1
from utils import sort_by_polar, play_all_submob


@dataclass
class AnimateMorpopOutput:
    array: np.ndarray
    selem: np.ndarray
    operation: Callable
    origin: List
    array_image: ArrayImage
    selem_image: ArrayImage
    morp_image: ArrayImage
    diff_morp_image: ArrayImage
    pixel_group: man.VGroup


def animate_morpop(
        scene,
        array,
        operation,
        selem,
        origin=man.ORIGIN
        ):
    color_dict = {1: man.BLUE, -1: man.RED}

    morp_array = operation(array, selem)
    diff_morp = operation(array) - array
    Xs, Ys = sort_by_polar(*np.where(diff_morp != 0))

    array_image = ArrayImage(array, mask=array, show_value=False)
    translation = origin - array_image.get_center()
    array_image.shift(translation)

    selem_image = ArrayImage(selem, mask=selem, show_value=False, cmap='Blues').next_to(array_image, man.RIGHT)
    morp_image = ArrayImage(morp_array, mask=morp_array, show_value=False).move_to(array_image)
    diff_morp_image = ArrayImage(diff_morp, mask=diff_morp, show_value=False, cmap='Reds').move_to(array_image)

    play_all_submob(scene, array_image)
    play_all_submob(scene, selem_image)

    pixel_group = man.VGroup(array_image, diff_morp_image)
    for x, y in zip(Xs, Ys):
        scale_x = array_image.vertical_stretch
        scale_y = array_image.horizontal_stretch

        pos = y * scale_y * man.RIGHT + x * scale_x * man.DOWN + translation
        scene.play(selem_image.animate.move_to(pos))

        x, y = round(x), round(y)
        pixel = Pixel(diff_morp[x, y], color=color_dict[diff_morp[x, y]], show_value=False, width=scale_x, height=scale_y).move_to(pos)
        scene.add(pixel)
        pixel_group.add(pixel)

    scene.play(man.FadeOut(selem_image))
    scene.play(man.FadeIn(morp_image), man.FadeOut(pixel_group))

    return AnimateMorpopOutput(
        array=array,
        selem=selem,
        operation=operation,
        origin=origin,
        array_image=array_image,
        selem_image=selem_image,
        morp_image=morp_image,
        diff_morp_image=diff_morp_image,
        pixel_group=pixel_group,
    )


class MorpAnimation(man.Scene):
    def construct(self):

        output = animate_morpop(self, array=example1, operation=dilation, selem=disk(1), origin=man.ORIGIN)
        # output = animate_morpop(self, array=example1, operation=dilation, selem=disk(1), origin=man.ORIGIN)

        array_image = ArrayImage(output.array, mask=output.array, show_value=False).move_to(man.ORIGIN)
        self.play(
            output.morp_image.animate.shift(2*man.RIGHT),
            man.Create(array_image.shift(2*man.LEFT)),
        )
        self.play(man.Create(man.Arrow(start=array_image.get_right(), end=output.morp_image.get_left())))
        self.wait(3)
