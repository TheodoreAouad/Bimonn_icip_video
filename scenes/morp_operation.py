from dataclasses import dataclass
from typing import Callable, List
import numpy as np
import manim as man
from skimage.morphology import disk, dilation, erosion, opening, closing

from mobjects.array_image import ArrayImage, Pixel
from example_array import example1
from utils import sort_by_polar, play_all_submob, reverse_crop
import cv2


@dataclass
class AnimateMorpopOutput:
    array: np.ndarray
    selem: np.ndarray
    operation: Callable
    origin: List
    array_mob: ArrayImage
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

    array_mob = ArrayImage(array, mask=array, show_value=False)
    translation = origin - array_mob.get_center()
    array_mob.shift(translation)

    selem_image = ArrayImage(selem, mask=selem, show_value=False, cmap='Blues').next_to(array_mob, man.RIGHT)
    morp_image = ArrayImage(morp_array, mask=morp_array, show_value=False).move_to(array_mob)
    diff_morp_image = ArrayImage(diff_morp, mask=diff_morp, show_value=False, cmap='Reds').move_to(array_mob)

    play_all_submob(scene, array_mob)
    play_all_submob(scene, selem_image)

    pixel_group = man.VGroup(array_mob, diff_morp_image)
    for x, y in zip(Xs, Ys):
        scale_x = array_mob.vertical_stretch
        scale_y = array_mob.horizontal_stretch

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
        array_mob=array_mob,
        selem_image=selem_image,
        morp_image=morp_image,
        diff_morp_image=diff_morp_image,
        pixel_group=pixel_group,
    )


class MorpAnimation(man.Scene):
    def construct(self):

        output = animate_morpop(self, array=example1, operation=dilation, selem=disk(1), origin=man.ORIGIN)
        # output = animate_morpop(self, array=example1, operation=dilation, selem=disk(1), origin=man.ORIGIN)

        array_mob = ArrayImage(output.array, mask=output.array, show_value=False).move_to(man.ORIGIN)
        self.play(
            output.morp_image.animate.shift(2*man.RIGHT),
            man.Create(array_mob.shift(2*man.LEFT)),
        )
        self.play(man.Create(man.Arrow(start=array_mob.get_right(), end=output.morp_image.get_left())))
        self.wait(3)


class MorpImageAnimation(man.Scene):
    def construct(self):
        array = np.load("shape.npy")
        array = reverse_crop(cv2.resize(array.astype(np.uint8), (50, 50)).astype(int), size=(60, 60))
        dil_array = dilation(array, selem=disk(2))
        ero_array = erosion(array, selem=disk(2))
        ope_array = opening(array, selem=disk(7))
        clo_array = closing(array, selem=disk(2))

        array_mob = ArrayImage(array, mask=array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1)
        dil_mob = ArrayImage(dil_array, mask=dil_array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1)
        ero_mob = ArrayImage(ero_array, mask=ero_array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1)
        ope_mob = ArrayImage(ope_array, mask=ope_array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1)
        clo_mob = ArrayImage(clo_array, mask=clo_array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1)


        self.play(man.FadeIn(array_mob))
        self.play(man.FadeOut(array_mob), man.FadeIn(dil_mob))
        self.play(man.FadeOut(dil_mob), man.FadeIn(ero_mob))
        self.play(man.FadeOut(ero_mob), man.FadeIn(ope_mob))
        self.play(man.FadeOut(ope_mob), man.FadeIn(clo_mob))
        self.wait(3)
