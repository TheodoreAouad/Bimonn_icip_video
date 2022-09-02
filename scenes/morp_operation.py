from dataclasses import dataclass
from typing import Callable, List
from collections import OrderedDict
import numpy as np
import manim as man
from skimage.morphology import disk, dilation, erosion, opening, closing

from mobjects.array_image import ArrayImage, Pixel
from example_array import example1
from utils import sort_by_polar, play_all_submob, reverse_crop
from spa.cropper import get_roi
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
        random_shape = np.load("shape.npy")
        random_shape = reverse_crop(cv2.resize(random_shape.astype(np.uint8), (50, 50)).astype(int), size=(60, 60))
        dil_array = dilation(random_shape, selem=disk(2))
        ero_array = erosion(random_shape, selem=disk(2))
        ope_array = opening(random_shape, selem=disk(7))
        clo_array = closing(random_shape, selem=disk(2))

        array_mob = ArrayImage(random_shape, mask=random_shape, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1)
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


class MorpSpAAnimation(man.Scene):
    def construct(self):

        # SPA Handling
        array_spa = np.load('spa/01-33-dl_pred_target3d.npy')[..., 6].astype(int)

        roi, all_results = get_roi(array_spa, dilation_check=.01, inter_dilation_prop=.08, closing_size=5, iliac_dilation_size=7, delete_wings_args={
            'prop_radius': .75,
        },)
        res_mobs = OrderedDict()

        resolution = (100, 100)
        shape_target = 5*man.UP + 5*man.RIGHT

        def resize_fn(ar):
            return cv2.resize(ar.astype(np.uint8), resolution, interpolation=cv2.INTER_NEAREST).astype(int)

        array_spa = resize_fn(array_spa)
        roi = resize_fn(roi)

        array_spa_mob = ArrayImage(array_spa, mask=(array_spa != 0), show_value=False, shape_target=shape_target, fill_opacity=1).shift(3 * man.LEFT)

        for key, value in all_results.items():
            step_array = np.zeros_like(array_spa)
            for ar in value:
                step_array += resize_fn(ar)
            res_mobs[key] = ArrayImage(step_array, mask=step_array != 0, show_value=False, shape_target=shape_target).shift(3 * man.LEFT)


        # random shape handling
        random_shape = np.load("shape.npy")
        random_shape = reverse_crop(cv2.resize(random_shape.astype(np.uint8), (50, 50)).astype(int), size=(60, 60))
        dil_array = dilation(random_shape, selem=disk(2))
        ero_array = erosion(random_shape, selem=disk(2))
        ope_array = opening(random_shape, selem=disk(7))
        clo_array = closing(random_shape, selem=disk(2))

        random_mob = ArrayImage(random_shape, mask=random_shape, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1).shift(3 * man.RIGHT)
        morp_mobs = [
            ArrayImage(dil_array, mask=dil_array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1).shift(3 * man.RIGHT),
            ArrayImage(ero_array, mask=ero_array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1).shift(3 * man.RIGHT),
            ArrayImage(ope_array, mask=ope_array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1).shift(3 * man.RIGHT),
            ArrayImage(clo_array, mask=clo_array, show_value=False, shape_target=(5*man.UP + 5*man.RIGHT), draw_grid=True).set_opacity(1).shift(3 * man.RIGHT),
        ]

        self.play(man.Create(man.Text("Shape Analysis").move_to(man.ORIGIN + 3 * man.UP))) # 1s
        self.play(man.FadeIn(array_spa_mob, random_mob)) # 2s

        keys = list(res_mobs.keys())
        self.play(
            man.FadeOut(array_spa_mob, random_mob),
            man.FadeIn(res_mobs[keys[0]], morp_mobs[0]),
            # man.FadeIn(keys_mob[0].move_to(man.ORIGIN + 3 * man.UP), ),
        ) # 3s
        self.wait(4)
        for i, key in enumerate(keys[1:], start=1):
            self.play(
                man.FadeOut(res_mobs[keys[i - 1]], morp_mobs[i - 1]),
                man.FadeIn(res_mobs[keys[i]], morp_mobs[i]),
            )
            self.wait(4)


        self.wait(3)
