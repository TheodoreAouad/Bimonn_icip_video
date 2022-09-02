from functools import partial
from typing import Callable, Tuple, List, Optional

import numpy as np
from manim.utils.color import Color
import manim as man

from tex.latex_templates import latex_template


TemplateMathTex = partial(man.MathTex, tex_template=latex_template)
TemplateTex = partial(man.Tex, tex_template=latex_template)


def polar2z(r, theta):
    out = r * np.exp(1j * theta)
    return np.real(out), np.imag(out)


def z2polar(z):
    return (np.abs(z), np.angle(z))


def get_color_from_rgb(rgb: Tuple[float], *args, **kwargs) -> Color:
    output = Color(*args, **kwargs)
    rgb = rgb[:3]
    output.red = rgb[0]
    output.green = rgb[1]
    output.blue = rgb[2]
    return output


def sort_by_polar(Xs: np.ndarray, Ys: np.ndarray) -> Tuple[np.ndarray]:
    XYs = np.vstack((Xs, Ys)).T
    mean_t = XYs.mean(0)
    XYs = XYs - mean_t

    out = np.vstack(z2polar(XYs[:, 0] + 1j*XYs[:, 1])).T
    out = out[np.lexsort((out[:, 0], out[:, 1]))]

    x, y = polar2z(out[:, 0], out[:, 1])
    return x + mean_t[0], y + mean_t[1]


def play_all_submob(scene: man.Scene, mob: man.Mobject, fn: Callable = man.Create, run_time: float = 1):
    return scene.play(*[fn(submob) for submob in mob.submobjects], run_time=run_time)


def play_horizontal_sequence(scene: man.Scene, seq: List[man.Mobject], origin: List = man.ORIGIN, aligned_edge: List = man.ORIGIN, direction: List = man.RIGHT, **kwargs):
    scene.play(man.Create(seq[0].move_to(origin, aligned_edge=aligned_edge)), **kwargs)
    for i in range(1, len(seq)):
        scene.play(man.Create(seq[i].next_to(seq[i-1], direction)), **kwargs)
    return scene


def play_transforming_tex(scene: man.Scene, texs: List[man.Tex], origin: List = man.ORIGIN, **kwargs):
    scene.play(man.Create(texs[0].move_to(origin)), **kwargs)
    for i in range(1, len(texs)):
        scene.play(man.TransformMatchingTex(texs[i-1], texs[i].move_to(origin)), **kwargs)
    return scene


def euclidean_division(x: int, y: int) -> Tuple[int]:
    return x // y, x % y


def animation_update_array_mob(scene: man.Scene, array_mobs: List["ArrayImage"], new_arrays: List[np.ndarray], run_time: float = 1, other_anims: List = []):
    if not isinstance(array_mobs, list):
        array_mobs = [array_mobs]
    if not isinstance(new_arrays, list):
        new_arrays = [new_arrays]

    scene.play(*[man.FadeOut(array_mob) for array_mob in array_mobs], run_time=run_time)
    for array_mob, new_array in zip(array_mobs, new_arrays):
        array_mob.update_array(new_array)
    scene.play(*[man.FadeIn(array_mob) for array_mob in array_mobs], *other_anims, run_time=run_time)


def reverse_crop(croped_ar: np.ndarray, size: Tuple, crop_xs: Tuple = None, crop_ys: Tuple = None, crop_zs: Optional[Tuple] = None,
                 fill_value: float = 0) -> np.ndarray:
    ar = np.zeros(size) + fill_value

    if crop_xs is None:
        crop_xs = np.array([(size[0] - croped_ar.shape[0]) / 2, -((size[0] - croped_ar.shape[0]) / 2)]).astype(int)

    if crop_ys is None:
        crop_ys = np.array([(size[1] - croped_ar.shape[1]) / 2, -((size[1] - croped_ar.shape[1]) / 2)]).astype(int)

    if len(size) == 2:
        ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1]] = croped_ar

    if len(size) == 3:
        ar[crop_xs[0]:crop_xs[1], crop_ys[0]:crop_ys[1], crop_zs[0]:crop_zs[1]] = croped_ar

    return ar.astype(croped_ar.dtype)
