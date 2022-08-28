from typing import Tuple

import numpy as np
from manim.utils.color import Color
import manim as man


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


def sort_by_polar(Xs, Ys):
    XYs = np.vstack((Xs, Ys)).T
    mean_t = XYs.mean(0)
    XYs = XYs - mean_t

    out = np.vstack(z2polar(XYs[:, 0] + 1j*XYs[:, 1])).T
    out = out[np.lexsort((out[:, 0], out[:, 1]))]

    x, y = polar2z(out[:, 0], out[:, 1])
    return x + mean_t[0], y + mean_t[1]


def play_all_submob(scene, mob, fn=man.Create, run_time=1):
    return scene.play(*[fn(submob) for submob in mob.submobjects], run_time=run_time)


def play_horizontal_sequence(scene, seq, origin=man.ORIGIN, aligned_edge=man.ORIGIN, direction=man.RIGHT, **kwargs):
    scene.play(man.Create(seq[0].move_to(origin, aligned_edge=aligned_edge)), **kwargs)
    for i in range(1, len(seq)):
        scene.play(man.Create(seq[i].next_to(seq[i-1], direction)), **kwargs)
    return scene


def play_transforming_tex(scene, texs, origin=man.ORIGIN, **kwargs):
    scene.play(man.Create(texs[0].move_to(origin)), **kwargs)
    for i in range(1, len(texs)):
        scene.play(man.TransformMatchingTex(texs[i-1], texs[i].move_to(origin)), **kwargs)
    return scene
