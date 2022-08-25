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
