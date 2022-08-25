from typing import Callable
import numpy as np
import manim as man
from manim.utils.color import Color
from matplotlib.cm import get_cmap

from utils import get_color_from_rgb


def default_show_value(value: float) -> str:
    return f'{value}'


class Pixel(man.Rectangle):
    def __init__(
            self,
            value: float,
            color: Color = man.WHITE,
            show_value: bool = True,
            draw_frontier: bool = True,
            height: float = 1,
            width: float = 1,
            show_value_fn: Callable = default_show_value,
            *args, **kwargs
            ):
        super().__init__(color=color, fill_color=color, height=height, width=width, fill_opacity=.5, *args, **kwargs)
        self.value = value

        self.show_value = show_value
        self.draw_frontier = draw_frontier
        self.show_value_fn = show_value_fn

        if show_value:
            self.value_mobject = man.Text(self.show_value_fn(self.value)).move_to(self.get_center())

            if (self.value_mobject.get_left() < .9 * self.get_left()).any():
                self.value_mobject.stretch_to_fit_width(.9*self.width)

            if (self.value_mobject.get_top() > .7 * self.get_top()).any():
                self.value_mobject.stretch_to_fit_height(.9*self.height)


            self.add_to_back(self.value_mobject)


class ArrayImage(man.VGroup):
    def __init__(
            self,
            array: np.ndarray,
            mask: np.ndarray = None,
            show_value: bool = True,
            draw_grid: bool = True,
            cmap: str = 'viridis',
            horizontal_stretch: float = .5,
            vertical_stretch: float = .5,
            show_value_fn: Callable = default_show_value,
            center_image: bool = False,
            *args, **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.array = array

        if mask is None:
            self.mask = np.ones(array.shape).astype(bool)
        else:
            self.mask = mask

        self.show_value = show_value
        self.cmap = None  # TODO: implement cmap
        self.draw_grid = draw_grid
        self.horizontal_stretch = horizontal_stretch
        self.vertical_stretch = vertical_stretch
        self.show_value_fn = show_value_fn
        self.center_image = center_image

        self.cmap_str = cmap
        if isinstance(cmap, str):
            self.cmap = get_cmap(cmap)
        else:
            self.cmap = cmap

        self.build_all_pixels()

    @property
    def vmin(self):
        return self.array.min()

    @property
    def vmax(self):
        return self.array.max()

    @property
    def shape(self):
        return self.array.shape + (1,)

    def build_all_pixels(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.mask[i, j]:
                    self.build_pixel(i, j)
                # self.build_pixel(i, j)

    def build_pixel(self, i, j):
        barycentre = 0
        if self.center_image:
            barycentre = np.array([(self.shape[1] - 1) / 2, - (self.shape[0] - 1 )/ 2, 0])
        pos = j * self.horizontal_stretch * man.RIGHT + i * self.vertical_stretch * man.DOWN - barycentre

        vmin, vmax = self.vmin, self.vmax

        color = get_color_from_rgb(self.cmap((self.array[i, j] - vmin) / (vmax - vmin)))

        pixel = Pixel(
            value=self.array[i, j],
            height=self.vertical_stretch,
            width=self.horizontal_stretch,
            show_value=self.show_value,
            show_value_fn=self.show_value_fn,
            color=color
        ).move_to(pos)
        self.add(pixel)
