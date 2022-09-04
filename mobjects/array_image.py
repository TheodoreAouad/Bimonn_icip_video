from typing import Callable, Tuple
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
            width: float = None,
            show_value_fn: Callable = default_show_value,
            fill_opacity: float = .5,
            *args, **kwargs
            ):
        if width is None:
            width = height
        if not draw_frontier:
            kwargs["stroke_width"] = 0

        super().__init__(color=color, fill_color=color, height=height, width=width, fill_opacity=fill_opacity, *args, **kwargs)
        self.value = value

        self.show_value = show_value
        self.draw_frontier = draw_frontier
        self.show_value_fn = show_value_fn

        if show_value:
            self.value_mobject = man.Tex(self.show_value_fn(self.value)).move_to(self.get_center())

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
            shape_target: Tuple[float] = None,
            show_value_fn: Callable = default_show_value,
            center_image: bool = False,
            vmin_cmap: float = None,
            vmax_cmap: float = None,
            fill_opacity: float = .5,
            *args, **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.array = array
        self.fill_opacity = fill_opacity

        if mask is None:
            self.mask = np.ones(array.shape).astype(bool)
        else:
            self.mask = mask

        self.show_value = show_value
        self.cmap = None  # TODO: implement cmap
        self.draw_grid = draw_grid

        self.shape_target = shape_target
        if self.shape_target is not None:
            self.horizontal_stretch = self.shape_target[0] / self.shape[0]
            self.vertical_stretch = self.shape_target[1] / self.shape[1]
        else:
            self.horizontal_stretch = horizontal_stretch
            self.vertical_stretch = vertical_stretch

        self.show_value_fn = show_value_fn
        self.center_image = center_image
        self._vmin_cmap = vmin_cmap
        self._vmax_cmap = vmax_cmap
        self.all_pixels = [[None for _ in range(self.shape[1])] for _ in range(self.shape[0])]

        self.cmap_str = cmap
        if isinstance(cmap, str):
            self.cmap = get_cmap(cmap)
        else:
            self.cmap = cmap


        if self.center_image:
            center = 0
        else:
            center = self.get_center()
        self.build_all_pixels(center)


    @property
    def vmin(self):
        return self.array.min()

    @property
    def vmax(self):
        return self.array.max()

    @property
    def vmin_cmap(self):
        if self._vmin_cmap is None:
            return self.array.min()
        return self._vmin_cmap

    @property
    def vmax_cmap(self):
        if self._vmax_cmap is None:
            return self.array.max()
        return self._vmax_cmap

    @property
    def shape(self):
        return self.array.shape + (1,)

    def build_all_pixels(self, center):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.mask[i, j]:
                    self.build_pixel(i, j, center)
                # self.build_pixel(i, j)

    def reset_all_pixels(self):
        center = self.get_center()
        for mob in self.submobjects.copy():
            self.remove(mob)
        self.build_all_pixels(center=center)

    def update_array(self, new_array: np.ndarray, new_mask: np.ndarray = None):
        self.array = new_array
        if new_mask is not None:
            self.mask = new_mask
        self.reset_all_pixels()
        return self

    def get_pixel(self, i, j):
        return self.all_pixels[i][j]

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def hscale(self):
        return self.horizontal_stretch

    @property
    def vscale(self):
        return self.vertical_stretch

    def build_pixel(self, i, j, center):
        pos = (
            j * self.hscale * man.RIGHT + i * self.vscale * man.DOWN +
            center - np.array([self.hscale * (self.shape[1] - 1) / 2, - self.vscale * (self.shape[0] - 1)/ 2, 0])
        )

        if self.dtype in [int, float]:
            vmin, vmax = self.vmin_cmap, self.vmax_cmap
            color = get_color_from_rgb(self.cmap((self.array[i, j] - vmin) / (vmax - vmin)))
        else:
            color = man.WHITE

        pixel = Pixel(
            value=self.array[i, j],
            height=self.vertical_stretch,
            width=self.horizontal_stretch,
            show_value=self.show_value,
            show_value_fn=self.show_value_fn,
            color=color,
            draw_frontier=self.draw_grid,
            fill_opacity=self.fill_opacity,
        ).move_to(pos)
        self.add(pixel)
        self.all_pixels[i][j] = pixel
        return pixel
