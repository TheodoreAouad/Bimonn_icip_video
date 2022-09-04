from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def set_borders_to(ar: np.ndarray, border: Tuple, value: float = 0, ):
    res = ar + 0
    if border[0] != 0:
        res[:border[0], :] = value
        res[-border[0]:, :] = value
    if border[1] != 0:
        res[:, :border[1]] = value
        res[:, -border[1]:] = value
    return res


def rand_shape_2d(shape, rng_float=lambda shape: np.random.rand(shape[0], shape[1])):
    try:
        return rng_float(shape)
    except TypeError:
        return rng_float(*shape)


def straight_rect(width, height):
    return np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])


def numba_straight_rect(width, height):
    return np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def numba_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def numba_transform_rect(rect: np.ndarray, R: np.ndarray, offset: np.ndarray):
    return np.dot(rect, R) + offset


def numba_correspondance(ar: np.ndarray) -> np.ndarray:
    return 1 - ar


def numba_invert_proba(ar: np.ndarray, p_invert: float) -> np.ndarray:
    if np.random.rand() < p_invert:
        return numba_correspondance(ar)
    return ar


def invert_proba(ar, p_invert: float, rng_float) -> np.ndarray:
    if rng_float() < p_invert:
        return 1 - ar
    return ar


def get_rect(x, y, width, height, angle):
    rect = straight_rect(width, height)
    # rect = numba_straight_rect(width, height)
    theta = (np.pi / 180.0) * angle
    R = rotation_matrix(theta)
    # R = numba_rotation_matrix(theta)
    offset = np.array([x, y])
    transformed_rect = np.dot(rect, R) + offset
    # transformed_rect = numba_transform_rect(rect.astype(float), R.astype(float), offset.astype(float))
    return transformed_rect

# def get_rect(x, y, width, height, angle):
#     rect = np.array([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
#     theta = (np.pi / 180.0) * angle
#     R = np.array([[np.cos(theta), -np.sin(theta)],
#                   [np.sin(theta), np.cos(theta)]])
#     offset = np.array([x, y])
#     transformed_rect = np.dot(rect, R) + offset
#     return transformed_rect


def draw_poly(draw, poly, fill_value=1):
    draw.polygon([tuple(p) for p in poly], fill=fill_value)


def draw_ellipse(draw, center, radius, fill_value=1):
    bbox = (center[0] - radius[0], center[1] - radius[1], center[0] + radius[0], center[1] + radius[1])
    draw.ellipse(bbox, fill=fill_value)


def get_random_rotated_diskorect(
    size: Tuple, n_shapes: int = 30, max_shape: Tuple[int] = (15, 15), p_invert: float = 0.5,
        border=(4, 4), n_holes: int = 15, max_shape_holes: Tuple[int] = (5, 5), noise_proba=0.05,
        rng_float=np.random.rand, rng_int=np.random.randint, return_all_results=False, **kwargs
):
    diskorect = np.zeros(size)
    img = Image.fromarray(diskorect)
    draw = ImageDraw.Draw(img)

    all_results = [np.asarray(img).copy()]

    def draw_shape(max_shape, fill_value):
        x = rng_int(0, size[0] - 2)
        y = rng_int(0, size[0] - 2)

        if rng_float() < .5:
            W = rng_int(1, max_shape[0])
            L = rng_int(1, max_shape[1])

            angle = rng_float() * 45
            draw_poly(draw, get_rect(x, y, W, L, angle), fill_value=fill_value)

        else:
            rx = rng_int(1, max_shape[0]//2)
            ry = rng_int(1, max_shape[1]//2)
            draw_ellipse(draw, np.array([x, y]), (rx, ry), fill_value=fill_value)

    for _ in range(n_shapes):
        draw_shape(max_shape=max_shape, fill_value=1)
        all_results.append(np.asarray(img).copy())

    for _ in range(n_holes):
        draw_shape(max_shape=max_shape_holes, fill_value=0)
        all_results.append(np.asarray(img).copy())

    diskorect = np.asarray(img) + 0
    # print((rand_shape_2d(diskorect.shape, rng_float=rng_float) < noise_proba))
    diskorect[rand_shape_2d(diskorect.shape, rng_float=rng_float) < noise_proba] = 1
    diskorect = invert_proba(diskorect, p_invert, rng_float=rng_float)

    all_results.append(diskorect.copy())
    diskorect = set_borders_to(diskorect, border, value=0)


    if return_all_results:
        return all_results
    return diskorect


def get_random_diskorect_channels(size: Tuple, squeeze: bool = False, return_all_results: bool = False, *args, **kwargs):
    """Applies diskorect to multiple channels.

    Args:
        size (Tuple): (W, L, H)
        squeeze (bool, optional): If True, squeeze the output: if H = 1, returns size (W, L). Defaults to False.

    Raises:
        ValueError: size must be of len 2 or 3, either (W, L) or (W, L, H) with H number of channels.

    Returns:
        np.ndarray: size (W, L) or (W, L, H)
    """
    if len(size) == 3:
        W, L, H = size
    elif len(size) == 2:
        W, L = size
        H = 1
    else:
        raise ValueError(f"size argument must have 3 or 2 values, not f{len(size)}.")

    final_img = np.zeros((W, L, H))
    all_results = {}
    for chan in range(H):
        if return_all_results:
            all_results[chan] = get_random_rotated_diskorect((W, L), return_all_results=return_all_results, *args, **kwargs)
            # final_img[..., chan] = all_results[chan][-1]
        else:
            final_img[..., chan] = get_random_rotated_diskorect((W, L), *args, **kwargs)

    if return_all_results:
        return all_results

    if squeeze:
        return np.squeeze(final_img)
    return final_img

