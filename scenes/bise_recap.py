import numpy as np
import manim as man
from scipy.signal import convolve2d

from example_array import example2, W1
from mobjects import ArrayImage
from bise.threshold_fn import tanh_threshold


class BiseRecapAnimation(man.Scene):
    def construct(self):
        texts = [
            man.Text("We define the BiSE operator"),  # 0
            man.Text("We apply a classical convolution on normalized weights"),  # 1
            man.Tex("We multiply by a scaling scaling factor $p$"), # 2
            man.Text("We apply a smooth thresholding"),  # 3
        ]

        texs = [
            man.MathTex("X", r"\circledast", r"\xi(", "W", ")", "- b"),  # 0
            man.MathTex("p(", "X", r"\circledast", r"\xi(", "W", ")", "- b", ")"),  # 1
            man.MathTex(r"\xi", r"\Big(", "p(", "X", r"\circledast", r"\xi(", "W", ")", "- b", ")", r"\Big)"),  # 2
            man.MathTex("X"),  # 3
        ]

        X = example2
        W = W1
        selem = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        b = (W[selem].min() + W[~selem].sum()) / 2
        p = 3
        xi = tanh_threshold

        XconvW = convolve2d(X, W, mode="same") - b
        pXconvW = p * X
        thresh_pXconvW = xi(pXconvW)

        mobs = {
            "X": ArrayImage(X, show_value=True),
            "W": ArrayImage(W, show_value=True),
            "selem": ArrayImage(selem, show_value=True),
            "b": man.MathTex("b"),
            "p": man.MathTex("p"),
            "xi": man.MathTex(r"\xi"),
            "XconvW": ArrayImage(XconvW, show_value=True),
            "pXconvW": ArrayImage(pXconvW, show_value=True),
            "thresh_pXconvW": ArrayImage(thresh_pXconvW, show_value=True),
        }

        
        self.play(mobs[X])
