import manim as man

from utils import TemplateMathTex


class TestScene(man.Scene):
    def construct(self):
        TemplateMathTex(r"\tau_{\ominus} =", r"\frac{\sum_{i, j \in \Omega}w_{i, j} - b}{1 - v_1}"),  # 8
