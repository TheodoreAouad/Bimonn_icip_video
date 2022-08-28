import manim as man

from tex.latex_templates import latex_template


class DilationDrawingMob(man.VGroup):

    def __init__(self, radius: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add(man.MathTex(r"\dil{}{}", tex_template=latex_template))
        # self.radius = radius
        # self.circle = man.Circle(radius=radius)
        # self.vline = man.Line(start=self.circle.get_bottom(), end=self.circle.get_top())
        # self.hline = man.Line(start=self.circle.get_left(), end=self.circle.get_right())

        # self.add(self.circle, self.vline, self.hline)


class ErosionDrawingMob(man.VGroup):

    def __init__(self, radius: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add(man.MathTex(r"\ero{}{}", tex_template=latex_template))
        # self.radius = radius
        # self.circle = man.Circle(radius=radius)
        # self.hline = man.Line(start=self.circle.get_left(), end=self.circle.get_right())

        # self.add(self.circle, self.hline)


class ConvolutionDrawingMob(man.VGroup):
    def __init__(self, radius: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add(man.MathTex(r"\conv{}{}", tex_template=latex_template))