import manim as man


class DilationDrawingMob(man.VGroup):

    def __init__(self, radius: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.circle = man.Circle(radius=radius)
        self.vline = man.Line(start=self.circle.get_bottom(), end=self.circle.get_top())
        self.hline = man.Line(start=self.circle.get_left(), end=self.circle.get_right())

        self.add(self.circle, self.vline, self.hline)


class ErosionDrawingMob(man.VGroup):

    def __init__(self, radius: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.circle = man.Circle(radius=radius)
        self.hline = man.Line(start=self.circle.get_left(), end=self.circle.get_right())

        self.add(self.circle, self.hline)
