import manim as man

from .operation_mob import DilationDrawingMob, ErosionDrawingMob


class OperationMob(man.VGroup):

    def __init__(self, operation: man.Mobject, mob1: man.Mobject, mob2: man.Mobject, horizontal_scale: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = operation
        self.mob1 = mob1
        self.mob2 = mob2
        self.horizontal_scale = horizontal_scale

        self.add(self.mob1)
        self.add(self.operation.nextto(self.mob1, self.horizontal_scale * man.RIGHT))
        self.add(self.mob2)

        self.brace1 = man.Brace(self, direction=man.LEFT).next_to(self, self.horizontal_scale * man.LEFT)
        self.brace2 = man.Brace(self, direction=man.LEFT).next_to(self, self.horizontal_scale * man.RIGHT)

        self.add(self.brace1)
        self.add(self.brace2)


class DilationOperationMob(OperationMob):
    def __init__(self, mob1: man.Mobject, mob2: man.Mobject, horizontal_scale: float = 1, *args, **kwargs):
        super.__init__(operation=DilationDrawingMob, mob1=mob1, mob2=mob2, horizontal_scale=horizontal_scale, *args, **kwargs)


class ErosionOperationMob(OperationMob):
    def __init__(self, mob1: man.Mobject, mob2: man.Mobject, horizontal_scale: float = 1, *args, **kwargs):
        super.__init__(operation=ErosionDrawingMob, mob1=mob1, mob2=mob2, horizontal_scale=horizontal_scale, *args, **kwargs)
