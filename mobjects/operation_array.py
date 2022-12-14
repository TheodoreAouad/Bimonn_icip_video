from typing import Optional, List

import manim as man

from .operation_mob import DilationDrawingMob, ErosionDrawingMob, ConvolutionDrawingMob


class OperationMob(man.VGroup):

    def __init__(
            self,
            operation: man.Mobject,
            mob1: man.Mobject,
            mob2: man.Mobject,
            mob3: Optional[man.Mobject] = None,
            subscripts: Optional[List[man.Mobject]] = [None, None, None],
            horizontal_scale: float = 1,
            show_braces: bool = True,
            *args, **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.operation = operation
        self.mob1 = mob1
        self.mob2 = mob2
        self.mob3 = mob3
        self.subscripts = subscripts
        self.horizontal_scale = horizontal_scale
        self.show_braces = show_braces

        self.add(self.mob1)
        self.add(self.operation.next_to(self.mob1, self.horizontal_scale * man.RIGHT))
        self.add(self.mob2.next_to(self.operation, self.horizontal_scale * man.RIGHT))

        if self.show_braces:
            self.brace1 = man.Brace(self, direction=man.LEFT).next_to(self, self.horizontal_scale * man.LEFT)
            self.brace2 = man.Brace(self, direction=man.RIGHT).next_to(self, self.horizontal_scale * man.RIGHT)

            self.add(self.brace1)
            self.add(self.brace2)

        if self.mob3 is not None:
            self.add(man.MathTex(r"=").next_to(self, self.horizontal_scale * man.RIGHT))
            self.add(self.mob3.next_to(self, self.horizontal_scale * man.RIGHT))

        if subscripts[0] is not None:
            self.add(subscripts[0].next_to(self.mob1, man.DOWN))

        if subscripts[1] is not None:
            self.add(subscripts[1].next_to(self.mob2, man.DOWN))

        if subscripts[2] is not None and self.mob3 is not None:
            self.add(subscripts[2].next_to(self.mob3, man.DOWN))



class DilationOperationMob(OperationMob):
    def __init__(self, mob1: man.Mobject, mob2: man.Mobject, mob3: Optional[man.Mobject] = None, horizontal_scale: float = 1, *args, **kwargs):
        super().__init__(operation=DilationDrawingMob(radius=horizontal_scale*.3), mob1=mob1, mob2=mob2, mob3=mob3, horizontal_scale=horizontal_scale, *args, **kwargs)


class ErosionOperationMob(OperationMob):
    def __init__(self, mob1: man.Mobject, mob2: man.Mobject, mob3: Optional[man.Mobject] = None, horizontal_scale: float = 1, *args, **kwargs):
        super().__init__(operation=ErosionDrawingMob(radius=horizontal_scale*.3), mob1=mob1, mob2=mob2, mob3=mob3, horizontal_scale=horizontal_scale, *args, **kwargs)


class ConvolutionOperationMob(OperationMob):
    def __init__(self, mob1: man.Mobject, mob2: man.Mobject, mob3: Optional[man.Mobject] = None, horizontal_scale: float = 1, *args, **kwargs):
        super().__init__(operation=ConvolutionDrawingMob(radius=horizontal_scale*.3), mob1=mob1, mob2=mob2, mob3=mob3, horizontal_scale=horizontal_scale, *args, **kwargs)
