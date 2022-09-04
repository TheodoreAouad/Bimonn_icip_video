import matplotlib.pyplot as plt
import manim as man


class TitleSlideAnimation(man.Scene):
    def construct(self):
        texs = [
            man.Text("Binary Morphological Neural", font_size=man.DEFAULT_FONT_SIZE),
            man.Text("Network", font_size=man.DEFAULT_FONT_SIZE),
            man.Text("Theodore Aouad, Hugues Talbot", slant=man.ITALIC, font_size=man.DEFAULT_FONT_SIZE*.6),
            man.Text(r"contact: {name}.{surname}@centralesupelec.fr", slant=man.ITALIC, font_size=man.DEFAULT_FONT_SIZE*.4),
            man.Text("University of Paris-Saclay, CentraleSupelec, Inria, CVN, France", font_size=man.DEFAULT_FONT_SIZE*.6),
            man.Text("ICIP 2022, Bordeaux, France"),
        ]

        texs[1].next_to(texs[0], man.DOWN)
        texs[2].next_to(texs[1], 2*man.DOWN)
        texs[3].next_to(texs[2], man.DOWN)
        texs[4].next_to(texs[3], man.DOWN)
        texs[5].next_to(texs[4], 2*man.DOWN)

        man.VGroup(*texs).move_to(man.ORIGIN)

        logo = plt.imread("Logo_OPIS.png")*255
        # logo[logo[..., -1] == 0] = 255
        # logo[..., -1] = 1
        
        img = man.ImageMobject(logo)
        img.height = 1.5
        img.next_to(texs[0], man.UP, aligned_edge=man.RIGHT)
        # rec = man.Rectangle(color=man.WHITE, height=img.height, width=img.width, fill_opacity=1).move_to(img)

        self.add(*texs, img)
        self.play(man.FadeIn(img.shift(2*man.RIGHT+.1*man.DOWN)), run_time=.5)
        self.wait(10)
        # self.play(man.FadeOut(*texs))
