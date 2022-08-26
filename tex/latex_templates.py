from manim import TexTemplate


latex_template = TexTemplate()
latex_template.add_to_preamble(r"\input{tex/packages}")
latex_template.add_to_preamble(r"\input{tex/newcommands}")
