import subprocess
import os

latex_code = r"""
\documentclass{article}
\usepackage{graphicx}

% Download image using curl (works on Windows)
\immediate\write18{
  curl -o beach.jpg https://thumbs.dreamstime.com/b/stunning-hd-pic-caribbean-beach-cocktail-featuring-coconut-pineapple-set-against-palm-trees-sand-blue-sea-359953956.jpg
}

\begin{document}

\includegraphics[width=\linewidth]{beach.jpg}

\end{document}
"""

# Save the LaTeX code to a file
with open("beach_image.tex", "w") as f:
    f.write(latex_code)

# Compile using pdflatex with shell-escape enabled
subprocess.run(["pdflatex", "--shell-escape", "beach_image.tex"])

# Optional: open the resulting PDF
if os.path.exists("beach_image.pdf"):
    os.startfile("beach_image.pdf")  # Only works on Windows
