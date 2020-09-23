<!--
Add here global page variables to use throughout your
website.
The website_* must be defined for the RSS to work
-->
@def website_title = "Deep Learning DIY"
@def website_descr = "Website for deep learning course"
@def website_url   = "https://tlienart.github.io/FranklinTemplates.jl/"

@def author = "Marc Lelarge"

@def mintoclevel = 2

<!--
Add here files or directories that should be ignored by Franklin, otherwise
these files might be copied and, if markdown, processed by Franklin which
you might not want. Indicate directories by ending the name with a `/`.
-->
@def ignore = ["node_modules/", "franklin", "franklin.pub"]

<!--
Add here global latex commands to use throughout your
pages. It can be math commands but does not need to be.
For instance:
* \newcommand{\phrase}{This is a long phrase to copy.}
-->
\newcommand{\R}{\mathbb R}
\newcommand{\scal}[1]{\langle #1 \rangle}
\newcommand{\blurb}[1]{~~~<p style="font-size: 1.15em; color: #333; line-height:1.5em">~~~#1~~~</p>~~~}
<!-- \newcommand{\youtube}[1]{~~~<iframe width="1020" height="574" src="https://www.youtube.com/embed/~~~#1~~~" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>~~~}
-->