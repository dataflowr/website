install:
	julia -O0 -e 'import Pkg; Pkg.add("Franklin"); Pkg.add("JSON")'

serve:
	julia -O0 -e 'using Franklin; serve()'
