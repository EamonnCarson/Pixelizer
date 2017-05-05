## Program Plan:

### So heres the plan as I see it in high level steps:
	1. Find the $n$ most prominent component colors in the picture.
	2. For each pixel, determine the components of the most prominent color in an area around that pixel inversely proportional to the energy of the pixel (see cs61b hw5).
	3. Select the maximum component and discretize it (i.e. take the floor of the magnitude).
	4. Color each pixel the color that corresponds to that discretized maximal component.

### As for implementation details, we'll also go by steps
	1. I'd rather not have to write true machine learning software for this small project, so either use premade ML software or use the K-centroid method from CS61A.
	2. Energy calculation can be done as per the method in CS61B hw5.
	4. It might be best to define regions rather than pixels so that colors can be changed in the end.

Desired Feature set:
	- GUI to select number of primary colors, change up primary colors before analysis, change the color of the primary colors after the analysis, and change the 'background color' (e.g. translate the origin of the RGB system in the output).
