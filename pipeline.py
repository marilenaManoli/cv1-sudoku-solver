# A tiny pipeline runner:
# You pass a list of functions; it calls them in order, keeps track of the ordered corners, and plots intermediate outputs if you ask.
# It recognizes special steps by name (e.g., functions with “edge”, “contour”, “corner”, “frontalize”) to decide how to visualize.
# Returns (image_after_last_resize/warp, output_of_last_step) which you use as (frontalized_image, sudoku_cells) for recognition.

import numpy as np
import matplotlib.pyplot as plt

from utils import show_image, show_contours, show_corners, show_sudoku_cells


class Pipeline(object):
    def __init__(self, functions, parameters={}):
        self.functions = functions
        self.parameters = parameters

    def __call__(self, image, plot=False, figsize=(18, 12)):
        output = image.copy()
        ordered_corners = None
        
        if plot:
            nrows = len(self.functions) // 3 + 1
            ncols = 3
            figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            if len(axes.shape) == 1:
                axes = axes[np.newaxis, ...]
            
            show_image(output, axis=axes[0][0], as_gray=True)
            axes[0][0].set_title("Sudoku image", fontsize=16)

            for j in range(len(self.functions) + 1, nrows * ncols):
                axis = axes[j // ncols][j % ncols]
                show_image(np.ones((1, 1, 3)), axis=axis)
        
        for i, function in enumerate(self.functions):
            kwargs = self.parameters.get(function.__name__, {})
            if "frontalize" in function.__name__:
                output = function(image=image, ordered_corners=ordered_corners)
            else:
                output = function(output, **kwargs)

            if ("rescale" in function.__name__) or ("frontalize" in function.__name__) or ("resize" in function.__name__):
                image = output.copy()

            if "corner" in function.__name__:
                ordered_corners = output
            
            if plot:
                axis = axes[(i + 1) // ncols][(i + 1) % ncols]
                if "edge" in function.__name__:
                    show_image(np.bitwise_not(output), axis=axis, as_gray=True)
                elif "contour" in function.__name__:
                    show_contours(image, contours=output, axis=axis)
                elif "corner" in function.__name__:
                    show_corners(image, corners=ordered_corners, axis=axis)
                elif "sudoku_cells" in function.__name__:
                    show_sudoku_cells(sudoku_cells=output, axis=axis)
                else:
                    show_image(output, axis=axis, as_gray=True)
                
                title = " ".join(function.__name__.split("_"))
                if kwargs:
                    title += " | " + ",".join([f"{key}: {value}" for key, value in kwargs.items()][:1])
                axis.set_title(title, fontsize=16)

        return image, output
