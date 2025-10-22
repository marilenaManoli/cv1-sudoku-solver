
# Define CELL_COORDINATES: for a few training images, list which cell positions contain which digits 
# (you’ll use these to crop ground-truth digit templates).
# Implement get_template_pipeline() — usually: 
# grayscale/denoise → edges → (optional morph) → biggest contour → 4 corners → warp to square → split into 81 cells.
# This pipeline is reused by both template creation and testing.

from pipeline import Pipeline

# BEGIN YOUR IMPORTS

# END YOUR IMPORTS

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example

CELL_COORDINATES = {"image_0.jpg": {1: (0, 0),
                                    2: (1, 1)},
                    "image_2.jpg": {1: (2, 3),
                                    3: [(2, 1), (0, 4)],
                                    9: (5, 6)}}
"""

# CELL_COORDINATES =

# END YOUR CODE


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def get_template_pipeline():
    # BEGIN YOUR CODE

    # pipeline =
    
    # return pipeline

    # END YOUR CODE

    raise NotImplementedError
