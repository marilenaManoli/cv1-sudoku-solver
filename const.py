# Just constants and paths:
# Where train/test images live.
# Where to save frontalized images and templates.
# Grid sizes: NUM_CELLS = 9, CELL_SIZE = (64,64), overall SUDOKU_SIZE = (576,576).
# Max 2 templates per digit.

import os

TRAIN_IMAGES_PATH = os.path.join(".", "sudoku_puzzles", "train")
TEST_IMAGES_PATH = os.path.join(".", "sudoku_puzzles", "test")

FRONTALIZED_IMAGES_PATH = os.path.join(".", "frontalized_images")
TEMPLATES_PATH = os.path.join(".", "templates")

NUM_CELLS = 9
CELL_SIZE = (64, 64)
SUDOKU_SIZE = (CELL_SIZE[0]*NUM_CELLS, CELL_SIZE[1]*NUM_CELLS)

MAX_TEMPLATES_FOR_DIGIT = 2
