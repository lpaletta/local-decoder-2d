import os
import sys
import pickle

import numpy as np
from PIL import Image
from glob import glob
from pdf2image import convert_from_path

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from param import PHYSICS_PARAMS, DECODER_PARAMS, SIMULATION_PARAMS
from plot import *

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

OUT_PATH = "visualization/out/"
FIG_PATH = "visualization/fig/new"

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

filename_1 = os.path.join(OUT_PATH, "configuration_history.npy")
with open(filename_1, "rb") as f:
    configuration_history = pickle.load(f)
filename_2 = os.path.join(OUT_PATH, "step_history.npy")
with open(filename_2, "rb") as f:
    step_history = pickle.load(f)
print(f"{filename_1} loaded as configuration history")
print(f"{filename_2} loaded as step history")

# --- Directory containing the PNG frames ---

particles_dir = os.path.join(FIG_PATH, "particles")
field_dir = os.path.join(FIG_PATH, "field")
os.makedirs(particles_dir, exist_ok=True)
os.makedirs(field_dir, exist_ok=True)

animation_path = os.path.join(FIG_PATH, "animation.gif")

# ---------------------------------------------------------------------
# Plot view
# ---------------------------------------------------------------------

# Preconfigurated view parameters
example = "pair_of_defects"
#example = "single_measurement_error"
#example = "artificial_defect"
#example = "test"

if example == "pair_of_defects":
    subgrid = None
    simple_view = True

elif example == "artificial_defect":
    #subgrid = None
    subgrid = (slice(16, 30), slice(16, 30))
    simple_view = True

elif example == "single_measurement_error":
    subgrid = None
    simple_view = False

elif example == "test":
    subgrid = None
    simple_view = True

k = 0
for i in range(len(configuration_history)):
    same = False
    if i!=0:
        same = all(np.array_equal(a, b) for a, b in zip(configuration_history[i],configuration_history[i-1]))
    if not same:
        defect_array,forward_signal_1_array,forward_signal_2_array,anti_signal_1_array,anti_signal_2_array,stack_1_array,stack_2_array = configuration_history[i]

        view_particles(k,
                defect_array,
                forward_signal_1_array,
                forward_signal_2_array,
                anti_signal_1_array,
                anti_signal_2_array,
                stack_1_array,
                stack_2_array,
                subgrid,
                simple_view,
                step=step_history[i],
                path=particles_dir
                )
        
        view_field(k,
                forward_signal_1_array,
                forward_signal_2_array,
                subgrid,
                path=field_dir)
        
        k+=1


# Get all PDF files and sort them
pdf_files = glob(os.path.join(particles_dir, "*.pdf"))
pdf_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

# Convert each PDF to PNG and collect the frames
frames = []
for pdf_file in pdf_files:
    # Convert the first page of the PDF to a PIL Image
    images = convert_from_path(pdf_file, dpi=400)
    png_path = pdf_file.replace(".pdf", ".png")
    images[0].save(png_path, "PNG")  # Save as PNG
    frames.append(Image.open(png_path))  # Open the PNG

# --- Save as GIF ---
frames[0].save(
    animation_path,
    format="GIF",
    append_images=frames[1:],
    save_all=True,
    duration=200,  # Duration in milliseconds (1000ms = 1s)
    loop=0,
)

print(f"GIF saved at {animation_path}, infinite loop.")