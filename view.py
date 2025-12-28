import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from init import *
from error import *

def get_data_to_visualize(defect_array,forward_signal_1_array,forward_signal_2_array,anti_signal_1_array,anti_signal_2_array,stack_1_array,stack_2_array):
    return(defect_array,np.sum(forward_signal_1_array,axis=0).astype(int),np.sum(forward_signal_2_array,axis=0).astype(int),np.sum(anti_signal_1_array,axis=0).astype(int),np.sum(anti_signal_2_array,axis=0).astype(int),np.max(stack_1_array,axis=0).astype(int),np.max(stack_2_array,axis=0).astype(int))

def get_condensed_defects(defect_array):
    condensed_defect_array = defect_array[::2, 1::2]
    return(condensed_defect_array)

def get_condensed_signals(signal_array):
    condensed_signal_array = np.stack([signal_array[0,::2,1::2],signal_array[1,::2,1::2]])
    return(condensed_signal_array)

def get_number_of_particles(forward_signal_1_array,anti_signal_1_array,stack_1_array,forward_signal_2_array,anti_signal_2_array,stack_2_array):
    c_fs1_array, c_as1_array, c_s1_array = get_condensed_signals(forward_signal_1_array), get_condensed_signals(anti_signal_1_array), get_condensed_signals(stack_1_array)
    c_fs2_array, c_as2_array, c_s2_array = get_condensed_signals(forward_signal_2_array), get_condensed_signals(anti_signal_2_array), get_condensed_signals(stack_2_array)
    return(np.sum(c_fs1_array+c_as1_array+c_s1_array+c_fs2_array+c_as2_array+c_s2_array))

def verify_charge_1_conservation(forward_signal_1_array,backward_signal_1_array,anti_signal_1_array,stack_1_array,switch_1_array):
    c_forward_signal_array, c_backward_signal_array, c_anti_signal_array, c_stack_array = get_condensed_signals(forward_signal_1_array), get_condensed_signals(backward_signal_1_array), get_condensed_signals(anti_signal_1_array), get_condensed_signals(stack_1_array)

    charge_conservation_rows = np.sum(c_forward_signal_array + c_backward_signal_array - c_anti_signal_array - c_stack_array, axis=2)
    charge_conservation_columns = np.sum(c_forward_signal_array + c_backward_signal_array - c_anti_signal_array - c_stack_array, axis=1)

    return(charge_conservation_rows[::2,:],charge_conservation_columns[1::2,:])

def verify_charge_2_conservation(forward_signal_2_array,anti_signal_2_array,stack_2_array):
    c_forward_signal_array, c_anti_signal_array, c_stack_array = get_condensed_signals(forward_signal_2_array), get_condensed_signals(anti_signal_2_array), get_condensed_signals(stack_2_array)

    charge_conservation_rows = np.sum(c_forward_signal_array - c_anti_signal_array - c_stack_array, axis=2)
    charge_conservation_columns = np.sum(c_forward_signal_array - c_anti_signal_array - c_stack_array, axis=1)

    return(charge_conservation_rows[::2,:],charge_conservation_columns[1::2,:])



# ======================
# Particle types (as strings)
# ======================
PARTICLES = [
    "EMPTY", "DEFECT",
    # Forward signals
    "SIGNAL_1_EAST","SIGNAL_1_WEST","SIGNAL_1_NORTH","SIGNAL_1_SOUTH","SIGNAL_1_HIGH",
    "SIGNAL_2_EAST","SIGNAL_2_WEST","SIGNAL_2_NORTH","SIGNAL_2_SOUTH","SIGNAL_2_HIGH",
    # Anti-signals
    "ANTI_1_EAST","ANTI_1_WEST","ANTI_1_NORTH","ANTI_1_SOUTH","ANTI_1_HIGH",
    "ANTI_2_EAST","ANTI_2_WEST","ANTI_2_NORTH","ANTI_2_SOUTH","ANTI_2_HIGH",
    # Stacks
    "STACK_1","STACK_2"
]

# ======================
# Colors & symbols
# ======================
def hex_to_rgba_float(hex_color):
    r = int(hex_color[0:2],16)
    g = int(hex_color[2:4],16)
    b = int(hex_color[4:6],16)
    a = int(hex_color[6:8],16)
    return (r/255,g/255,b/255,a/255)

GRID_COLORS = {
    "GRID_LINES": hex_to_rgba_float("ecececff"),
}

PARTICLE_COLORS = {
    "DEFECT": hex_to_rgba_float("0071bcff"),
    "SIGNAL_1_EAST": hex_to_rgba_float("0071bcff"),
    "SIGNAL_1_WEST": hex_to_rgba_float("0071bcff"),
    "SIGNAL_1_NORTH": hex_to_rgba_float("0071bcff"),
    "SIGNAL_1_SOUTH": hex_to_rgba_float("0071bcff"),
    "SIGNAL_1_HIGH": hex_to_rgba_float("0071bcff"),
    "SIGNAL_2_EAST": hex_to_rgba_float("b4c8e6ff"),
    "SIGNAL_2_WEST": hex_to_rgba_float("b4c8e6ff"),
    "SIGNAL_2_NORTH": hex_to_rgba_float("b4c8e6ff"),
    "SIGNAL_2_SOUTH": hex_to_rgba_float("b4c8e6ff"),
    "SIGNAL_2_HIGH": hex_to_rgba_float("b4c8e6ff"),
    "ANTI_1_EAST": hex_to_rgba_float("fbb03bff"),
    "ANTI_1_WEST": hex_to_rgba_float("fbb03bff"),
    "ANTI_1_NORTH": hex_to_rgba_float("fbb03bff"),
    "ANTI_1_SOUTH": hex_to_rgba_float("fbb03bff"),
    "ANTI_1_HIGH": hex_to_rgba_float("fbb03bff"),
    "ANTI_2_EAST": hex_to_rgba_float("fac88cff"),
    "ANTI_2_WEST": hex_to_rgba_float("fac88cff"),
    "ANTI_2_NORTH": hex_to_rgba_float("fac88cff"),
    "ANTI_2_SOUTH": hex_to_rgba_float("fac88cff"),
    "ANTI_2_HIGH": hex_to_rgba_float("fac88cff"),
    "STACK_1": hex_to_rgba_float("fbb03bff"),
    "STACK_2": hex_to_rgba_float("fbb03bff")
}

PARTICLE_SYMBOLS = {
    "DEFECT": 'o',
    "SIGNAL_1_EAST": '>', "SIGNAL_1_WEST":'<', "SIGNAL_1_NORTH":'^', "SIGNAL_1_SOUTH":'v',
    "SIGNAL_1_HIGH": 's',
    "SIGNAL_2_EAST": '>', "SIGNAL_2_WEST":'<', "SIGNAL_2_NORTH":'^', "SIGNAL_2_SOUTH":'v',
    "SIGNAL_2_HIGH": 's',
    "ANTI_1_EAST": '>', "ANTI_1_WEST":'<', "ANTI_1_NORTH":'^', "ANTI_1_SOUTH":'v',
    "ANTI_1_HIGH": 's',
    "ANTI_2_EAST": '>', "ANTI_2_WEST":'<', "ANTI_2_NORTH":'^', "ANTI_2_SOUTH":'v',
    "ANTI_2_HIGH": 's',
    "STACK_1": 'o',
    "STACK_2": 'o'
}

PARTICLE_SIZES = {
    "DEFECT": 300,
    "SIGNAL_1_EAST": 300, "SIGNAL_1_WEST": 300, "SIGNAL_1_NORTH": 300, "SIGNAL_1_SOUTH": 300,
    "SIGNAL_1_HIGH": 300,
    "SIGNAL_2_EAST": 300, "SIGNAL_2_WEST": 300, "SIGNAL_2_NORTH": 300, "SIGNAL_2_SOUTH": 300,
    "SIGNAL_2_HIGH": 300,
    "ANTI_1_EAST": 300, "ANTI_1_WEST": 300, "ANTI_1_NORTH": 300, "ANTI_1_SOUTH": 300,
    "ANTI_1_HIGH": 300,
    "ANTI_2_EAST": 300, "ANTI_2_WEST": 300, "ANTI_2_NORTH": 300, "ANTI_2_SOUTH": 300,
    "ANTI_2_HIGH": 300,
    "STACK_1": 300,
    "STACK_2": 300
}

# ======================
# Build particle dictionary
# ======================
def build_particle_dict(defect_array,
                        forward_signal_1_array,
                        forward_signal_2_array,
                        anti_signal_1_array,
                        anti_signal_2_array,
                        stack_1_array,
                        stack_2_array):
    L = defect_array.shape[0]
    particle_dict = {}

    particle_dict["DEFECT"] = (defect_array > 0).astype(int)

    def split_signal(signal_array, base_codes, high_code):
        parts = {}

        # Count number of active directions at each site
        combined = np.sum(signal_array > 0, axis=0)  # shape (L, L)

        # HIGH = two or more directions
        high_mask = (combined >= 2).astype(int)
        parts[high_code] = high_mask

        # Single-direction arrows only where combined == 1
        single_mask = (combined == 1).astype(int)

        for d, code in enumerate(base_codes):
            parts[code] = ((signal_array[d] > 0) & (single_mask == 1)).astype(int)

        return parts

    # Forward signals
    particle_dict.update(split_signal(forward_signal_1_array,
                                    ["SIGNAL_1_EAST","SIGNAL_1_NORTH","SIGNAL_1_WEST","SIGNAL_1_SOUTH"],
                                    "SIGNAL_1_HIGH"))
    particle_dict.update(split_signal(forward_signal_2_array,
                                    ["SIGNAL_2_EAST","SIGNAL_2_NORTH","SIGNAL_2_WEST","SIGNAL_2_SOUTH"],
                                    "SIGNAL_2_HIGH"))
    # Anti-signals
    particle_dict.update(split_signal(anti_signal_1_array,
                                    ["ANTI_1_EAST","ANTI_1_NORTH","ANTI_1_WEST","ANTI_1_SOUTH"],
                                    "ANTI_1_HIGH"))
    particle_dict.update(split_signal(anti_signal_2_array,
                                    ["ANTI_2_EAST","ANTI_2_NORTH","ANTI_2_WEST","ANTI_2_SOUTH"],
                                    "ANTI_2_HIGH"))
    # Stacks
    particle_dict["STACK_1"] = (np.any(stack_1_array > 0, axis=0)).astype(int)
    particle_dict["STACK_2"] = (np.any(stack_2_array > 0, axis=0)).astype(int)

    return particle_dict

# ======================
# Plot function
# ======================
def plot_particles_by_priority(ax, particle_dict, priority_list):
    
    L = next(iter(particle_dict.values())).shape[0]
    mask_total = np.zeros((L,L), dtype=int)

    for ptype in priority_list:
        if ptype not in particle_dict:
            continue   # nothing of this type present

        mask = particle_dict[ptype] * (1 - mask_total)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            continue

        ax.scatter(xs, ys,
                   marker=PARTICLE_SYMBOLS[ptype],
                   c=[PARTICLE_COLORS[ptype]] * len(xs),
                   s=PARTICLE_SIZES[ptype],
                   edgecolors='black',
                   linewidths=0,
                   zorder=priority_list.index(ptype))

        mask_total += mask

# ======================
# Visualization
# ======================
def view_particles(t,
                                defect_array,
                                forward_signal_1_array,
                                forward_signal_2_array,
                                anti_signal_1_array,
                                anti_signal_2_array,
                                stack_1_array,
                                stack_2_array,
                                subgrid=None,
                                single_layer_view=False
                                ):
    
    # ======================
    # OPTIONAL SUBGRID
    # ======================

    if subgrid is None:
        slice_y = slice(None)
        slice_x = slice(None)
    else:
        slice_y, slice_x = subgrid

    # ======================
    # SCALE / REDUCE LATTICE
    # ======================

    reduced_defect_array = defect_array[slice_y, slice_x][::2, 1::2]

    reduced_forward_signal_1_array = forward_signal_1_array[
        :, slice_y, slice_x
    ][:, ::2, 1::2]

    reduced_forward_signal_2_array = forward_signal_2_array[
        :, slice_y, slice_x
    ][:, ::2, 1::2]

    reduced_anti_signal_1_array = anti_signal_1_array[
        :, slice_y, slice_x
    ][:, ::2, 1::2]

    reduced_anti_signal_2_array = anti_signal_2_array[
        :, slice_y, slice_x
    ][:, ::2, 1::2]

    reduced_stack_1_array = stack_1_array[
        :, slice_y, slice_x
    ][:, ::2, 1::2]

    reduced_stack_2_array = stack_2_array[
        :, slice_y, slice_x
    ][:, ::2, 1::2]

    # ======================
    # BUILD PARTICLES (unchanged)
    # ======================

    particle_dict = build_particle_dict(
        reduced_defect_array,
        reduced_forward_signal_1_array,
        reduced_forward_signal_2_array,
        reduced_anti_signal_1_array,
        reduced_anti_signal_2_array,
        reduced_stack_1_array,
        reduced_stack_2_array
    )

    # --- rest of your plotting code stays exactly the same ---

    fig, axes = plt.subplots(1,2,figsize=(10,5))
    L = reduced_defect_array.shape[0]

    # Draw light-grey grid
    for ax in axes:
        delta = 0.3
        for x in range(L):
            ax.plot([x,x],[0-delta,L-1+delta],color=GRID_COLORS["GRID_LINES"],linewidth=1,zorder=0)
        for y in range(L):
            ax.plot([0-delta,L-1+delta],[y,y],color=GRID_COLORS["GRID_LINES"],linewidth=1,zorder=0)
        ax.set_xlim(-0.5,L-0.5)
        ax.set_ylim(-0.5, L-0.5)
        ax.set_aspect('equal')
        ax.axis('off')

    # Priority lists
    left_priority = ["DEFECT",
                     "SIGNAL_1_EAST","SIGNAL_1_WEST","SIGNAL_1_NORTH","SIGNAL_1_SOUTH","SIGNAL_1_HIGH",
                     "SIGNAL_2_EAST","SIGNAL_2_WEST","SIGNAL_2_NORTH","SIGNAL_2_SOUTH","SIGNAL_2_HIGH"]

    right_priority = ["STACK_1","STACK_2",
                      "ANTI_1_EAST","ANTI_1_WEST","ANTI_1_NORTH","ANTI_1_SOUTH","ANTI_1_HIGH",
                      "ANTI_2_EAST","ANTI_2_WEST","ANTI_2_NORTH","ANTI_2_SOUTH","ANTI_2_HIGH"]

    plot_particles_by_priority(axes[0], particle_dict, left_priority)
    plot_particles_by_priority(axes[1], particle_dict, right_priority)

    if single_layer_view:
        fig.canvas.draw()  # ensure positions are computed
        bbox = axes[0].get_tightbbox(fig.canvas.get_renderer())
        fig.savefig(
            f"view/fig/view_particles_{t}.pdf",
            bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted())
        )
    else:
        plt.tight_layout()
        plt.savefig(
            f"view/fig/view_particles_{t}.pdf",
            bbox_inches='tight'
        )

    plt.close()


def get_vector(forward_signal_1_array,forward_signal_2_array):

    inter_11_array = forward_signal_1_array[1]
    inter_12_array = forward_signal_1_array[2]*(inter_11_array==0)
    inter_13_array = forward_signal_1_array[3]*(inter_11_array==0)*(inter_12_array==0)

    not_inter_1x_array = (inter_11_array==0)*(inter_12_array==0)*(inter_13_array==0)

    inter_20_array = forward_signal_2_array[0]*not_inter_1x_array
    inter_21_array = forward_signal_2_array[1]*not_inter_1x_array*(inter_20_array==0)
    inter_23_array = forward_signal_2_array[3]*not_inter_1x_array*(inter_20_array==0)*(inter_21_array==0)

    X = np.stack([inter_20_array-inter_12_array,inter_11_array+inter_21_array-(inter_13_array+inter_23_array)]).astype(np.int8)
    vector_array = np.stack(X, axis=-1)

    return(vector_array)


def view_field(t,forward_signal_1_array,forward_signal_2_array,subgrid=None):
    """
    Visualize the vector field formed by the forward signals.
    """
    vector_array = get_vector(forward_signal_1_array,forward_signal_2_array)

    # ======================
    # OPTIONAL SUBGRID
    # ======================

    if subgrid is None:
        slice_y = slice(None)
        slice_x = slice(None)
    else:
        slice_y, slice_x = subgrid

    # ======================
    # SCALE / REDUCE LATTICE
    # ======================

    vector_array = vector_array[
        slice_y, slice_x
    ][::2, 1::2]

    # Suppose vecs is your (d,d,2) array
    l = vector_array.shape[0]

    # Grid coordinates (array indices)
    x = np.arange(l)-l//2
    y = np.arange(l)-l//2
    X, Y = np.meshgrid(x, y, indexing="ij")

    #cond = ((np.absolute(X) + np.absolute(Y)) < l/2).astype(np.int8)

    U = vector_array[..., 0]#*cond  # x-component
    V = vector_array[..., 1]#*cond  # y-component

    fig, ax = plt.subplots(1,1,figsize=(2.7,2.4))
    plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    ax.quiver(Y, X, -U, -V, angles="xy", scale_units="xy", scale=2)

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.grid(False)
    #ax.legend(fontsize=7.5)

    plt.savefig(
            f"view/fig/field_{t}.pdf",
            bbox_inches='tight'
        )
    plt.close()

    


"""

# ======================
# Particle type codes
# ======================
EMPTY = 0
DEFECT = 1
SIGNAL_1 = 2
SIGNAL_2_LOW = 3
SIGNAL_2_HIGH = 4
ANTI_1 = 5
ANTI_2_LOW = 6
ANTI_2_HIGH = 7
STACK_1 = 8
STACK_2 = 9

# ======================
# Utilities
# ======================
def hex_to_rgba_float(hex_color):
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = int(hex_color[6:8], 16)
    return (r/255, g/255, b/255, a/255)

PARTICLE_COLORS = {
    DEFECT: hex_to_rgba_float("0071bcff"),
    SIGNAL_1: hex_to_rgba_float("b4c8e6ff"),
    SIGNAL_2_LOW: hex_to_rgba_float("b4c8e6ff"),
    SIGNAL_2_HIGH: (1, 1, 1, 1),
    ANTI_1: hex_to_rgba_float("fac88cff"),
    ANTI_2_LOW: hex_to_rgba_float("fac88cff"),
    ANTI_2_HIGH: (1, 1, 1, 1),
    STACK_1: hex_to_rgba_float("fbb03bff"),
    STACK_2: hex_to_rgba_float("fbb03bff"),
}

# ======================
# Direction convention
# ======================
# 0 = West, 1 = North, 2 = East, 3 = South
DIRS = {
    0: (-1,  0, '>'),
    1: ( 0, -1, 'v'),
    2: ( 1,  0, '<'),
    3: ( 0,  1, '^'),
}

# ======================
# Grid construction
# ======================
def get_grid(
    defect_array,
    signal_1_array,
    signal_2_array,
    anti_signal_1_array,
    anti_signal_2_array,
    stack_1_array,
    stack_2_array):

    reduced_stack_1_array = np.max(stack_1_array,axis=0).astype(int)
    reduced_stack_2_array = np.max(stack_2_array,axis=0).astype(int)

    defect = defect_array[::2, 1::2]
    s1 = signal_1_array[::2, 1::2]
    s2 = signal_2_array[::2, 1::2]
    a1 = anti_signal_1_array[::2, 1::2]
    a2 = anti_signal_2_array[::2, 1::2]
    st1 = reduced_stack_1_array[::2, 1::2]
    st2 = reduced_stack_2_array[::2, 1::2]

    shape = defect.shape
    grid_binary = np.full(shape, EMPTY, dtype=int)
    grid_int = np.full(shape, EMPTY, dtype=int)

    # defects & forward signals
    mask_defect = defect == 1
    mask_s1 = (s1 > 0) & (~mask_defect)
    mask_s2_high = (s2 > 1) & (~mask_defect) & (~mask_s1)
    mask_s2_low = (s2 == 1) & (~mask_defect) & (~mask_s1) & (~mask_s2_high)

    grid_binary[mask_defect] = DEFECT
    grid_binary[mask_s1] = SIGNAL_1
    grid_binary[mask_s2_high] = SIGNAL_2_HIGH
    grid_binary[mask_s2_low] = SIGNAL_2_LOW

    # stacks & anti-signals
    mask_st1 = st1 > 0
    mask_st2 = (st2 > 0) & (~mask_st1)
    mask_a1 = (a1 > 0) & (~mask_st1) & (~mask_st2)
    mask_a2_high = (a2 > 1) & (~mask_st1) & (~mask_st2) & (~mask_a1)
    mask_a2_low = (a2 == 1) & (~mask_st1) & (~mask_st2) & (~mask_a1) & (~mask_a2_high)

    grid_int[mask_st1] = STACK_1
    grid_int[mask_st2] = STACK_2
    grid_int[mask_a1] = ANTI_1
    grid_int[mask_a2_high] = ANTI_2_HIGH
    grid_int[mask_a2_low] = ANTI_2_LOW

    return grid_binary, grid_int

# ======================
# Drawing helpers
# ======================
def draw_vertex_particles(ax, grid, title):
    h, w = grid.shape

    # First draw stacks
    for i in range(h):
        for j in range(w):
            val = grid[i, j]
            if val in (STACK_1, STACK_2):
                ax.scatter(
                    j, i,
                    s=220,
                    c=[PARTICLE_COLORS[val]],
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=2
                )

    # Then draw defects on top
    for i in range(h):
        for j in range(w):
            val = grid[i, j]
            if val == DEFECT:
                ax.scatter(
                    j, i,
                    s=220,
                    c=[PARTICLE_COLORS[val]],
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=5   # higher than signals/stacks
                )

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")

def draw_directional_signals(ax, signal_array, color_low, color_high=None, size=140):
    
    signal_array shape = (4, L, L)

    Rule per vertex:
      - 0 directions  -> nothing
      - 1 direction   -> oriented triangle
      - >=2 directions -> square
    
    reduced = signal_array[:, ::2, 1::2]
    _, h, w = reduced.shape

    for i in range(h):
        for j in range(w):
            active_dirs = []
            max_val = 0

            for d in range(4):
                val = reduced[d, i, j]
                if val > 0:
                    active_dirs.append(d)
                    max_val = max(max_val, val)

            if not active_dirs:
                continue

            # choose color (low / high)
            color = color_high if (color_high and max_val > 1) else color_low

            # ---- single direction → triangle ----
            if len(active_dirs) == 1:
                d = active_dirs[0]
                marker = DIRS[d][2]

                ax.scatter(
                    j, i,
                    s=size,
                    marker=marker,
                    c=[color],
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=4
                )

            # ---- multiple directions → square ----
            else:
                ax.scatter(
                    j, i,
                    s=size * 1.15,
                    marker='s',
                    c=[color],
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=4
                )


# ======================
# Main visualization
# ======================
def visualize_ancilla_particles(
    t,
    defect_array,
    forward_signal_1_array,
    forward_signal_2_array,
    anti_signal_1_array,
    anti_signal_2_array,
    stack_1_array,
    stack_2_array):

    grid_binary, grid_int = get_grid(
        defect_array,
        np.sum(forward_signal_1_array, axis=0),
        np.sum(forward_signal_2_array, axis=0),
        np.sum(anti_signal_1_array, axis=0),
        np.sum(anti_signal_2_array, axis=0),
        stack_1_array,
        stack_2_array
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    h, w = grid_binary.shape

    # light-grey lattice
    for ax in axes:
        for x in range(w):
            ax.plot([x, x], [0, h-1], color="lightgrey", linewidth=0.5, zorder=1)
        for y in range(h):
            ax.plot([0, w-1], [y, y], color="lightgrey", linewidth=0.5, zorder=1)

    draw_vertex_particles(axes[0], grid_binary, "Defects & Signals")
    draw_vertex_particles(axes[1], grid_int, "Stacks & Anti-signals")

    # ===== Forward signals (left panel) =====
    draw_directional_signals(
        axes[0],
        forward_signal_1_array*(defect_array[None, :, :] == 0),
        PARTICLE_COLORS[SIGNAL_1]
    )

    draw_directional_signals(
        axes[0],
        forward_signal_2_array*(defect_array[None, :, :] == 0),
        PARTICLE_COLORS[SIGNAL_2_LOW],
        PARTICLE_COLORS[SIGNAL_2_HIGH]
    )

# ===== Anti-signals (right panel) =====
    draw_directional_signals(
        axes[1],
        anti_signal_1_array*(np.max(stack_1_array + stack_2_array, axis=0) == 0),
        PARTICLE_COLORS[ANTI_1]
    )

    draw_directional_signals(
        axes[1],
        anti_signal_2_array*(np.max(stack_1_array + stack_2_array, axis=0) == 0),
        PARTICLE_COLORS[ANTI_2_LOW],
        PARTICLE_COLORS[ANTI_2_HIGH]
    )

    plt.tight_layout()
    plt.savefig(f"view/fig/ancilla_particles_{t}.png", bbox_inches="tight")
    plt.close()

"""

"""
EMPTY = 0
DEFECT = 1
SIGNAL_1 = 2
SIGNAL_2_LOW = 3
SIGNAL_2_HIGH = 4
ANTI_1 = 5
ANTI_2_LOW = 6
ANTI_2_HIGH = 7
STACK_1 = 8
STACK_2 = 9

def get_grid(
    defect_array,
    signal_1_array,
    signal_2_array,
    anti_signal_1_array,
    anti_signal_2_array,
    stack_1_array,
    stack_2_array):

    # Réduction comme dans ton code
    defect = defect_array[::2, 1::2]
    s1 = signal_1_array[::2, 1::2]
    s2 = signal_2_array[::2, 1::2]
    a1 = anti_signal_1_array[::2, 1::2]
    a2 = anti_signal_2_array[::2, 1::2]
    st1 = stack_1_array[::2, 1::2]
    st2 = stack_2_array[::2, 1::2]

    shape = defect.shape
    grid_binary = np.full(shape, EMPTY, dtype=int)
    grid_int = np.full(shape, EMPTY, dtype=int)

    # ===== LEFT GRID (defects + signals) =====
    mask_defect = defect == 1
    mask_s1 = (s1 > 0) & (~mask_defect)
    mask_s2_high = (s2 > 1) & (~mask_defect) & (~mask_s1)
    mask_s2_low = (s2 == 1) & (~mask_defect) & (~mask_s1) & (~mask_s2_high)

    grid_binary[mask_defect] = DEFECT
    grid_binary[mask_s1] = SIGNAL_1
    grid_binary[mask_s2_high] = SIGNAL_2_HIGH
    grid_binary[mask_s2_low] = SIGNAL_2_LOW
    # Everything else remains EMPTY

    # ===== RIGHT GRID (stack + anti-signals) =====
    mask_st1 = st1 > 0
    mask_st2 = (st2 > 0) & (~mask_st1)
    mask_a1 = (a1 > 0) & (~mask_st1) & (~mask_st2)
    mask_a2_high = (a2 > 1) & (~mask_st1) & (~mask_st2) & (~mask_a1)
    mask_a2_low = (a2 == 1) & (~mask_st1) & (~mask_st2) & (~mask_a1) & (~mask_a2_high)

    grid_int[mask_st1] = STACK_1
    grid_int[mask_st2] = STACK_2
    grid_int[mask_a1] = ANTI_1
    grid_int[mask_a2_high] = ANTI_2_HIGH
    grid_int[mask_a2_low] = ANTI_2_LOW
    # Everything else remains EMPTY

    return grid_binary, grid_int

def hex_to_rgba_float(hex_color):
    
    #Convert 8-digit hex RGBA (RRGGBBAA) to Matplotlib RGBA floats (0-1)
    
    hex_color = hex_color.strip()  # remove spaces if any
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    a = int(hex_color[6:8], 16)
    return (r/255, g/255, b/255, a/255)

PARTICLE_COLORS = {
    DEFECT: hex_to_rgba_float("0071bcff"),
    SIGNAL_1: hex_to_rgba_float("b4c8e6ff"),
    SIGNAL_2_LOW: hex_to_rgba_float("b4c8e6ff"),
    SIGNAL_2_HIGH: (1, 1, 1, 1),

    ANTI_1: hex_to_rgba_float("fac88cff"),
    ANTI_2_LOW: hex_to_rgba_float("fac88cff"),
    ANTI_2_HIGH: (1, 1, 1, 1),

    STACK_1: hex_to_rgba_float("fbb03bff"),
    STACK_2: hex_to_rgba_float("fbb03bff"),
}

def draw_vertex_particles(ax, grid, title):
    h, w = grid.shape

    for i in range(h):
        for j in range(w):
            val = grid[i, j]
            if val != EMPTY:
                ax.scatter(
                    j, i,
                    s=200,                     # circle size
                    c=[PARTICLE_COLORS[val]],
                    marker='o',
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=3
                )

    # Draw lattice (optional)
    ax.set_xticks(np.arange(-0.5, w, 1))
    ax.set_yticks(np.arange(-0.5, h, 1))
    ax.grid(color="gray", linewidth=0.5, alpha=0.3)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # invert y to match array indexing
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")

def visualize_ancilla_particles(t,
    defect_array,
    signal_1_array,
    signal_2_array,
    anti_signal_1_array,
    anti_signal_2_array,
    stack_1_array,
    stack_2_array):

    grid_binary, grid_int = get_grid(
        defect_array,
        signal_1_array,
        signal_2_array,
        anti_signal_1_array,
        anti_signal_2_array,
        stack_1_array,
        stack_2_array)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    h, w = grid_binary.shape

    for x in range(w):
        axes[0].plot([x, x], [0, h-1], color="lightgrey", linewidth=0.5, zorder=1)
        axes[1].plot([x, x], [0, h-1], color="lightgrey", linewidth=0.5, zorder=1)
    for y in range(h):
        axes[0].plot([0, w-1], [y, y], color="lightgrey", linewidth=0.5, zorder=1)
        axes[1].plot([0, w-1], [y, y], color="lightgrey", linewidth=0.5, zorder=1)

    draw_vertex_particles(axes[0], grid_binary, "Defects & Signals")
    draw_vertex_particles(axes[1], grid_int, "Stacks & Anti-signals")

    plt.tight_layout()
    plt.savefig(f"view/fig/ancilla_particles_{t}.png", bbox_inches="tight")
    plt.close()


def visualize_ancilla_image(t,
    defect_array,
    signal_1_array,
    signal_2_array,
    anti_signal_1_array,
    anti_signal_2_array,
    stack_1_array,
    stack_2_array):

    grid_binary, grid_int = get_grid(
        defect_array,
        signal_1_array,
        signal_2_array,
        anti_signal_1_array,
        anti_signal_2_array,
        stack_1_array,
        stack_2_array)

    cmap = ListedColormap([
    "black",   # EMPTY
    hex_to_rgba_float("0071bcff"),  # DEFECT
    hex_to_rgba_float("b4c8e6ff"),  # SIGNAL_1
    hex_to_rgba_float("b4c8e6ff"),  # SIGNAL_2_LOW 
    "white",  # SIGNAL_2_HIGH
    hex_to_rgba_float("fac88cff"),  # ANTI_1
    hex_to_rgba_float("fac88cff"),  # ANTI_2_LOW
    "white",  # ANTI_2_HIGH
    hex_to_rgba_float("fbb03bff"),  # STACK_1
    hex_to_rgba_float("fbb03bff")   # STACK_2
    ])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(grid_binary, cmap=cmap, interpolation="nearest",vmin=0, vmax=9)
    axes[0].set_title("Defects & Signals")
    axes[0].axis("off")

    axes[1].imshow(grid_int, cmap=cmap, interpolation="nearest",vmin=0, vmax=9)
    axes[1].set_title("Stacks & Anti-signals")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("view/fig/ancilla_visualization_{}.png".format(t), bbox_inches="tight")"""


"""

# ANSI escape sequences
BLACK = "\033[30m"
RED     = "\033[31m"
BLUE    = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"
PINK = "\033[38;5;211m"
LIGHT_PINK = "\033[38;5;217m"  # soft light pink
WHITE = "\033[97m"
GREY = "\033[90m"

def visualize_data(data_array,signal_1_array,signal_2_array):

    dim = data_array.shape
    d = dim[0]//2
    mask_array = init_mask(d)
    
    data_array_to_visualize = np.nan_to_num(data_array, nan=8).astype(np.int8)
    defect_array = get_defect_determistic(data_array,mask_array)

    for i in range(dim[0]):
        for j in range(dim[1]):
            val = data_array_to_visualize[i, j]
            if val == 8:
                if mask_array[i, j] == 1:
                    if defect_array[i,j] == 1:
                        print(BLUE + '●' + RESET, end=" ")
                    elif signal_1_array[i,j] == 1:
                        print(RED + '●' + RESET, end=" ")
                    elif signal_2_array[i,j] == 1:
                        print(WHITE + '●' + RESET, end=" ")
                    else:
                        print(BLACK + '*' + RESET, end=" ")
                elif mask_array[i, j] == -1:
                    print(BLACK + '-' + RESET, end=" ")
            else:
                print(str(val), end=" ")
        print()

def visualize_ancilla(defect_array,signal_1_array,signal_2_array,anti_signal_1_array,anti_signal_2_array,stack_1_array,stack_2_array):

    reduced_defect_array = defect_array[::2, 1::2]
    reduced_signal_1_array = signal_1_array[::2, 1::2]
    reduced_signal_2_array = signal_2_array[::2, 1::2]

    reduced_anti_signal_1_array = anti_signal_1_array[::2, 1::2]
    reduced_anti_signal_2_array = anti_signal_2_array[::2, 1::2]
    reduced_stack_1_array = stack_1_array[::2, 1::2]
    reduced_stack_2_array = stack_2_array[::2, 1::2]

    dim = reduced_defect_array.shape
    d = dim[0]

    for i in range(dim[0]):
        for j in range(dim[1]):
            if reduced_defect_array[i,j] == 1:
                print(BLUE + '●' + RESET, end=" ")
            elif reduced_signal_1_array[i,j] > 0:
                print(RED + '●' + RESET, end=" ")
            elif reduced_signal_2_array[i,j] > 1:
                print(WHITE + '●' + RESET, end=" ")
            elif reduced_signal_2_array[i,j] == 1:
                print(GREY + '●' + RESET, end=" ")
            else:
                print(BLACK + '*' + RESET, end=" ")
        print(' ', end=" ")

        for j in range(dim[1]):
            if reduced_stack_1_array[i,j] > 0:
                if reduced_stack_1_array[i,j] < 10:
                    print(RED + str(reduced_stack_1_array[i,j]) + RESET, end=" ")
                else:
                    print(RED + '■' + RESET, end=" ")
            elif reduced_stack_2_array[i,j] > 0:
                if reduced_stack_2_array[i,j] < 10:
                    print(GREY + str(reduced_stack_2_array[i,j]) + RESET, end=" ")
                else:
                    print(GREY + '■' + RESET, end=" ")
            elif reduced_anti_signal_1_array[i,j] > 0:
                print(RED + '●' + RESET, end=" ")
            elif reduced_anti_signal_2_array[i,j] > 1:
                print(WHITE + '●' + RESET, end=" ")
            elif reduced_anti_signal_2_array[i,j] == 1:
                print(GREY + '●' + RESET, end=" ")
            else:
                print(BLACK + '*' + RESET, end=" ")

        print()"""