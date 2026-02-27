import numpy as np
import matplotlib.pyplot as plt


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
    "ANTI_2_EAST": hex_to_rgba_float("f7ddbbff"),
    "ANTI_2_WEST": hex_to_rgba_float("f7ddbbff"),
    "ANTI_2_NORTH": hex_to_rgba_float("f7ddbbff"),
    "ANTI_2_SOUTH": hex_to_rgba_float("f7ddbbff"),
    "ANTI_2_HIGH": hex_to_rgba_float("f7ddbbff"),
    "STACK_1": hex_to_rgba_float("fbb03bff"),
    "STACK_2": hex_to_rgba_float("f7ddbbff")
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

# PARTICLE_SIZES = {
#     "DEFECT": 120,
#     "SIGNAL_1_EAST": 120, "SIGNAL_1_WEST": 120, "SIGNAL_1_NORTH": 120, "SIGNAL_1_SOUTH": 120,
#     "SIGNAL_1_HIGH": 80,
#     "SIGNAL_2_EAST": 120, "SIGNAL_2_WEST": 120, "SIGNAL_2_NORTH": 120, "SIGNAL_2_SOUTH": 120,
#     "SIGNAL_2_HIGH": 80,
#     "ANTI_1_EAST": 120, "ANTI_1_WEST": 120, "ANTI_1_NORTH": 120, "ANTI_1_SOUTH": 120,
#     "ANTI_1_HIGH": 120,
#     "ANTI_2_EAST": 120, "ANTI_2_WEST": 120, "ANTI_2_NORTH": 120, "ANTI_2_SOUTH": 120,
#     "ANTI_2_HIGH": 120,
#     "STACK_1": 120,
#     "STACK_2": 120
# }

PARTICLE_SIZES = {
    "DEFECT": 400,
    "SIGNAL_1_EAST": 400, "SIGNAL_1_WEST": 400, "SIGNAL_1_NORTH": 400, "SIGNAL_1_SOUTH": 400,
    "SIGNAL_1_HIGH": 250,
    "SIGNAL_2_EAST": 400, "SIGNAL_2_WEST": 400, "SIGNAL_2_NORTH": 400, "SIGNAL_2_SOUTH": 400,
    "SIGNAL_2_HIGH": 250,
    "ANTI_1_EAST": 400, "ANTI_1_WEST": 400, "ANTI_1_NORTH": 400, "ANTI_1_SOUTH": 400,
    "ANTI_1_HIGH": 400,
    "ANTI_2_EAST": 400, "ANTI_2_WEST": 400, "ANTI_2_NORTH": 400, "ANTI_2_SOUTH": 400,
    "ANTI_2_HIGH": 400,
    "STACK_1": 400,
    "STACK_2": 400
}

L_REFERENCE = 13

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

    stack_values = {
    "STACK_1": np.max(stack_1_array, axis=0),
    "STACK_2": np.max(stack_2_array, axis=0),
    }

    return particle_dict, stack_values

# ======================
# Plot function
# ======================
def plot_particles_by_priority(ax, particle_dict, priority_list, stack_values=None):
    
    L = next(iter(particle_dict.values())).shape[0]
    mask_total = np.zeros((L,L), dtype=int)

    scaling_factor = (L_REFERENCE/L) ** 2

    for ptype in priority_list:
        if ptype not in particle_dict:
            continue

        mask = particle_dict[ptype] * (1 - mask_total)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            continue

        ax.scatter(xs, ys,
                   marker=PARTICLE_SYMBOLS[ptype],
                   c=[PARTICLE_COLORS[ptype]] * len(xs),
                   s=PARTICLE_SIZES[ptype] * scaling_factor,
                   edgecolors='black',
                   linewidths=0,
                   zorder=priority_list.index(ptype))

        # ---- ADD VALUE LABELS FOR STACKS ----
        if stack_values is not None and ptype in stack_values:
            values = stack_values[ptype]
            for x, y in zip(xs, ys):
                val = int(values[y, x])
                if val > 0:
                    ax.text(
                        x, y - 0.05, f"{val}",
                        color="white",
                        ha="center", va="center",
                        fontsize=14 * np.sqrt(scaling_factor),
                        fontweight="bold",
                        zorder=priority_list.index(ptype) + 1
                    )

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
            subgrid,
            simple_view,
            step,
            path
            ):
    
    scaling_factor = (L_REFERENCE/defect_array.shape[0]) ** 2
    
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

    reduced_defect_array = defect_array

    reduced_forward_signal_1_array = forward_signal_1_array

    reduced_forward_signal_2_array = forward_signal_2_array

    reduced_anti_signal_1_array = anti_signal_1_array

    reduced_anti_signal_2_array = anti_signal_2_array

    reduced_stack_1_array = stack_1_array

    reduced_stack_2_array = stack_2_array

    # ======================
    # BUILD PARTICLES (unchanged)
    # ======================

    particle_dict, stack_values = build_particle_dict(
        reduced_defect_array,
        reduced_forward_signal_1_array,
        reduced_forward_signal_2_array,
        reduced_anti_signal_1_array,
        reduced_anti_signal_2_array,
        reduced_stack_1_array,
        reduced_stack_2_array
    )

    z = np.zeros_like(reduced_forward_signal_1_array[0,:,:])
    particle_dict_west, stack_values_west = build_particle_dict(
        reduced_defect_array,
        np.stack([reduced_forward_signal_1_array[0,:,:],z,z,z],axis=0),
        np.stack([reduced_forward_signal_2_array[0,:,:],z,z,z],axis=0),
        np.stack([reduced_anti_signal_1_array[0,:,:],z,z,z],axis=0),
        np.stack([reduced_anti_signal_2_array[0,:,:],z,z,z],axis=0),
        np.stack([reduced_stack_1_array[0,:,:],z,z,z],axis=0),
        np.stack([reduced_stack_2_array[0,:,:],z,z,z],axis=0)
    )

    particle_dict_north, stack_values_north = build_particle_dict(
        reduced_defect_array,
        np.stack([z,reduced_forward_signal_1_array[1,:,:],z,z],axis=0),
        np.stack([z,reduced_forward_signal_2_array[1,:,:],z,z],axis=0),
        np.stack([z,reduced_anti_signal_1_array[1,:,:],z,z],axis=0),
        np.stack([z,reduced_anti_signal_2_array[1,:,:],z,z],axis=0),
        np.stack([z,reduced_stack_1_array[1,:,:],z,z],axis=0),
        np.stack([z,reduced_stack_2_array[1,:,:],z,z],axis=0)
    )

    particle_dict_east, stack_values_east = build_particle_dict(
        reduced_defect_array,
        np.stack([z,z,reduced_forward_signal_1_array[2,:,:],z],axis=0),
        np.stack([z,z,reduced_forward_signal_2_array[2,:,:],z],axis=0),
        np.stack([z,z,reduced_anti_signal_1_array[2,:,:],z],axis=0),
        np.stack([z,z,reduced_anti_signal_2_array[2,:,:],z],axis=0),
        np.stack([z,z,reduced_stack_1_array[2,:,:],z],axis=0),
        np.stack([z,z,reduced_stack_2_array[2,:,:],z],axis=0)
    )

    particle_dict_south, stack_values_south = build_particle_dict(
        reduced_defect_array,
        np.stack([z,z,z,reduced_forward_signal_1_array[3,:,:]],axis=0),
        np.stack([z,z,z,reduced_forward_signal_2_array[3,:,:]],axis=0),
        np.stack([z,z,z,reduced_anti_signal_1_array[3,:,:]],axis=0),
        np.stack([z,z,z,reduced_anti_signal_2_array[3,:,:]],axis=0),
        np.stack([z,z,z,reduced_stack_1_array[3,:,:]],axis=0),
        np.stack([z,z,z,reduced_stack_2_array[3,:,:]],axis=0)
    )

    # --- rest of your plotting code stays exactly the same ---

    if simple_view:
        fig, axes = plt.subplots(1,2,figsize=(10,5))
    else:
        fig, axes = plt.subplots(1,6,figsize=(30,5))
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

    plot_particles_by_priority(
        axes[0], particle_dict, left_priority
    )

    plot_particles_by_priority(
        axes[1], particle_dict, right_priority, stack_values=stack_values
    )

    if not simple_view:

        plot_particles_by_priority(
            axes[2], particle_dict_west, right_priority, stack_values=stack_values_west
        )

        plot_particles_by_priority(
            axes[3], particle_dict_north, right_priority, stack_values=stack_values_north
        )

        plot_particles_by_priority(
            axes[4], particle_dict_east, right_priority, stack_values=stack_values_east
        )

        plot_particles_by_priority(
            axes[5], particle_dict_south, right_priority, stack_values=stack_values_south
        )

    if not simple_view:
        fig.text(0.17, 1.05,
            f"iter. {step[0]}.{step[1]}",
            ha="center",
            va="top",
            fontsize=25 * np.sqrt(scaling_factor))
    else:
        fig.text(0.5, 1.05,
            f"iter. {step[0]}.{step[1]}",
            ha="center",
            va="top",
            fontsize=25 * np.sqrt(scaling_factor))
    fig.tight_layout(w_pad=0.05, h_pad=0.05)
    fig.savefig(
        path+f"/{t}.pdf",
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


def view_field(t,forward_signal_1_array,forward_signal_2_array,subgrid,path):
    """
    Visualize the vector field formed by the forward signals.
    """
    L = forward_signal_1_array.shape[0]
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

    vector_array = vector_array

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

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    scaling_factor = (max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / L)

    ax.quiver(Y, X, -U, -V, angles="xy", scale_units="xy", scale=7*scaling_factor)

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.grid(False)
    #ax.legend(fontsize=7.5)

    plt.savefig(
            path+f"/{t}.pdf",
            bbox_inches='tight'
        )
    plt.close()