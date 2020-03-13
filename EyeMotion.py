from argparse import ArgumentParser, Namespace
from colorline import colorline
from decimal import *
from math import floor, fsum, inf, isinf, isnan, nan, sqrt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from os import chdir
import scipy.interpolate as si
from skimage.draw import line_aa
import warnings

"""
TODO: visited-activation would need line width/blurring, otherwise small steps, which remain inside the cell,
become unlikely due to the high activation from the previous step, or worse, if no other cell is reached, a random
walk will be performed inside the cell until another one can be reached. however increasing the grid resolution is limited,
otherwise it is too unlikely to hit already visited cells exactly and the line will behave like a snake player and
sneak through very tight paths which is not very reasonable...
A second problem which is more apparent when using minimum selection aka potential_weight == inf at a relatively uniform step size
is that the path will consist of squares of increasing size, which stems from the square size of the visited activation cells.
Furthermore: use continuous coords for drawing the lines, not the cell centers, ignore anti aliased line pixels with small weight.
--> Workaround: Since the grid resolution currently depends on the average step size, the step frequency should be chosen such
that the mean velocity is in accordance with eye motion literature. when the 
Currently the first point is not added to the potential since we were standing there already. similarly, subtract a blurred point
at the location of the starting location from the added distribution to not step on this area again...
 v
note: subtracting the added activation at the start location blurred by the kernel only works, when the step size is at least
the kernel diameter, otherwise the small steps add on top of each other and the "subtraction of the current location" doesn't work.
 `--> TODO: what about not adding the end of the step to the activation?

TODO?: depending on activation grid resolution and decay rate, computing the distance to the lines directly instead of using a grid
as intermeidate accumulater might be more efficient (if #lines << #cells, which directly follows from #steps << #cells). if an
epsilon is introduced to ignore lines with almost no contribution, this would be a new error source, which would need to be
compared to the error from the grid approximation.

note: non-anti aliased lines could cross each other without notice if they are roughly orthogonal -> use line_aa to draw a wider line
      the weighting is not very helpful as long as only the max of the line is considered.

try using an (inverse) normal distribution as fixation potential?
"""


def parse_args() -> (Namespace, int):
    parser = ArgumentParser('Simulation of drift eye motion to simulate the distortion in raster scans.')
    parser.add_argument('--field_size', default=2, type=float,
                      help='Width / Height of the field in which the simulated random walk takes place in degree.')
    parser.add_argument('--duration', default=3.5, type=float,
                      help='Duration of simulated scan in seconds.')
    parser.add_argument('--simulation_frequency', default=20.0, type=float,
                      help='Step frequency of simulated drift in 1/sec.')
    parser.add_argument('--dist_mean', default=0.05, type=float,
                      help='Mean drift distance in degree per step (further influenced by the potentials).')
    parser.add_argument('--dist_sigma', default=0.01, type=float,
                      help='Standard deviation of drift step distance in degree (further influenced by the potentials).')
    parser.add_argument('--dist_weight', default=1, type=float,
                      help='Exponential weight applied to the normal distribution. See potential_weight. nan uses the same value as potential_weight.')
    parser.add_argument('--abs_dist_min', default=1.0e-5, type=float,
                      help='Identical positions can cause problems in spline construction -> ensure a nonzero absolute value of the distance.')
    parser.add_argument('--abs_dist_max', default=inf, type=float,
                      help='Max distance to prevent unnaturally far, i.e. fast steps, set to inf for no limit (default). Setting a max limit allows dist_sigma == inf for uniform selection (in this special case, dist_weight (and dist_mean) will be ignored).')
    parser.add_argument('--step_candidates', default=100, type=int,
                      help='Number of candidate steps / potential evaluations used for selection.')
    parser.add_argument('--potential_weight', default=5.0, type=float,
                      help='Exponential weight applied to the sampled potentials (0 -> uniform selection, inf -> minimum).')
    parser.add_argument('--potential_resolution', default=0, type=float,
                      help='Number of visited / fixation potential cells per degree. This influences how densly the field of view will be scanned and is thus indirectly proportional to the scanned area.')
    parser.add_argument('--fixation_potential_factor', default=5.0, type=float,
                      help='Scaling factor of the fixation potential.')
    parser.add_argument('--relaxation_rate', default=0.2, type=float,
                      help='Relaxation rate of the visited-activation over a duration of 1 second.')
    parser.add_argument('--sampling_frequency', default=100.0, type=float,
                      help='Sampling frequency of the simulated scanner in 1/sec.')
    parser.add_argument('--start_position_sigma', default=nan, type=float,
                      help='Start position is normally distributed around the center with specified standard deviation in degree. The default value nan uses field_size/8. 0 always selects the center, inf corresponds to uniform selection.')
    parser.add_argument('--potential_norm_exponent', default=1.0, type=float,
                      help='Exponent of the fixation potential\'s distance norm. Use 1.0 for squared L2 norm, and 0.5 for L2 norm.')
    parser.add_argument('--walk_along_axes', default='False', type=str,
                      help='Restrict the walking direction to the axes. step_candidates == 0 will do exactly one step in all 4 directions. Allows to emulate motion simulation on a grid when setting dist_sigma and potential_resolution to 0.')
    parser.add_argument('--random_seed', default=-1, type=int,
                      help='Random seed in the range [0, 2**32-1]. Use the default -1 for random initialization.')
    parser.add_argument('--sampling_start', default=0.0, type=float,
                      help='Start of time span in which sampling locations are computed, defaults to 0 (start of random walk). Sampling always starts at time 0 with constant frequency, not at sampling_start. This can be used to extract subranges or to avoid different walking behavior during the beginning of the random walk.')
    parser.add_argument('--sampling_duration', default=inf, type=float,
                      help='Duration in which sampling locations are computed.')
    parser.add_argument('--debug_colors', default='False', type=str,
                      help='Colorful curve. Note that line and sampling point colors are shifted, so if the curve color is similar at a crossing, look at the dots.')
    parser.add_argument('--step_through', default=0, type=int,
                      help='0: off, 2: include debug messages, 1: Show the randomly sampled locations with intensity corresponding to the potential-derived probability for each step during walking. (Note that the probability of sampling a certain location is not included in the intensity, however it is related to the sampling density at each location.)')
    parser.add_argument('--fpath_sim', default=None, type=str,
                      help='File path for writing the simulated step positions as .npy array or None for no output.')
    parser.add_argument('--fpath_sampled', default=None, type=str,
                      help='File path for writing the sampled curve positions as .npy array or None for no output.')
    parser.add_argument('--base_dir', default=None, type=str,
                      help='Changes the base directory, intended to specify a directory for writing the files.')
    parser.add_argument('--show_plots', default='True', type=str,
                      help='Should be turned off for high sampling rates.')
    parser.add_argument('--use_decimal', default=None, type=str,
                      help='Binary switch for using the Decimal type for exponential weight computations which provides a nearly unlimited exponent. Use when getting an overflow exception with ordinary double precision accuracy, slows down computation. For an integer value >= 2, specifies the number of used decimal digits, which is of minor relevance.')

    args = parser.parse_args()

    # some simple conversions including format checks
    try:
        args.debug_colors = str2bool(args.debug_colors, 'debug_colors')
        args.walk_along_axes = str2bool(args.walk_along_axes, 'walk_along_axes')
        args.show_plots = str2bool(args.show_plots, 'show_plots')
    except ValueError as e:
        die(e)
    convert_none = lambda x : None if x is None or x.lower() == 'none' else x
    args.fpath_sim = convert_none(args.fpath_sim)
    args.fpath_sampled = convert_none(args.fpath_sampled)
    args.base_dir = convert_none(args.base_dir)

    # check argument values & set dependant default values
    if args.field_size <= 0.0 or isinf(args.field_size):
        die('field_size must be positive and finite.')
    if args.duration <= 0.0 or isinf(args.duration):
        die('duration must be positive and finite.')
    if args.simulation_frequency <= 0.0 or isinf(args.simulation_frequency):
        die('simulation_frequency must be positive and finite.')
    if args.dist_mean < 0.0 or isinf(args.dist_mean):
        die('dist_mean must be non-negative and finite.')
    if args.abs_dist_min < 0.0 or isinf(args.abs_dist_min): # need to check before dist_sigma
        die('abs_dist_min must be non-negative and finite.')
    if args.abs_dist_max < args.abs_dist_min or args.abs_dist_max == 0:
        die('abs_dist_max must be positive and at least abs_dist_min.')
    if args.dist_sigma < 0.0:
        die('dist_sigma must be non-negative.')
    elif isinf(args.dist_sigma) and isinf(args.abs_dist_max):
        die('dist_sigma may only be infinity when dist_max is finite.')
    if isnan(args.dist_weight):
        args.dist_weight = args.potential_weight
    if args.dist_weight < 0.0:
        die('dist_weight must be non-negative.')
    if args.step_candidates < 0:
        die('step_candidates must be non-zero.')
    elif args.step_candidates == 0:
        if not args.walk_along_axes:
            die('step_candidates may only be zero if walk_along_axes is true.')
    if args.potential_weight < 0.0:
        warnings.warn('WARNING: A negative potential weight will prefer nodes with high activation thus turning the self-avoidance property upside down.')
    round_to_odd = lambda x : floor(x/2)*2+1 # always rounding downwards at the boundary!
    if args.potential_resolution < 0.0:
        die('potential_resolution must be positive or 0 for automatic adaption on mean_dist.')
    elif args.potential_resolution == 0.0:
        if args.dist_mean == 0.0:
            die('For automatic setup of potential_resolution, mean_dist must be positive.')
        N = round_to_odd(args.field_size / args.dist_mean)
        args.potential_resolution = N / args.field_size
    else:
        N = round_to_odd(args.field_size * args.potential_resolution)
    effective_potential_resolution = N / args.field_size
    print('effective potential_resolution:',effective_potential_resolution,'[1/°] (N = ' + str(N) +')')
    if args.fixation_potential_factor < 0.0 or isinf(args.fixation_potential_factor):
        die('fixation_potential_factor must be non-negative and finite.')
    if args.relaxation_rate < 0.0 or args.relaxation_rate > 1.0:
        die('relaxation_rate must be in the range [0.0, 1.0].')
    if args.sampling_frequency <= 0.0 or isinf(args.sampling_frequency):
        die('sampling_frequency must be positive and finite.')
    if isnan(args.start_position_sigma):
        args.start_position_sigma = args.field_size/8.0
    if args.start_position_sigma < 0.0:
        die('start_position_sigma must be non-negative.')
    if not (-1 <= args.random_seed and args.random_seed <= 2**32 - 1):
        die('random_seed must be in the range [-1,2**32-1].')
    if args.sampling_start < 0.0:
        die('sampling_start must be non-negative.')
    if args.sampling_duration < 0.0:
        die('sampling_duration must be non-negative.')
    if not (0 <= args.step_through and args.step_through <= 2):
        die('step_through must be in the range [0,2].')
    if args.use_decimal is not None:
        try:
            args.use_decimal = str2bool(args.use_decimal, 'use_decimal')
        except ValueError as e:
            try:
                decimal_digits = int(args.use_decimal)
                getcontext().prec = decimal_digits
                args.use_decimal = True
            except ValueError as e2:
                die(str(e) + ' or an integer number >= 2 specifying the precision in decimal digits (' + str(e2) + ')')

    return args, N


def die(msg : str) -> None:
    print(msg)
    exit()


def str2bool(val : str, argument_name : str) -> None:
    true_literals  = ['1', 'true']
    false_literals = ['0', 'false']
    is_true  = any(elem == val.lower() for elem in true_literals)
    is_false = any(elem == val.lower() for elem in false_literals)
    if is_true == is_false:
        raise ValueError(argument_name + ' must be any value in ["' + '", "'.join(true_literals + false_literals) + '"].') # this error message also triggers when val is in both lists...
    return is_true


def get_random_directions(position_x : float, position_y : float, min_bound_xy : float, max_bound_xy : float) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    :return: Four arrays of step_candidates randomly choosen angles and distances.
             random_x: New x values according to distance and angle.
             random_y: New y values according to distance and angle.
             angles: Angles in range of 0 to 2*pi
             distances: Sampled from a normal distance distribution.
    """

    #   Pick step_candidates random directions in radians
    if args.walk_along_axes:
        if args.step_candidates == 0:
            # walk in each direction once (useful if step size is fixed, i.e. dist_sigma == 0)
            angles = np.asarray([0.0, 0.5*np.pi, np.pi, 1.5*np.pi])
        else:
            angles = np.random.choice(np.asarray([0.0, 0.5*np.pi, np.pi, 1.5*np.pi]), args.step_candidates)
    else:
        angles = np.random.uniform(0.0, 2.0*np.pi, args.step_candidates)

    #   Pick R random distances
    if args.abs_dist_min == args.abs_dist_max: # special case fixed distance
        dists = np.full(angles.shape, args.abs_dist_max)
    elif args.dist_sigma == inf: # special case uniform distribution
        dists = np.random.uniform(args.abs_dist_min, args.abs_dist_max, angles.shape)
    elif args.dist_sigma == 0.0 or args.dist_weight == inf: # special case fixed distance
        dists = np.full(angles.shape, args.dist_mean)
    else:
        dists = np.random.normal(args.dist_mean, args.dist_sigma / sqrt(args.dist_weight), angles.shape)

    #   Mirror boundary for abs_dist_max as long as necessary
    if not isinf(args.abs_dist_max):
        while np.any(np.abs(dists) > args.abs_dist_max):
            dists[dists >  args.abs_dist_max] =  2*args.abs_dist_max - dists[dists >  args.abs_dist_max]
            dists[dists < -args.abs_dist_max] = -2*args.abs_dist_max - dists[dists < -args.abs_dist_max]

    #   "boundary condition" to ensure that spline construction works properly
    if args.abs_dist_min > 0.0:
        # clamp to nearest valid value (assuming that abs_dist_min is small and thus that the oversampling bias is negligable)
        dists[(0 <= dists) & (dists <  args.abs_dist_min)] =  args.abs_dist_min
        dists[(0 >  dists) & (dists > -args.abs_dist_min)] = -args.abs_dist_min

    #   Compute candidate positions
    random_x = position_x + dists * np.cos(angles)
    random_y = position_y + dists * np.sin(angles)

    #   field size boundary mirroring as long as necessary
    cond = np.any((random_x < min_bound_xy) | (max_bound_xy <= random_x) | (random_y < min_bound_xy) | (max_bound_xy <= random_y))
    global warned_mirror
    if not warned_mirror and cond:
        warned_mirror = True
        warnings.warn('Note: mirror boundary condition became effective. You may want to use a larger field size.')
    while cond:
        #print(cond.nonzero()[0].size) # number of remaining elements
        random_x[random_x < min_bound_xy] = min_bound_xy - (random_x[random_x < min_bound_xy] - min_bound_xy)
        random_x[random_x >= max_bound_xy] = max_bound_xy - (random_x[random_x >= max_bound_xy] - max_bound_xy)
        random_y[random_y < min_bound_xy] = min_bound_xy - (random_y[random_y < min_bound_xy] - min_bound_xy)
        random_y[random_y >= max_bound_xy] = max_bound_xy - (random_y[random_y >= max_bound_xy] - max_bound_xy)
        cond = np.any((random_x < min_bound_xy) | (max_bound_xy <= random_x) | (random_y < min_bound_xy) | (max_bound_xy <= random_y))

    return random_x, random_y, angles, dists


def convert_range(range_min : float, range_max : float, float_value : np.ndarray, N : int) -> np.ndarray:
    """
    :param range_min: Minimum value of the range.
    :param range_max: Minimum value of the range.
    :param float_value: Old value / array to convert.
    :param N: Array size.
    :return: Returns index / index-array for the closest element center(s) given the float range
    """
    range_span = (range_max - range_min)
    idx = np.floor(((float_value - range_min) / range_span) * N)

    #   Convert to integer
    idx = np.array(idx, dtype=int)
    idx[idx > N-1] = N-1

    return idx


def weighted_random_selection(weights : np.ndarray, potential_weight : float) -> (np.int64, np.ndarray):
    """
    :param weights: Weights from random points. Weight 0 is the best possible. No maximum value.
    :return: Index of the weighted-randomly selected entry.
    """

    #   Exponential weight with correct handling of inf values
    if potential_weight == 0.0:
        mod_weights = np.ones(weights.shape, dtype=float) # in case weights is 0, when 0^0 would be undefined, the result will be 1. There shouldn't be a weight with value exact 0 anyways...
    else:
        if potential_weight < 0.0:
            potential_weight = -potential_weight
            prev_err = np.geterr()
            np.seterr(divide='ignore')
            mod_weights = np.float64(1.0) / weights.astype(np.float64)
            np.seterr(divide=prev_err['divide'])
        else:
            mod_weights = weights

        if np.isinf(potential_weight) or np.any(np.isinf(mod_weights)):
            mod_weights = (mod_weights == np.max(mod_weights)).astype(float) # don't mess with a tolerance...
        else:
            if args.use_decimal:
                mod_weights = mod_weights.tolist()
                mod_weights = [Decimal(x) for x in mod_weights]
                mod_weights = np.array(mod_weights)

                mod_weights = mod_weights**Decimal(potential_weight)

                # need to normalize to be sure that the result fits into double, small values may get lost
                mod_weights = mod_weights / np.sum(mod_weights)
                mod_weights = mod_weights.astype(float)
            else:
                mod_weights = mod_weights**potential_weight

    cum_weights = np.cumsum(mod_weights)

    # Pick a number between 0 and the sum of all weights
    random_w = np.random.uniform(0.0, cum_weights[-1])

    # find corresponding index
    nonzeros = (random_w < cum_weights).nonzero()[0] # need to exclude equality here, otherwise, if random_w is zero, the first entry would get picked even if its weight is exact 0
    # handle the case random_w == cum_weights[-1]
    last_idx = mod_weights.shape[0] - 1
    idx = nonzeros[0] if nonzeros.size > 0 else np.int64(last_idx)

    return (idx, mod_weights)


def line_length(x : np.ndarray, y : np.ndarray) -> float:
    """
    :param x: X indices of walked locations.
    :param y: Y indices of walked locations.
    :return: Returns summed up difference of all euclidean lengths between neighboring points .
    """
    x_diff = x[0:-1] - x[1:]
    y_diff = y[0:-1] - y[1:]
    diff_norm = np.sqrt(x_diff * x_diff + y_diff * y_diff)
    return fsum(diff_norm)


def line_segment_length_std(x : np.ndarray, y : np.ndarray) -> float:
    x_diff = x[0:-1] - x[1:]
    y_diff = y[0:-1] - y[1:]
    diff_norm = np.sqrt(x_diff * x_diff + y_diff * y_diff)
    return np.std(diff_norm)

# Init
warned_mirror = False
min_line_weight = 0.25

#   Get parsed parameters
args, N = parse_args()

if args.base_dir is not None:
    chdir(args.base_dir)

if args.random_seed != -1:
    np.random.seed(args.random_seed)

#   Create a quadratic grid representing the eye with size NxN
visited_activation = np.zeros((N, N)) # i,j indexing

field_min = -args.field_size/2.0 # degree
field_max = args.field_size/2.0 # degree
field_size_extent = [field_min,field_max,field_min,field_max] # degree
cell_width = args.field_size/N # degree
cell_center_1d = np.linspace(field_min+0.5*cell_width, field_max-0.5*cell_width, N) # degree, note: from first to last cell _center_
cell_center_x, cell_center_y = np.meshgrid(cell_center_1d, cell_center_1d)

fixation_potential = lambda offset_x, offset_y : (args.fixation_potential_factor * (offset_x**2 + offset_y**2)**args.potential_norm_exponent)
fixation_potential_display = fixation_potential(cell_center_x, cell_center_y)

T = args.duration
f_sampling = args.sampling_frequency
f_sim = args.simulation_frequency

#   Ceil number of required simulation steps such that the Bspline is valid until _at least_ T
steps_sim = int(np.ceil(T * f_sim))
visited_points = np.zeros((1+steps_sim, 2)) # degree (+1 for the starting point)

#   Init random walk
if args.start_position_sigma == 0:
    visited_points[0, :] = np.full(2, 0.0)
elif isinf(args.start_position_sigma):
    visited_points[0, :] = np.random.uniform(field_min, field_max, 2)
else:
    visited_points[0, :] = np.random.normal(0, args.start_position_sigma, 2)
    visited_points[0, :] = np.clip(visited_points[0, :], field_min, field_max)
walker_x = visited_points[0, 0] # degree
walker_y = visited_points[0, 1] # degree
walker_i = convert_range(field_min, field_max, walker_y, N) # degree
walker_j = convert_range(field_min, field_max, walker_x, N) # degree
visited_activation[walker_i, walker_j] = 1.0
steps_dist = 0.0 # degree

"""
Random Walk
"""

for i in range(steps_sim):
    #   Get possible next locations
    rand_x, rand_y, angle, dist = get_random_directions(walker_x, walker_y, field_min, field_max)

    #   Convert float locations to potential grid indices
    rand_i = convert_range(field_min, field_max, rand_y, N)
    rand_j = convert_range(field_min, field_max, rand_x, N)
    walker_i = convert_range(field_min, field_max, walker_y, N)
    walker_j = convert_range(field_min, field_max, walker_x, N)

    #   Maximum potential within the linear path approximation with starting point excluded
    potential_sum = np.empty(4 if args.step_candidates == 0 else args.step_candidates)
    for idx in range(potential_sum.size):
        line_i, line_j, line_w = line_aa(walker_i, walker_j, rand_i[idx], rand_j[idx])

        # omit the current point when leaving it (otherwise intentionally don't, standing still should be costly!)
        # it likely dominates the potentials because it was most recently activated
        if line_i.size > 1:
            line_i = line_i[1:]
            line_j = line_j[1:]
            line_w = line_w[1:]

        # prefer thin lines, otherwise the walker is quite restricted
        # long term, by using a smooth activation increase function instead of a line and a better rule than max, this can be properly improved. for now we're stuck with the grid.
        line_i = line_i[line_w >= min_line_weight]
        line_j = line_j[line_w >= min_line_weight]

        potential_sum[idx] = np.max(visited_activation[line_i, line_j]) + fixation_potential(rand_x[idx], rand_y[idx])

    #   Decision weights are inverse to the sum of potentials as those describe where the walker shouldn't go
    prev_err = np.geterr()
    np.seterr(divide='ignore')
    weights = np.float64(1.0) / potential_sum.astype(np.float64) # note: inf possible in the exact center or if fixation_potential_factor is zero
    np.seterr(divide=prev_err['divide'])

    [selected_idx, mod_weights] = weighted_random_selection(weights, args.potential_weight)

    steps_dist = steps_dist + dist[selected_idx]

    #   Update new position
    walker_x = rand_x[selected_idx]
    walker_y = rand_y[selected_idx]
    visited_points[i+1, 0] = walker_x
    visited_points[i+1, 1] = walker_y

    #   Step visualization (do this before updating the activation so it reflects the values during decision)
    if args.step_through > 0 and args.sampling_start <= i/f_sim and i/f_sim < args.sampling_start + args.sampling_duration:
        if args.step_through > 1:
            sort_idces = np.argsort(angle)
            print(np.stack((np.rint(np.rad2deg(angle[sort_idces])), rand_x[sort_idces], rand_y[sort_idces], mod_weights[sort_idces])).T)
            #print(np.round(np.stack((np.rint(np.rad2deg(angle[sort_idces])), rand_x[sort_idces], rand_y[sort_idces], visited_activation[rand_i[sort_idces], rand_j[sort_idces]], fixation_potential(rand_i[sort_idces], rand_j[sort_idces]), potential_sum[sort_idces])).T, 3))
        fig, ax = plt.subplots()
        ax.imshow(np.flipud(visited_activation + fixation_potential_display), extent=field_size_extent)
        ax.plot(visited_points[0:i+1,0], visited_points[0:i+1,1], marker='o', color='#D95319') # simulated step positions (exclusive range-end!)
        color = np.log(1 + 3 * mod_weights / np.max(mod_weights))
        lower_bound = 0 # 0.25
        color[color < lower_bound] = lower_bound
        #color = 1.0-color
        ax.scatter(rand_x, rand_y, s=3, c=color, cmap=plt.get_cmap('gray'), norm=matplotlib.colors.NoNorm(), marker='o')
        plt.arrow(visited_points[i,0], visited_points[i,1], walker_x - visited_points[i,0], walker_y - visited_points[i,1], length_includes_head=True, zorder=2, width=0.0003, color='#D95319')
        ax.set_title('PDF approximation at t = ' + str(round(i/f_sim, 5)))
        ax.set_xlabel('Lateral fixation offset (°)')
        ax.set_ylabel('Longitudinal fixation offset (°)')
        ax.set_aspect('equal', adjustable='box')

        # zoom to walker's original location
        fov = 2.0*(args.dist_mean + 3.0*args.dist_sigma)
        ax.set_xlim([visited_points[i,0] - fov/2.0, visited_points[i,0] + fov/2.0])
        ax.set_ylim([visited_points[i,1] - fov/2.0, visited_points[i,1] + fov/2.0])

        plt.show()

    #   Get indices that lie on the line from start to most probable end point
    line_i, line_j, line_w = line_aa(walker_i, walker_j, rand_i[selected_idx], rand_j[selected_idx])

    # the current point was already stepped on in the previous iteration so don't increase again, unless the walker stays within that cell
    if line_i.size > 1:
        line_i = line_i[1:]
        line_j = line_j[1:]
        line_w = line_w[1:]

    # prefer thin lines, otherwise the walker is quite restricted
    # long term, by using a smooth activation increase function instead of a line and a better rule than max, this can be properly improved. for now we're stuck with the grid.
    line_i = line_i[line_w >= min_line_weight]
    line_j = line_j[line_w >= min_line_weight]

    #   Mask of (linear) path of traveled pixels (bspline curve is yet unknown)
    walked_mask = np.zeros(visited_activation.shape, dtype=bool)
    walked_mask[line_i, line_j] = True

    #   Update visited-activation
    visited_activation[~walked_mask] = visited_activation[~walked_mask] * (1.0 - args.relaxation_rate)**(1/f_sim)
    visited_activation[walked_mask] = visited_activation[walked_mask] + 1.0

"""
BSpline Interpolation
"""

x = visited_points[:, 0]
y = visited_points[:, 1]

# construct spline based on simulated steps
T_sim = steps_sim / f_sim # in the range [T, T + 1/f_sim)
t_sim = np.linspace(0, T_sim, len(visited_points))
spline_x = si.splrep(t_sim, x, k=3)
spline_y = si.splrep(t_sim, y, k=3)
#t_sim_idces = range(len(visited_points)) # use unit in indices for potentially faster spline construction -> TODO: is there a spline method that benefits from this?
#spline_x_idces = si.splrep(t_sim_idces, x, k=3)
#spline_y_idces = si.splrep(t_sim_idces, y, k=3)

# sample with sampling frequency of the scanner
sampling_end = args.sampling_start + args.sampling_duration
t_sampled = np.arange(0.0, T+1.0/f_sampling, 1.0/f_sampling) # extend the range a little bit to avoid rounding issues
t_sampled = t_sampled[(args.sampling_start <= t_sampled) & (t_sampled <= sampling_end)]
steps_sampling = len(t_sampled)-1 # the starting point is not a step / the number of intervals is 1 greater than the number of sampling points

if steps_sampling <= 0:
    T_sampling = 0
    x_sampled = np.ndarray(0)
    y_sampled = np.ndarray(0)
    samples_dist = nan
else:
    T_sampling = t_sampled[-1] - t_sampled[0]
    x_sampled = si.splev(t_sampled, spline_x)
    y_sampled = si.splev(t_sampled, spline_y)
    #t_sampled_sim_idces = t_sampled * f_sim # since the spline was constructed with time in unit 'simulation steps', convert the sampling times from scale seconds to simulation steps
    #x_sampled = si.splev(t_sampled_sim_idces, spline_x_idces)
    #y_sampled = si.splev(t_sampled_sim_idces, spline_y_idces)

    samples_dist = line_length(x_sampled, y_sampled)

"""
BSpline Length Approximation
"""

if steps_sampling <= 0:
    f_sampling_power = nan
    bspline_dist = nan
    bspline_dist_half = nan
else:
    dist_approx_change_bound = 1e-5

    # double approximation frequency until length change is below the bound
    dist_prev_approx = 0.0
    dist_finer_approx = samples_dist
    f_sampling_power = 0
    while dist_finer_approx - dist_prev_approx > dist_approx_change_bound: # note: finer approx. must be longer
        dist_prev_approx = dist_finer_approx
        f_sampling_power = f_sampling_power + 1

        #   Approximate distance of B-Spline with doubled sampling rate to estimate error bound
        t_sampled_e = np.arange(0.0, T+1.0/(f_sampling*2**f_sampling_power), 1.0/(f_sampling*2**f_sampling_power)) # extend the range a little bit to avoid rounding issues
        t_sampled_e = t_sampled_e[(args.sampling_start <= t_sampled_e) & (t_sampled_e <= sampling_end)]

        x_e = si.splev(t_sampled_e, spline_x)
        y_e = si.splev(t_sampled_e, spline_y)
        dist_finer_approx = line_length(x_e, y_e)

    bspline_dist = dist_finer_approx
    bspline_dist_half = dist_prev_approx
    bspline_velocity_std = line_segment_length_std(x_e, y_e) * f_sampling*2**f_sampling_power

"""
Summary Report
"""

print("Requested simulation duration:", T, "sec")
print("Effective simulation duration:", T_sim, "sec")
print("Effective sampling duration:", T_sampling, "sec")
print("Number of sampled locations (including start):", len(t_sampled))
print("Distance linear pathway:", steps_dist, "deg")
print("Length of B-Spline:", bspline_dist, "deg")
print("Mean distance between steps (linear):", steps_dist/steps_sim, "deg")
print("Mean distance between steps along B-Spline:", bspline_dist/steps_sim, "deg")
print("Mean distance between samples (linear):", samples_dist/steps_sampling, "deg")
print("Mean distance between samples along B-Spline:", bspline_dist/steps_sampling, "deg")
prev_err = np.geterr()
np.seterr(divide='ignore')
print("Mean velocity:", np.float64(bspline_dist)/np.float64(T_sampling), "deg/sec")
print("Stddev velocity:", bspline_velocity_std, "deg/sec")
np.seterr(divide=prev_err['divide'])
print("Length change compared to approximation with half frequency (f_sampling*2**" + str(f_sampling_power) + "):", bspline_dist_half - bspline_dist, "deg")

"""
Write step positions & curve to disk.
"""

if args.fpath_sim is not None:
    np.save(args.fpath_sim, np.stack((x, y), axis=1))

if args.fpath_sampled is not None:
    np.save(args.fpath_sampled, np.stack((x_sampled, y_sampled), axis=1))

"""
PLOT: Plotting the movement of the eye as well as the two potentials of the random walk
"""

if not args.show_plots:
    exit()

if steps_sampling <= 0:
    print('No sampled points.')
else:
    print("Last sampled point:", [x_sampled[-1], y_sampled[-1]])
print("Last simulated point:", visited_points[-1, :])

x_range_sampled = x[(args.sampling_start <= t_sim) & (t_sim <= sampling_end)]
y_range_sampled = y[(args.sampling_start <= t_sim) & (t_sim <= sampling_end)]

# sampled step positions
fig, ax = plt.subplots()
if args.debug_colors:
    # samples with both lines and points gradient-colored, but with different frequency such that the course of the line is evident in most situations
    colorline(x_sampled, y_sampled, np.mod(np.linspace(0.0, len(x_sampled)-1, len(x_sampled)), 100) / 100, cmap=plt.get_cmap('hsv'))
    ax.scatter(x_sampled, y_sampled, s=4.0**2, c=np.mod(np.linspace(0.0, len(x_sampled)-1, len(x_sampled)), 171) / 171, marker='o', cmap=plt.get_cmap('hsv'), zorder=2)
    #ax.plot(x, y, 'x',  marker='o', markersize=6.0) # simulated step positions
else:
    # regular samples
    #ax.plot(x_sampled, y_sampled, color='orange', markersize=4.0, marker='o')
    #ax.plot(x, y, 'x',  marker='o', markersize=6.0) # simulated step positions
    
    ax.plot(x_sampled, y_sampled, color='orange',  marker='o', markersize=2.0)
    ax.plot(x_range_sampled, y_range_sampled, 'x',  marker='o', markersize=3.0) # simulated step positions
ax.set_title('Course of viewing direction')
ax.set_xlabel('Lateral fixation offset (°)')
ax.set_ylabel('Longitudinal fixation offset (°)')
ax.set_aspect('equal', adjustable='box')
#plt.show()
#plt.savefig("Pathway.eps", format='eps')

# visited activation
fig, ax = plt.subplots()
ax.imshow(np.flipud(visited_activation), extent=field_size_extent)
ax.set_title('Visited activation')
ax.set_xlabel('Lateral fixation offset (°)')
ax.set_ylabel('Longitudinal fixation offset (°)')
ax.set_aspect('equal', adjustable='box')
#plt.show()
#plt.savefig("HeatMapWalk.eps", format='eps')

# fixation potential only
#fig, ax = plt.subplots()
#ax.imshow(np.flipud(fixation_potential_display), extent=field_size_extent)
#ax.set_title('Fixation potential')
#ax.set_xlabel('Lateral fixation offset (°)')
#ax.set_ylabel('Longitudinal fixation offset (°)')
#ax.set_aspect('equal', adjustable='box')
#plt.show()
#plt.savefig("HeatMapFoveal.eps", format='eps')

# summed potentials
fig, ax = plt.subplots()
ax.imshow(np.flipud(visited_activation + fixation_potential_display), extent=field_size_extent)
ax.set_title('Sum of potentials')
ax.set_xlabel('Lateral fixation offset (°)')
ax.set_ylabel('Longitudinal fixation offset (°)')
ax.set_aspect('equal', adjustable='box')
plt.show()
