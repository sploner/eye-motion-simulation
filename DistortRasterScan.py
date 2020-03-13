from argparse import ArgumentParser, Namespace
import numpy as np
from os import path
from scipy.interpolate import RectBivariateSpline
from skimage import io
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

"""
Purpose: apply an already simulated displacement field to a raster scan.
  The scan should be motion corrected already.

Advantage:
- Ground truth displacement is known and motion correction can be evaluated
  exactly.
- Evaluation geht viel schneller, weil nicht erst das segmentation-based
  measure, meist mit Layer-Segmentierung, berechnet werden muss...
    - (Wenn man nur ilm und superficial vessels macht bräuchte man nicht
      unbedingt was kompliziertes, funktioniert dann aber nicht so
      zuverlässig...)
    - (Erlaubt dadurch evtl auch eher lernbasierten Quatsch...)
  ===> solange man nur grob gute Parameter sucht ist das ok, solange man
       Kontrolliert, ob es sich übertragen lässt -> "additional evaluation
       on true clincal data indispensable

Disadvantages:
- To some degree, this is a chicken and egg problem. However, even if you use
  a motion correction algorithm as ground truth, that does not correct for drift
  motion, this data is usable to evaluate orthogonal scan based motion correc-
  tion, which should reproduce the distorted image. It is also possible to quan-
  tify / visualize the error that algorithms without distortion correction do not
  correct for. This would hovewer be a problem for e.g. lag bias motion correc-
  tions, which possibly could correct remaining distortion in the "ground truth".
- Eye movements are limited by the model accuracy
- Both the generation of the motion corrected and the retrospective distortion
  cause smoothing during the interpolation. Thus the images will be more
  blurred. Evaluation on clinical (raw-)data is thus indispensible, however
  it also has a downside: evaluation accuracy is limited to the precision of
  the error measure and it's not easily possible to derive a displacement
  error in µm.
- noise is not modeled and was usually reduced during ground truth creation.
"""

def parse_args() -> Namespace:
    parser = ArgumentParser('"scan_" is related to the input data, "disp_field_" is related to the previously simulated motion, "out_" is related to the output, everything else are parameters of the simulated raster scan.')

    parser.add_argument('--scan_path', type=str,
      help='En face image of the scan to be distorted or unspecified for GUI file selection dialog')
    parser.add_argument('--scan_field_width', type=float, default=mm2deg_approx(6.0),
      help='Horizontal field size of input scan in degree, default corresponds to approximately 6 mm in an eye with diameter 24 mm.')
    parser.add_argument('--scan_field_width_mm', type=float, default=None,
      help='Horizontal field size of input scan in mm. If specified, scan_field_width will be ignored.')
    parser.add_argument('--scan_field_height', type=float, default=mm2deg_approx(6.0),
      help='Vertical field size of input scan in degree, default corresponds to approximately 6 mm in an eye with diameter 24 mm.')
    parser.add_argument('--scan_field_height_mm', type=float, default=None,
      help='Vertical field size of input scan in mm. If specified, scan_field_height will be ignored.')
    parser.add_argument('--disp_field_path', type=str, default='XYCoordinates.npy',
      help='Previously simulated displacements as .npy file in degree, unspecified for GUI file selection dialog')
    parser.add_argument('--scan_dir', type=str, default='ru',
      help='[u]p / [d]own and [l]eft / [r]ight, fast / slow scan direction first / second')
    parser.add_argument('--samples_width', type=int, default=500,
      help='Horizontal number of samples in the simulated scan')
    parser.add_argument('--samples_height', type=int, default=500,
      help='Vertical number of samples in the simulated scan')
    parser.add_argument('--flyback', type=int, default=87,
      help='Number of flyback scans (number of ignored A-scans during flyback time)')
    parser.add_argument('--repeats', type=int, default=1,
      help='B-scan repeats (for angiography protocols)')
    parser.add_argument('--field_width', type=float, default=mm2deg_approx(6.0),
      help='Horizontal field size of the output scan in degree, default corresponds to approximately 6 mm in an eye with diameter 24 mm.')
    parser.add_argument('--field_width_mm', type=float, default=None,
      help='Horizontal field size of the output scan in mm. If specified, field_width will be ignored.')
    parser.add_argument('--field_height', type=float, default=mm2deg_approx(6.0),
      help='Vertical field size of the output scan in degree, default corresponds to approximately 6 mm in an eye with diameter 24 mm.')
    parser.add_argument('--field_height_mm', type=float, default=None,
      help='Vertical field size of the output scan in mm. If specified, field_height will be ignored.')
    parser.add_argument('--field_center_horz', type=float, default=0,
      help='Horizontal center of scan area w.r.t. the input scan')
    parser.add_argument('--field_center_vert', type=float, default=0,
      help='Vertical center of scan area w.r.t. the input scan')
    parser.add_argument('--out_path', type=str, default='displaced',
      help='Output path or empty for GUI dialog. If the format is not specified with a file ending, ".png" will be added for single slice results and ".tiff" otherwise.')
    parser.add_argument('--out_dtype', type=str,
      help='Output image element type as numpy dtype initializer str, defaults to input scan\'s element type. Note that currently value ranges are not adjusted, so this is currently only useful if you want float output from (u)int scan input.')
    parser.add_argument('--out_clamp_to_scan_dtype_range', type=str, default='False',
      help='Clamp.')
    parser.add_argument('--out_what', type=str, default=None,
      help='"dx" or "dy" prints the displacement field\'s x or y coordinate instead. Should use out_dtype "float32".')
    parser.add_argument('--suppress_warnings', type=str, default='False',
      help='Do not print any warnings to console.')

    args = parser.parse_args()

    # some simple conversions including format checks
    if args.out_dtype != None:
      args.out_dtype = np.dtype(args.out_dtype)
    args.suppress_warnings = str2bool(args.suppress_warnings, 'suppress_warnings')
    args.out_clamp_to_scan_dtype_range = str2bool(args.out_clamp_to_scan_dtype_range, 'out_clamp_to_scan_dtype_range')

    # value range checks
    if args.scan_field_width <= 0.0:
      die('scan_field_width must be positive')
    if args.scan_field_width_mm is not None:
      if args.scan_field_width_mm <= 0.0:
        die('scan_field_width_mm must be positive')
      args.scan_field_width = mm2deg_approx(args.scan_field_width_mm)
    if args.scan_field_height <= 0.0:
      die('scan_field_height must be positive')
    if args.scan_field_height_mm is not None:
      if args.scan_field_height_mm <= 0.0:
        die('scan_field_height_mm must be positive')
      args.scan_field_height = mm2deg_approx(args.scan_field_height_mm)
    if len(args.scan_dir) != 2 or not ("u" in args.scan_dir or "d" in args.scan_dir) or not ("l" in args.scan_dir or "r" in args.scan_dir):
      die('scan_dir has the wrong format.')
    if args.samples_width <= 0:
      die('samples_width must be positive')
    if args.samples_height <= 0:
      die('samples_height must be positive')
    if args.flyback < 0:
      die('flyback must be non-negative')
    if args.repeats < 1:
      die('repeats must be positive')
    if args.field_width <= 0.0:
      die('field_width must be positive')
    if args.field_width_mm is not None:
      if args.field_width_mm <= 0.0:
        die('field_width_mm must be positive')
      args.field_width = mm2deg_approx(args.field_width_mm)
    if not args.suppress_warnings and args.field_width > args.scan_field_width:
      print('WARNING: field_width should usually be at most scan_field_width, preferably less to avoid extrapolation for typical eye motion')
    if args.field_height <= 0.0:
      die('field_height must be positive')
    if args.field_height_mm is not None:
      if args.field_height_mm <= 0.0:
        die('field_height_mm must be positive')
      args.field_height = mm2deg_approx(args.field_height_mm)
    if not args.suppress_warnings and args.field_height > args.scan_field_height:
      print('WARNING: field_height should usually be at most scan_field_height, preferably less to avoid extrapolation for typical eye motion')
    if not args.suppress_warnings and args.field_center_horz < -args.scan_field_width/2.0 + args.field_width/2.0 or args.field_center_horz > args.scan_field_width/2.0 - args.field_width/2.0:
      print('WARNING: field_center_horz should usually be in the range [-scan_field_width/2.0 + field_width/2.0, scan_field_width/2.0 - field_width/2.0]')
    if not args.suppress_warnings and args.field_center_vert < -args.scan_field_height/2.0 + args.field_height/2.0 or args.field_center_vert > args.scan_field_width/2.0 - args.field_height/2.0:
      print('WARNING: field_center_vert should usually be in the range [-scan_field_height/2.0 + field_height/2.0, scan_field_height/2.0 - field_height/2.0]')
    if args.out_what is not None and not any(elem == args.out_what.lower() for elem in ['dx', 'dy']):
      print('out_what must be either "dx" or "dy".')

    return args

def die(msg : str) -> None:
  print(msg)
  exit()

def str2bool(val : str, argument_name : str) -> None:
  true_literals  = ['1', 'true']
  false_literals = ['0', 'false']
  is_true  = any(elem == val.lower() for elem in true_literals)
  is_false = any(elem == val.lower() for elem in false_literals)
  if is_true == is_false:
    die(argument_name + ' must be any value in ["' + '", "'.join(true_literals + false_literals) + '"].') # this error message also triggers when val is in both lists...
  return is_true

def request_paths(args : Namespace) -> Namespace:
  # request image path if unspecified
  if args.scan_path is None:
    args.scan_path = askopenfilename()
    if args.scan_path == '':
      die('Aborted.')

  # request displacement field path if unspecified
  if args.disp_field_path is None:
    args.disp_field_path = askopenfilename()
    if args.disp_field_path == '':
      die('Aborted.')

  # request out path if empty
  if args.out_path is None:
    args.out_path = asksaveasfilename()
    if args.out_path == '':
      die('Aborted.')

  return args

# assumes a perfectly round eye (not quite right!) with default radius of 12mm
# note that you need (to be precise: more than) twice the eye-radius for the beam angle whose center is in the cornea
def mm2deg_approx(mm : float, radius : float = 12.0) -> float:
  rad = mm / radius
  deg = rad / (2.0 * np.pi) * 360.0
  return deg

# assumes a perfectly round eye (not quite right!) with default radius of 12mm
# note that you need (to be precise: more than) twice the eye-radius for the beam angle whose center is in the cornea
def px2deg_approx(px : float, num_samples : int, field_size : float, radius : float = 12.0) -> float:
  mm = px / num_samples * field_size
  return mm2deg_approx(mm, radius)


# parse arguments
args = parse_args()

# don't show a main window
Tk().withdraw()

# request yet unspecified paths
args = request_paths(args)

# read input & convert to float32
scan_zyx = io.imread(args.scan_path)
scan_dtype = scan_zyx.dtype
if len(scan_zyx.shape) == 2: # depth axis missing
  scan_zyx = scan_zyx.reshape((1,) + scan_zyx.shape)
elif len(scan_zyx.shape) != 3:
  die('Please use a gray scale en face image or tiff stack.')
scan_zyx = scan_zyx.astype(np.float32)

# original pixel coordinates (in degree)
x_pos = np.linspace(-args.scan_field_width /2.0, args.scan_field_width /2.0, scan_zyx.shape[2])
y_pos = np.linspace(-args.scan_field_height/2.0, args.scan_field_height/2.0, scan_zyx.shape[1])

# scanner displacement field (in degree)
# note: this assumes that the pixels are uniformly spaced by the same angle
#       to be realistic we would need to know the position of the scan, i.e. central or off-center and the dewarping method that was applied to position the individual pixels
x_new_min = args.field_center_horz - args.field_width/2.0
x_new_max = args.field_center_horz + args.field_width/2.0
x_new = np.linspace(x_new_min, x_new_max, args.samples_width)
y_new_min = args.field_center_vert - args.field_height/2.0
y_new_max = args.field_center_vert + args.field_height/2.0
y_new = np.linspace(y_new_min, y_new_max, args.samples_height)
x_new_d, y_new_d = np.meshgrid(x_new, y_new) # parameters are usually flipped, but the y-axis is the first iamge dimension... TODO: not 100% sure, test
x_new_d = np.repeat(x_new_d.reshape((x_new_d.shape[0], 1, x_new_d.shape[1])), args.repeats, 1)
y_new_d = np.repeat(y_new_d.reshape((x_new_d.shape[0], 1, y_new_d.shape[1])), args.repeats, 1)

# expected (minimum) length of displacement field
is_xfast = args.scan_dir[0] in 'lr'
virtual_size = [args.samples_width + args.flyback * (1 if is_xfast else 0), args.repeats, args.samples_height + args.flyback * (0 if is_xfast else 1)]
total_samples = np.prod(virtual_size)

# motion displacement field (in degree)
debug = False
if not debug:
  # load simulated motion, i.e. displacements
  displacements = np.load(args.disp_field_path)

  # better error messages
  if len(displacements.shape) != 2 or (displacements.shape[1] != 2 and displacements.shape[1] != 3):
    die('Unexpected displacement field shape ' + str(displacements.shape) + '.')
  if displacements.shape[0] < total_samples:
    die('Displacement vector (' + str(displacements.shape) + ') is not long enough (' + str(total_samples) + '). Simulate with a longer scan duration.')

  # extract as many displacements as required
  dx = displacements[0:total_samples,0]
  dy = displacements[0:total_samples,1]
else:
  # note: in case you're wondering, behavior appears quite different in both axes depending on scan direction ;)
  width_px2deg_approx  = lambda px : px2deg_approx(px, args.samples_width,  args.field_width)
  height_px2deg_approx = lambda px : px2deg_approx(px, args.samples_height, args.field_height)

  # init
  dx = np.zeros(total_samples)
  dy = np.zeros(total_samples)

  # add / combine test motion patterns
  dx += width_px2deg_approx(5) * np.cos(np.linspace(0.0, 5.0 * (2.0*np.pi), total_samples))
  #dx += np.linspace( width_px2deg_approx(-0.1),  width_px2deg_approx(0.1), total_samples)
  dy  += np.linspace(height_px2deg_approx(-0.1), height_px2deg_approx(0.1), total_samples)

# remove flyback
dx = dx.reshape(virtual_size)[0:args.samples_height,:,0:args.samples_width] # looks weird, but the y-axis is the first image dimension...
dy = dy.reshape(virtual_size)[0:args.samples_height,:,0:args.samples_width]

# add displacements
print('Using', total_samples, '/', dx.size, 'samples with/without flyback of the', len(displacements), 'provided.')
x_new_d += dx
y_new_d += dy

# slicewise resampling, 1 slice per scan repeat
# note: increasing transverse voxel spacing with increasing depth is not modeled during interpolation
out = np.empty([args.repeats, scan_zyx.shape[0], args.samples_height, args.samples_width])
for z in range(scan_zyx.shape[0]):
  f = RectBivariateSpline(y_pos, x_pos, scan_zyx[z,:,:])
  for r in range(args.repeats):
    for y in range(args.samples_height):
      if args.out_what == 'dx':
        out[r,z,y,:] = dx[y,r,:]
      elif args.out_what == 'dy':
        out[r,z,y,:] = dy[y,r,:]
      else:
        out[r,z,y,:] = f.ev(y_new_d[y,r,:], x_new_d[y,r,:]) # actually i,j indexing for f

# add an appropriate file format specifier if not provided
if '.' not in path.basename(args.out_path):
  args.out_path += '.png' if scan_zyx.shape[0] == 1 else '.tiff'

# ensure file type is chosen appropriately
if args.out_path.endswith('.png'):
  if not scan_zyx.shape[0] == 1:
    die('PNG files only support single slice images. Use an out_path with ending ".tiff" instead.')

# bound output to scan_dtype's range if it was of an integer type
if args.out_clamp_to_scan_dtype_range and 'int' in scan_dtype.name:
  iinfo = np.iinfo(scan_dtype)
  out = np.maximum(out, iinfo.min)
  out = np.minimum(out, iinfo.max)

# find output element type
if args.out_dtype is None:
  out_dtype = scan_dtype
else:
  out_dtype = args.out_dtype

# bound output to out_dtype's range to avoid wrap-around of the interpolated values
if 'int' in out_dtype.name:
  iinfo = np.iinfo(out_dtype)
  out = np.maximum(out, iinfo.min)
  out = np.minimum(out, iinfo.max)

# save simulation result with requested element type
if args.repeats > 1 and out.shape[1] == 1 and args.out_path.endswith('.tiff'):
  # special case en face image with multiple repeats: write tiff stack of the scan repeats instead of individual files
  cur_repeat = out[:,0,:,:]
  io.imsave(args.out_path, cur_repeat.astype(out_dtype))
else:
  for r in range(args.repeats):
    cur_repeat = out[r,:,:,:]
    if args.out_path.endswith('.png'):
      cur_repeat = cur_repeat[0,:,:]
    out_path = args.out_path
    if args.repeats > 1:
      pos = args.out_path.rfind('.')
      if pos == -1:
        out_path = out_path + '_' + str(r)
      else:
        out_path = out_path[0:pos] + '_' + str(r) + out_path[pos:]
    io.imsave(args.out_path, cur_repeat.astype(out_dtype))
