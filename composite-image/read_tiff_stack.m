% slices can be ommitted or specifiy the slices to be read.
function [I, fpath] = read_tiff_stack(fpath, slices)

if ~exist('fpath', 'var'), fpath = []; end
if ~exist('slices', 'var'), slices = []; end

if ~isempty(slices)
    assert(all(floor(slices) == slices), [...
        'Please make sure that the sceond parameter is an integer array ' ...
        'containging the slice numbers that shall be read. An old '...
        'implementation had a boolean normalization parameter at this '...
        'place, which can be achieved by wrapping the call to '...
        'read_tiff_stack by im2double().']);

    assert(isvector(slices), ...
        'The specified range of slices is not a vector.');
end

if isempty(fpath) || exist(fpath, 'dir')
    [fname, dir] = uigetfile('*.tiff', 'Select the X-/Y-fast volume for preprocessing', fpath);
    if (isscalar(fname) && fname == 0) || (isscalar(dir) && dir == 0), disp('Aborted.'); I = []; fname = []; return; end
    fpath = [dir fname];
end

% read tiff stack into I
info = imfinfo(fpath);

if ~strcmp(info(1).Format, 'tif')
    % not a tiff image, try standard read
    assert(isempty(slices) || slices == 1, ...
        'For non-Tif-filetypes, only the first slice can be read.');

    I = imread(fpath);
else
    % tiff images
    num_images = numel(info);
    
    % Default slices to full stack (only possible after knowing its size)
    if isempty(slices), slices = 1:num_images; end
    num_slices = numel(slices);

    % TODO: generate volume using the filetype from info instead of reading
    % a whole slice for nothing
    A = imread(fpath, 1, 'Info', info);
    I = zeros([num_slices size(A)], 'like', A); % preserve file's datatype
    for i = 1:num_slices
        sliceNr = slices(i);
        slice = imread(fpath, sliceNr, 'Info', info);
        I(i, :, :, :) = slice;
    end
end

end
