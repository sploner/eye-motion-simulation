% make a composite image with original in red & displaced in cyan

fpath_in = '../paper-driftmotion/OS_3x3mm_500x500_vcsel_avg.png';
fpath_displaced = '../paper-driftmotion/OS_3x3mm_500x500_vcsel_avg_distorted.tiff';
fpath_out = '../composite.png';
rep_scans = 1; % if multiple repeated scan indices are given, intensities are averaged

I_org = imread(fpath_in);
I_disp = cast(read_tiff_stack(fpath_displaced), 'like', I_org);
I_disp = reshape(mean(I_disp(rep_scans,:,:), 1), size(I_org));

composite = zeros([size(I_disp) 3], 'like', I_org);
composite(:,:,1) = I_org; % red
composite(:,:,2) = I_disp; % cyan
composite(:,:,3) = I_disp;

imshow(composite);
imwrite(im2uint8(composite), fpath_out);
