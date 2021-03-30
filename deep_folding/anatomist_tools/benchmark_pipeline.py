# The aim of this script is to generate a benchmark of sulcal abnormalities.
# The script gathers all the steps necessary.

# Modules import
import os
import re
import json
import benchmark_generation


# Creation of altered skeleton images
b_num = 2
side = 'L'
region = 'S.T.s'
ss_size = 1000

directory = '/neurospin/dico/lguillon/mic21/anomalies_set/dataset/benchmark'+ str(b_num) + '/'
if not os.path.isdir(directory):
    os.mkdir(directory)
    os.mkdir(directory + '0_'+ side+ 'side/')

new_dir = directory + '0_'+ side+ 'side/'

benchmark_generation.generate(b_num, side, region, ss_size)

for img in os.listdir(new_dir):
    if '.nii.gz' in img and 'minf' not in img:
        sub = re.search('_(\d{6})', img).group(1)

        # Normalization and resampling of altered skeleton images
        dir_m = '/neurospin/dico/lguillon/skeleton/transfo_pre_process/natif_to_template_spm_' + sub +'.trm'
        dir_r = '/neurospin/hcp/ANALYSIS/3T_morphologist/' + sub + '/t1mri/default_acquisition/normalized_SPM_' + sub +'.nii'
        cmd_normalize = "AimsResample -i " + new_dir+ img + " -o " + new_dir + img[:-7] + "_normalized.nii.gz -m " + dir_m + " -r " + dir_r
        os.system(cmd_normalize)

        # Crop of the images
        file = new_dir + img[:-7] + "_normalized.nii.gz"
        cmd_crop = "AimsSubVolume -i " + file + " -o " + file + " -x 105 -y 109 -z 23 -X 147 -Y 171 -Z 93"
        os.system(cmd_crop)

input_dict = {'region': region, 'simple_surface_min_size': ss_size}
log_file = open(new_dir + "logs.json", "a+")
log_file.write(json.dumps(input_dict))
log_file.close()
