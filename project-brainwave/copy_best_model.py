import os
import sys

indir = sys.argv[1]
outdir = sys.argv[2]

files = ['class_model_best.h5',
         'class_weights_best.h5',
         'resnet50_bw_best.data-00000-of-00001',
         'resnet50_bw_best.index', 
         'resnet50_bw_best.meta'
        ]

os.system('mkdir -p %s'%outdir)

for file in files:
    os.system('cp %s/%s %s/%s'%(indir, file, outdir, file.replace('_best','')))

with open('%s/checkpoint'%outdir,'w') as f:
    f.write('model_checkpoint_path: "resnet50_bw"\n')
    f.write('all_model_checkpoint_paths: "resnet50_bw"\n')
