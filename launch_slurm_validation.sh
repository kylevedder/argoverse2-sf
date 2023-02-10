srun --ntasks=1 --mem-per-cpu=2G --cpus-per-task=64 --time=03:00:00 --job-name=argosf --container-mounts=../../datasets/:/efs/,`pwd`:/project --container-image=kylevedder/argoverse2_sf:latest bash -c "OMP_NUM_THREADS=1 python create.py --argo_dir /efs/argoverse2/val --output_dir /efs/argoverse2/val_sceneflow/ --skip_pcs --skip_reverse_flow"