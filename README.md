# Argoverse 2.0 Scene Flow Dataset
A set of scripts for working with sceneflow pseudolabels for Argoverse 2.0 Sensor

## Installing
- Download and set up the [Argoverse 2.0 Sensor dataset](https://github.com/argoverse/av2-api)
- Clone this repository
- In this directory run `python create.py --argo_dir <path/to/argo/sensor/> --output_dir <output/directory>`

## Evaluating
- Output your predictions as single Nx3 numpy arrays saved to a directory with the format <log_id>_<timestamp>.npy
- Run `python eval.py --git-dir <path/to/sceneflow/files> --pred-dir <path/to/predictions>`. This produces a file `results.parquet` which contains a pandas Dataframe with all the results. By default it breaks down the error metrics into Dynamic/Static and Foreground/Background classes, you can change the classes with the argument `--breakdown`.

