# Code Setup

## Environment
You need a Linux environment that has python>=3.8 and CUDA installed.

## Installing benchmark toolkit

We provide a benchmark toolkit for streaming evaluation, which serves the image frames, collects algorithm outputs, and generating results for leaderboard submission.

You can simply install it through `pip`:

```
pip install sap_toolkit
```
**If you plan to run the baselines, it's recommended to create a conda environment first as described in the next section.**

For more details about the benchmark toolkit including its design and API, you can check out [this](https://github.com/karthiksharma98/sap-starterkit/tree/master/sap-toolkit) page. You can also check out the baseline examples in this repo to see how the toolkit API is used. 

## Installing baseline dependencies (not required for the benchmark)

Most of the dependencies can be installed through this command with Conda environment. You might want to change the version for `cudatoolkit` in `environment.yml` to match your CUDA version <em>before</em> running it.

```
conda env create -f environment.yml
```

The created virtual environment is named `sap` and you can activate it by
```
conda activate sap
```

The next step is to manually install `mmcv` and `mmdetection` *within the `sap` environment*. (Make sure `sap_toolkit` has also been installed within the environment). The baseline code has been tested with `mmcv` v1.1.5 and `mmdetection` v2.7.0 and compatibility with other versions is not tested.

### mmcv installation

The command for installing `mmcv` is as follows:

```shell
pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace `{cu_version}` and ``{torch_version}`` with the versions you are currently using.
You will get import or runtime errors if the versions are incorrect.

For example, with ``CUDA 10.2`` and ``PyTorch 1.6.0``, you can use the following command:

```shell
pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
```
You should see that `pip` is downloading *a pre-compiled wheel file*:
```
Collecting mmcv-full==1.1.5
  Downloading https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/mmcv_full-1.1.5-cp38-cp38-manylinux1_x86_64.whl (18.5 MB)
```

If `pip` downloads a tar file:
```
Collecting mmcv-full==1.1.5
  Downloading mmcv-full-1.1.5.tar.gz (239 kB)
```
that means `mmcv` has not been compiled for your specific configuration.
We recommended you to change your CUDA or PyTorch versions.
Otherwise, you will need to compile `mmcv` from source to enable its CUDA components.

More information on `mmcv` installation can be found on their [Github page](https://github.com/open-mmlab/mmcv/).

### mmdetection installation

To install mmdetection, first clone the repo and checkout a specific version:
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout tags/v2.7.0 -b v2.7.0
```

Then run:
```
pip install -v -e .  # or "python setup.py develop"
```

More information on `mmdetection` installation can be found their [Github page](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md).

## Prepare detection models

Download the pretrained model weights from mmdetection's [model zoo](https://github.com/open-mmlab/mmdetection/blob/v2.7.0/docs/model_zoo.md). Please use the link above (for mmdetection v2.7.0) to access the right version of the model zoo to avoid any compatibility issues. If you are using another version of mmdetection, make sure to use the correct model zoo.

We have used the Mask R-CNN model with ResNet 50 (FPN) backbone for our baseline examples. 
Links: 
- [config](https://github.com/open-mmlab/mmdetection/blob/v2.7.0/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py)
- [model weights](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth)

Note that since the benchmark dataset Argoverse-HD is annotated according to COCO's format and class definitions, it's reasonable to directly test out COCO pretrained models on Argoverse-HD.

## Modify paths

 You need to modify the paths for the dataset, model configuration and weights, and the output folder before running them. *Note that those scripts should be run from the root directory of this repo*. 

The parameters for the various scripts are explained below:

1. `start_toolkit.sh`: This script is used to run the benchmark toolkit daemon process and must be run before running anything else. It initializes two processes - an ImageService process to stream images to your application and a ResultService process to receive output from your application. The various parameters are:

- `--data-root`: This provides the root directory of the dataset.
- `--annot-path`: This provides the annotations file. Annotations for the training and validation sets are provided (see dataset setup instructions [here](https://github.com/mtli/sAP/blob/master/doc/data_setup.md)). *Annotations for the test set will not be released. In this case, this parameter provides the file containing meta-info about the dataset in COCO format, i.e., a .json file similar to the validation and training annotations, but without the actual annotations*.
- `--overwrite`: Whether to overwrite existing output if present.
- `--out-dir`: Specifies the output directory. 
- `--eval-config`: (Optional) Specifies some networking parameters for the toolkit. Can be found in this repo as `config.json`.
- `--perf-factor`: (Set by default to 1.0 if not provided). Provides a runtime scaling factor to compensate for difference in hardware. Refer to [this](https://docs.google.com/spreadsheets/d/1g5jUAbeNswO-EQHrU9EHYpTT2XYCbepeL9KFEAVehMo/edit#gid=0) spreadsheet for a list of performance conversion factors based on your GPU.

    This script also launches a command-line interface (CLI) that supports the following options:

- `log`: View server log
- `evaluate <filename>`: If annotations are provided, run a local evaluation and generate evaluation metrics for the provided .json file with the given filename.
- `help`: show help

2. `start_det_baseline.sh` / `start_fs_baseline.sh`: These scripts are used to run the illustrative examples for the detection-only and detection+forecasting settings respectively. The parameters are:

- `--config`: Provides the configuration file for the mmdetection model. Can be found in the mmdetection repo.
- `--weights`: Provides the weights for the mmdetection model. Can be downloaded from the [model zoo](https://github.com/open-mmlab/mmdetection/blob/v2.7.0/docs/model_zoo.md).
- `--in-scale`: Input scale for the mmdetection model. We use 1.0 for our baselines. This is the scale relative to the size of images in the Argoverse-HD dataset. So a scale of 1.0 refers to images with resolution 1920x1200 whereas a scale of 0.5 would be images resized to 960x600.
- `--annot-path`: Same as in start_toolkit.sh described above.
- `--eval-config`: Same as in start_toolkit.sh described above.
- `--perf-factor`: Same as in start_toolkit.sh described above.


## Setup verification

If you have set up correctly, running `start_toolkit.sh` followed by  `start_det_baseline.sh` and then typing `evaluate results.json` on the benchmark toolkit CLI once the second script has finished running should be able to get you an AP of ~15.5 on the validation set: (This number may vary based on hardware. We obtain this baseline on a Tesla V100 GPU on an AWS p3.2x instance, for Mask RCNN FPN R50 with input scale 1.0).
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.155
```

## Other details

The file `forecasting_baseline.py` contains code which has our implementation of a Kalman filter for forecasting, and an implementation of dynamic scheduling from the Streaming perception paper. This uses a couple of other parameters (`--match-iou-th` and `--forecast-rt-ub`) which have been tuned for our detector baseline with <= 100 detections. These may need to modified if you are using a different model to achieve the best results.
