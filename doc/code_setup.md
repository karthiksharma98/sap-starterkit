# Code Setup

## Environment
You need an environment that has python>=3.8 and CUDA installed. 

## Installing dependencies

Most of the dependencies can be installed through this command with Conda environment. You might want to change the version for `cudatoolkit` in `environment.yml` to match your CUDA version <em>before</em> running it.

```
conda env create -f environment.yml
```

The created virtual environment is named `sap` and you can activate it by
```
conda activate sap
```

## Installing sap_toolkit

We provide a toolkit that performs the tasks of streaming frames and receiving output, and generating the .json file required by the [challenge website](https://eval.ai/web/challenges/challenge-page/800/overview).

You can install the toolkit by running:

```
pip install sap_toolkit
```

The next steps are only required if you wish to run the  baseline examples. For more details about the benchmark toolkit including its design and API, you can check out [this](https://github.com/karthiksharma98/sap-starterkit/tree/master/sap-toolkit) page. You can also check out the baseline examples in this repo to see how the toolkit API is used. 

## Installing dependencies to run baseline examples

To run the baseline examples, first mmcv-full and mmdetection must be installed.

The baseline example code has been tested to work with mmcv version v1.1.5 and mmdetection v2.7.0. Steps to install these are given below:

### mmcv installation

First, install [mmcv](https://github.com/open-mmlab/mmcv/tree/v1.1.5). See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions. We have tested our baseline code with mmcv v1.1.5, CUDA 10.2 and PyTorch 1.6.

The rule for installing a specified ``mmcv-full`` is as follows:

```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace `{cu_version}` and ``{torch_version}`` in the url to your desired one. For example,
to install the ``mmcv-full v1.1.5``  with ``CUDA 10.2`` and ``PyTorch 1.6.0``, use the following command:

```shell
pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html
```

More detailed instructions on mmcv installation can be found [here](https://github.com/open-mmlab/mmcv/blob/master/README.md).

### mmdetection installation

To install mmdetection, first clone the repo and checkout a certain version:
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout tags/v2.7.0 -b v2.7.0
```

Install mmdetection within the `sap` environment:
```
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```
This step will compile some CUDA and C++ files and might take some time. 

More detailed instructions on mmdetection installation can be found [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md).

## Prepare detection models

Download the pretrained model weights from mmdetection's [model zoo](https://github.com/open-mmlab/mmdetection/blob/v2.7.0/docs/model_zoo.md). Please use the link above (for mmdetection v2.7.0) to access the right version of the model zoo to avoid any compatibility issues. If you are using another version of mmdetection, make sure to use the correct model zoo.

We have used the Mask-RCNN model with R-FPN-50 backbone for our baseline examples. 
Links: 
- [config](https://github.com/open-mmlab/mmdetection/blob/v2.7.0/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py)
- [model weights](http://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_2x_coco/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth)


Note that Argoverse-HD is annotated according to COCO's format and class definitions. Therefore, it's reasonable to directly test out COCO pretrained models on Argoverse-HD.

## Modify paths

 You need to modify the paths for the dataset, model configuration and weights, and the output folder before running them. <em>Note that those scripts should be run from the root directory of this repo</em>. 

The parameters for the various scripts are explained below:

1. `start_server.sh`: This script is used to run the benchmark toolkit server process and must be run before running anything else. It initializes two services - an ImageService to stream images to your application and a ResultService to receive output. The various parameters are:

- `--data-root`: This provides the root directory of the dataset.
- `--annot-path`: This provides the annotations file. Annotations for the training and validation sets are provided (see dataset setup instructions [here](https://github.com/mtli/sAP/blob/master/doc/data_setup.md)). *Annotations for the test set will not be released. In this case, this parameter provides the file containing meta-info about the dataset in COCO format, i.e., a .json file similar to the validation and training annotations, but without the actual annotations*.
- `--overwrite`: Whether to overwrite existing output if present.
- `--out-dir`: Specifies the output directory. 
- `--eval-config`: (Optional) Specifies some networking parameters for the toolkit. Can be found in this repo as `config.json`.

    This script also launches a command-line interface (CLI) for the local toolkit server that supports the following options:

- `log`: View server log
- `evaluate <filename>`: If annotations are provided, run a local evaluation and generate evaluation metrics for the provided .json file with the given filename.
- `help`: show help

2. `start_client.sh` / `start_client_forecast.sh`: These scripts are used to run the illustrative examples for the detection-only and detection+forecasting settings respectively. The parameters are:

- `--config`: Provides the configuration file for the mmdetection model. Can be found in the mmdetection repo.
- `--weights`: Provides the weights for the mmdetection model. Can be downloaded from the [model zoo](https://github.com/open-mmlab/mmdetection/blob/v2.7.0/docs/model_zoo.md).
- `--in-scale`: Input scale for the mmdetection model. We use 1.0 for our baselines. This is the scale relative to the size of images in the Argoverse-HD dataset. So a scale of 1.0 refers to images with resolution 1920x1200 whereas a scale of 0.5 would be images resized to 960x600.
- `--annot-path`: Same as in start_server.sh described above.
- `--eval-config`: Same as in start_server.sh described above.


## Setup verification

If you have set up correctly, running `start_server.sh` followed by  `start_client.sh` and then typing `evaluate results.json` on the evaluation server CLI once the second script has finished running should be able to get you an AP of ~15.0 on the validation set: (This number may vary based on hardware. We obtain this baseline on a Tesla V100 GPU on an AWS p3.2x instance, for Mask RCNN FPN R50 with input scale 1.0).
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.149
```

## Other details

The file `forecasting_baseline.py` contains code which has our implementation of a Kalman filter for forecasting, and an implementation of dynamic scheduling from the Streaming perception paper. This uses a couple of other parameters (`--match-iou-th` and `--forecast-rt-ub`) which have been tuned for our detector baseline with <= 100 detections. These may need to modified if you are using a different model to achieve the best results.
