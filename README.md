# sap-starterkit

<b>UPDATE (05/28/2021): We will host two office hours for any questions you have at
10-11pm Jun 1st (EDT)
[<a target="_blank" href="https://cmu.zoom.us/j/98380272281?pwd=QzFCU1gwS1UrL092MGt6djFZR1oxZz09">Zoom link</a>]
and 10-11am Jun 3rd (EDT)
[<a target="_blank" href="https://cmu.zoom.us/j/91673157392?pwd=NGx1NTBQNmNJQjBVQlBVRmFaUEJEZz09">Zoom link</a>].</b>
<br>

This repo is the starter kit for the [Streaming Perception Challenge](https://eval.ai/web/challenges/challenge-page/800/overview) to be held as part of the Workshop on Autonomous Driving at CVPR 2021. It contains a benchmark toolkit for measuring algorithm's performance and several baseline methods to get you started. The benchmark toolkit and the provided APIs used in the baselines aim to provide an abstraction over the streaming perception framework, which allows participants to solely focus on developing the algorithm itself and also ensures a fair competition for the challenge.

![benchmark-toolkit](doc/img/benchmark-toolkit.png)

The benchmark toolkit sends frames to the algorithm in real-time and queries for the state of the world at predefined time steps (determined by the framerate of the video). Therefore, unlike other benchmark toolkits (e.g., COCO toolkit) which run in a post-processing fashion, our toolkit is a daemon process that runs along side with the algorithm process.

## Getting started
  1. Follow the instructions [here](https://github.com/mtli/sAP/blob/master/doc/data_setup.md) to download and set up the dataset.
  1. Follow the instructions [here](doc/code_setup.md) to setup the required libraries.
  1. Run the toolkit and baselines:
       1. Use the script `run.sh` using either `bash run.sh det` or `bash run.sh forecast` to run the detection-only or detection+forecasting baselines respectively. Once this has completed running, a `results.json` file will be generated in the output directory.
       2. If annotations are available (e.g., for the val set), you can type `evaluate results.json` in the evaluation server CLI to evaluate locally.
       3. For the test set, evaluation can only be performed on the [challenge website](https://eval.ai/web/challenges/challenge-page/800/overview) by uploading the `results.json` file.

**Note:** You may also use two separate scripts to run the baselines as follows:
  1.  First, run `start_toolkit.sh`. This will launch the benchmark toolkit daemon locally with a command-line interface (CLI). You can type `?` for help.
  2.  Then, in another terminal run either `start_det_baseline.sh` (for detection-only baseline) or `start_fs_baseline.sh` (for detection+forecasting baseline) as required. 

Further documentation on the `sap_toolkit` can be found [here](https://github.com/karthiksharma98/sap-starterkit/tree/master/sap-toolkit). The scripts ending in `_perf.py` are variants that measure a few additional runtime metrics such as detection time, association time and forecasting time.

## Upload your results to the challenge website

Once you have written your algorithm and generated your output using the toolkit (results.json), you can upload it to the [challenge website](https://eval.ai/web/challenges/challenge-page/800/overview).
   
## Baseline Methods Explained

### Detection-only track:

This method uses the Mask-RCNN (with ResNet-FPN-50 backbone) single-frame object detector with a relative input scale of 1.0 (it accepts Argoverse-HD images at their original 1920x1200 resolution). It also incorporates the dynamic scheduling method described in the paper.

**Note:** For the detection-only track, we **don't allow any methods that use temporal reasoning**, such as forecasting, tracking or detectors that use a sequence of frames. We only allow single-frame detector methods. (scheduling methods that don't use temporal reasoning are allowed)

### Full-stack track:

This uses the same detector as above, however it incorporates a few techniques from the paper - dynamic scheduling, association and forecasting using a Kalman filter. This is explained in the figure below (taken from Fig. 7 from the paper).

![fig7](doc/img/fig7.png)

## Advanced topics for development

**Simulation**: Simulation allows for efficient stagewise development (detection first and then tracking and forecasting) and also accuracy-priortized development (since the runtime can be simulated). This concept is explained in the [streaming perception paper](http://www.cs.cmu.edu/~mengtial/proj/streaming/) and its accompanying repo [sAP](https://github.com/mtli/sAP).

**Visualization**: Visualization helps to identify issues and provides intuitions for innovation. Visual analysis of the baseline methods can be found on [here](http://www.cs.cmu.edu/~mengtial/proj/streaming/streaming-visuals.html) and how to produce such visualization can be found in the [sAP](https://github.com/mtli/sAP) repo.





