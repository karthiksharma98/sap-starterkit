# sap-starterkit
Starter Kit for the Streaming Perception Challenge 2021

This repo contains the starter kit for the Streaming Perception challenge to be held as part of the Workshop on Autonomous Driving at CVPR 2021. 

## Getting Started
  1. Follow the instructions [here](https://github.com/mtli/sAP/blob/master/doc/data_setup.md) to download and set up the dataset.
  2. Follow the instructions [here](doc/code_setup.md) to setup the required libraries.
  3.  Run the scripts:
       - First, run `start_server.sh`. This will launch the evaluation server instance locally with a command-line interface (CLI). You can type `?` for help.
       -  Then, run either `start_client.sh` (for detection-only baseline) or `start_client_forecast.sh` (for detection+forecasting baseline) as required. Once this has completed running, a `results.json` file will be generated in the output directory. If annotations are available (for e.g. validation set), you can type `evaluate results.json` in the evaluation server CLI to evaluate locally. For test set, evaluation can only be performed on the [evaluation server]() by uploading the `results.json` file.

## Explore the illustrative examples:

The illustrative examples (also used to obtain the baseline numbers on the leaderboard for the challenge) show you how to use the API provided by our `sap_toolkit` to communicate with the evaluation server. You can take a look at the python code provided in `det_baseline.py` and `forecast_baseline.py` to see our implementations. There are also versions of these scripts (ending in `_perf.py`) that measure a few runtime metrics such as detection time, association time and forecasting time.

Further documentation on the `sap_toolkit` can be found [here](https://github.com/karthiksharma98/sap-toolkit)


## Upload your results to the evaluation server:

Once you have written your algorithm and generated your output on the test set using the toolkit, you can upload it to the [evaluation server]().
   






