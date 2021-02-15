# sap_toolkit

This toolkit is provided as part of the Streaming Perception Challenge 2021. It includes an evaluation server module that takes care of reading frames from disk, streaming them to your application and receiving results from your application. It also generates the `.json` file required by the challenge evaluation server, and can perform local evaluation if annotations are provided in the COCO format.

## Installation

You can install the toolkit by using pip:

```
pip install sap_toolkit
```

## Usage:

The sap_toolkit comes with two main features:

*  A server module to stream frames, receive results and generate output.
*  A client object to communicate with the server module.

### Server:

The evaluation server can be launched using the following syntax:

```
python -m sap_toolkit.server \
	--data-root <path-to-data> \ 
	--annot-path <path-to-annotations> \
	--overwrite <overwrite-flag> \
	--out-dir <path-to-output-directory> \
	--eval-config <config-file> 
```

### Client/Application:

The sap_toolkit provides an EvalClient class that can be used to communicate with the evaluation server.

The API consists of the following  calls:

1. `sap_toolkit.EvalClient(config, state=None, verbose=True)`: Constructor for the EvalClient object. Can optionally accept a state object to initialize with a given state, which can be obtained by calling the `get_state()` method of another EvalClient object (useful in multiprocessing applications). 
2. `EvalClient.request_stream(seq)`: Requests the server to stream a particular sequence.
3. `EvalClient.get_frame(fid=None)`: If fid is None, returns the latest available (fid, frame) tuple. If fid is provided, returns the frame corresponding to that fid, if available.
4. `EvalClient.send_result_to_server(bboxes, scores, labels)`: Send a result to the evaluation server. This accepts three arguments of sizes Nx4, Nx1, Nx1, representing the bounding box co-ordinates (x1, y1, x2, y2) representing the bounding box left-top and right-bottom points, bounding box scores and labels.
5. `EvalClient.stop_stream()`: Requests server to processing current stream.
6. `EvalClient.close(filename='results.json')`: Closes the evaluation client object and generates the `results.json` output file. Should only be called when all processing is done.
7. `EvalClient.get_state()`: Returns the state of the current evaluation client object (which stores info such as currently processing stream and so on). Useful for multiprocessing applications.
8. `EvalClient.get_stream_start_time()`: Get the time the current stream's first frame was received.

For example applications using the sap_toolkit, check out the illustrative examples [here](https://github.com/karthiksharma98/sap-starterkit).
