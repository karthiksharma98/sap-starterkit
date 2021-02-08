import torch
import argparse
import numpy as np
import json

# vision specific
from pycocotools.coco import COCO

# evaluation server utility functions
from sap_toolkit.client import EvalClient

# utility functions for mmdetection
from utils.mmdet import init_detector, inference_detector, parse_det_result

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--eval-config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    opts = parser.parse_args()
    return opts

def get_model(opts):
    model = init_detector(opts)
    model.eval()

    # warm up the GPU
    _ = inference_detector(model, np.zeros((1920, 1200, 3), np.uint8))
    torch.cuda.synchronize()
    return model

def run():
    # parse arguments
    opts = parse_args()

    # initialize model and mapping
    config = json.load(open(opts.eval_config, 'r'))

    # initialize annotations 
    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    
    # initialize model
    print("Loading model")
    model = get_model(opts)

    # initialize evaluation client 
    eval_client = EvalClient(config, verbose=True)

    # iterate over all sequences (videos)
    for sid, seq in enumerate(db.dataset['sequences']):

        # Request stream for current sequence from evaluation server
        eval_client.request_stream(seq)

        # get first frame - this is important as otherwise the while loop will not run
        fid, frame = eval_client.get_frame()
        fid_last = fid

        while frame is not None:

            # run model on current frame 
            result = inference_detector(model, frame, gpu_pre=not opts.cpu_pre)

            # bboxes, bbox_scores, labels are numpy arrays of dimensions Nx4, Nx1, Nx1   
            bboxes, bbox_scores, labels, _ = parse_det_result(result, coco_mapping, n_class)
            torch.cuda.synchronize()

            # send result to server
            # The evaluation server requires three inputs to send results
            # which are numpy arrays of dimensions Nx4, Nx1, Nx1 representing
            # bounding box coordinates, bounding box scores, and labels respectively
            # where N is the number of detections (100 detections maximum)
            eval_client.send_result_to_server(bboxes, bbox_scores, labels)

            # get next frame, wait if frame is same as last frame
            while fid == fid_last:
                fid, frame = eval_client.get_frame()
            fid_last = fid

        # stop current stream
        eval_client.stop_stream()
    
    # shut down evaluation client
    eval_client.close()

if __name__ == '__main__':
    run()

