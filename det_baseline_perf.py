# vision specific
import torch
import argparse
import numpy as np
from pycocotools.coco import COCO
from time import perf_counter
import json

# utility functions for mmdetection
from utils.mmdet import init_detector, inference_detector, parse_det_result

from sap_toolkit.client import EvalClient

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

def detector_process():
    # initialize model and mapping
    opts = parse_args()
    config = json.load(open(opts.eval_config, 'r'))

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    
    print("Loading model")
    model = get_model(opts)

    eval_client = EvalClient(config, verbose=True)

    send_result_times = []
    get_frame_times = []
    get_frame_copy_times = []

    runtime_all = []

    for sid, seq in enumerate(db.dataset['sequences']):
        # Request stream from server
        
        eval_client.request_stream(seq)
        fid, frame = eval_client.get_frame()
        runtime = []

        while frame is not None:
            t_run1 = perf_counter()
            result = inference_detector(model, frame, gpu_pre=not opts.cpu_pre)
            bboxes, bbox_scores, labels, _ = parse_det_result(result, coco_mapping, n_class)
            torch.cuda.synchronize()

            t1 = perf_counter()
            eval_client.send_result_to_server(bboxes, bbox_scores, labels)
            send_result_times.append(perf_counter() - t1)

            t2 = perf_counter()
            fid, frame = eval_client.get_frame()
            get_frame_times.append(perf_counter() - t2)

            runtime.append(perf_counter() - t_run1)
        
        eval_client.stop_stream()
        print("Sent all results for seq ", seq)

        runtime_all += runtime

        runtime = np.asarray(runtime)
        runtime *= 1e3
        print('Runtime per frame (seq) ', seq, ' (ms): mean: %g; std: %g; min: %g; max: %g' % (
            runtime.mean(),
            runtime.std(ddof=1),
            runtime.min(),
            runtime.max(),
        ))

    runtime_all_np = np.asarray(runtime_all)
    n_small_runtime = (runtime_all_np < 1.0/30).sum()

    # convert to ms for display
    runtime_all_np *= 1e3
    print('Runtime (ms): mean: %g; std: %g; min: %g; max: %g' % (
        runtime_all_np.mean(),
        runtime_all_np.std(ddof=1),
        runtime_all_np.min(),
        runtime_all_np.max(),
    ))

    eval_client.close()


if __name__ == '__main__':
    detector_process()

