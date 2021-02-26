import argparse, json, pickle

import os, sys
from os.path import join, isfile, basename
from glob import glob
from time import perf_counter
import multiprocessing as mp
import traceback
from tqdm import tqdm

# setting environment variables - we have noticed that
# numpy creates extra threads which slows down computation,
# these environment variables prevent that
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"

import numpy as np
import torch

from pycocotools.coco import COCO

# utility functions for mmdetection
from utils import print_stats
from utils.mmdet import init_detector, inference_detector, parse_det_result

# utility functions to forecast
from utils.forecast import ltrb2ltwh_, ltwh2ltrb_, iou_assoc, extrap_clean_up, \
    bbox2z, bbox2x, x2bbox, make_F, make_Q, \
    batch_kf_predict_only, batch_kf_predict, \
    batch_kf_update

# multiprocessing
import multiprocessing as mp

# benchmark toolkit API
from sap_toolkit.client import EvalClient

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot-path', type=str, required=True)
    # 30 is the fps of the stream received from evaluation server, don't change this
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--eta', type=float, default=0, help='eta >= -1') # frame

    parser.add_argument('--eval-config', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)    
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    
    parser.add_argument('--dynamic-schedule', action='store_true', default=False)
    parser.add_argument('--perf-factor', type=float, default=1)

    parser.add_argument('--match-iou-th', type=float, default=0.3)
    parser.add_argument('--forecast-rt-ub', type=float, default=0.003) # seconds
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def det_process(opts, frame_recv, det_res_send, w_img, h_img, config, client_state):

    try:
        # load model
        model = init_detector(opts)

        # warm up the GPU
        _ = inference_detector(model, np.zeros((h_img, w_img, 3), np.uint8))
        torch.cuda.synchronize()

        eval_client = EvalClient(config, state=client_state, verbose=False)
        
        while 1:
            fidx = frame_recv.recv()
            if fidx == 'wait_for_ready':
                det_res_send.send('ready')
                continue
            
            if fidx is None:
                # exit flag
                break

            _, img = eval_client.get_frame(fid=fidx)

            t1 = perf_counter()
            result =  inference_detector(model, img, gpu_pre=not opts.cpu_pre)
            torch.cuda.synchronize()
            det_res_send.send([result, perf_counter() - t1])

    except Exception:
        # report all errors from the child process to the parent
        # forward traceback info as well
        det_res_send.send(Exception("".join(traceback.format_exception(*sys.exc_info()))))


def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing
    opts = parse_args()

    # initialize model config and mapping
    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    config = json.load(open(opts.eval_config, 'r'))

    img = db.imgs[0]
    w_img, h_img = img['width'], img['height']

    # initialize evaluation client 
    eval_client = EvalClient(config, verbose=False)

    # launch detector process
    frame_recv, frame_send = mp.Pipe(False)
    det_res_recv, det_res_send = mp.Pipe(False)
    det_proc = mp.Process(target=det_process, args=(opts, frame_recv, det_res_send, w_img, h_img, config, eval_client.get_state()))
    det_proc.start()

    # dynamic scheduling
    if opts.dynamic_schedule:
        # initialize runtime mean to 0.0
        mean_rtf = 0

    with torch.no_grad():

        # initialize Kalman filter parameters
        kf_F = torch.eye(8)
        kf_F[3, 7] = 1
        kf_Q = torch.eye(8)
        kf_R = 10*torch.eye(4)
        kf_P_init = 100*torch.eye(8).unsqueeze(0)

        for seq in tqdm(seqs):
            # Request stream for current sequence from evaluation server
            eval_client.request_stream(seq)

            fidx = 0
            processing = False  
            fidx_t2 = None            # detection input index at t2
            fidx_latest = None
            tkidx = 0                 # track starting index
            kf_x = torch.empty((0, 8, 1))
            kf_P = torch.empty((0, 8, 8))
            n_matched12 = 0

            # let detector process get ready to process sequence 
            # it is possible that unfetched results remain in the pipe, this asks the detector to flush those
            frame_send.send('wait_for_ready')
            while 1:
                msg = det_res_recv.recv() # wait till the detector is ready
                if msg == 'ready':
                    break
                elif isinstance(msg, Exception):
                    raise msg

            t_unit = 1/opts.fps

            # get the time when stream's first frame was received
            t_start = eval_client.get_stream_start_time()

            count_detections = 0
            while fidx is not None:
                t1 = perf_counter()
                t_elapsed = t1 - t_start

                # identify latest available frame
                fidx, _ = eval_client.get_frame()

                if fidx is None:
                    break

                # for dynamic scheduling
                fidx_continous = t_elapsed*opts.fps

                if fidx == fidx_latest:
                    # algorithm is fast and has some idle time
                    wait_for_next = True
                else:
                    wait_for_next = False
                    # for dynamic scheduling
                    if opts.dynamic_schedule:
                        if mean_rtf >= 1: # when runtime < 1, it should always process every frame
                            fidx_remainder = fidx_continous - fidx
                            if mean_rtf < np.floor(fidx_remainder + mean_rtf):
                                # wait till next frame
                                wait_for_next = True

                if wait_for_next:
                    continue

                # send frame to detector process
                if not processing:
                    frame_send.send(fidx)
                    fidx_latest = fidx
                    processing = True
                
                # wait till query - forecast-rt-ub to check if detector process has finished
                wait_time = t_unit - opts.forecast_rt_ub
                if det_res_recv.poll(wait_time): # wait
                    # new result received
                    result, t_det = det_res_recv.recv() 
                    if opts.dynamic_schedule:
                        sum_rtf = mean_rtf*count_detections + t_det*opts.fps
                        count_detections += 1
                        mean_rtf = sum_rtf/count_detections
                    
                    if isinstance(result, Exception):
                        raise result
                    bboxes_t2, scores_t2, labels_t2, _ = \
                        parse_det_result(result, coco_mapping, n_class)

                    processing = False
                   
                    # associate across frames
                    if len(kf_x):
                        dt = fidx_latest - fidx_t2

                        kf_F = make_F(kf_F, dt)
                        kf_Q = make_Q(kf_Q, dt)
                        kf_x, kf_P = batch_kf_predict(kf_F, kf_x, kf_P, kf_Q)
                        bboxes_f = x2bbox(kf_x)
                                        
                    fidx_t2 = fidx_latest
                    n = len(bboxes_t2)
                    if n:
                        # put high scores det first for better iou matching
                        score_argsort = np.argsort(scores_t2)[::-1]
                        bboxes_t2 = bboxes_t2[score_argsort]
                        scores_t2 = scores_t2[score_argsort]
                        labels_t2 = labels_t2[score_argsort]

                        ltrb2ltwh_(bboxes_t2)

                    updated = False
                    if len(kf_x):
                        order1, order2, n_matched12, tracks, tkidx = iou_assoc(
                            bboxes_f, labels, tracks, tkidx,
                            bboxes_t2, labels_t2, opts.match_iou_th,
                            no_unmatched1=True,
                        )

                        if n_matched12:
                            kf_x = kf_x[order1]
                            kf_P = kf_P[order1]
                            kf_x, kf_P = batch_kf_update(
                                bbox2z(bboxes_t2[order2[:n_matched12]]),
                                kf_x,
                                kf_P,
                                kf_R,
                            )
                    
                            kf_x_new = bbox2x(bboxes_t2[order2[n_matched12:]])
                            n_unmatched2 = len(bboxes_t2) - n_matched12
                            kf_P_new = kf_P_init.expand(n_unmatched2, -1, -1)
                            kf_x = torch.cat((kf_x, kf_x_new))
                            kf_P = torch.cat((kf_P, kf_P_new))
                            labels = labels_t2[order2]
                            scores = scores_t2[order2]
                            updated = True

                    if not updated:
                        # start from scratch
                        kf_x = bbox2x(bboxes_t2)
                        kf_P = kf_P_init.expand(len(bboxes_t2), -1, -1)
                        labels = labels_t2
                        scores = scores_t2
                        tracks = np.arange(tkidx, tkidx + n, dtype=np.uint32)
                        tkidx += n

                # apply forecasting for the current query
                query_pointer = fidx + opts.eta + 1
                
                if len(kf_x):
                    dt = (query_pointer - fidx_t2)

                    kf_x_np = kf_x[:, :, 0].numpy()
                    bboxes_t3 = kf_x_np[:n_matched12, :4] + dt*kf_x_np[:n_matched12, 4:]
                    if n_matched12 < len(kf_x):
                        bboxes_t3 = np.concatenate((bboxes_t3, kf_x_np[n_matched12:, :4]))
                        
                    bboxes_t3, keep = extrap_clean_up(bboxes_t3, w_img, h_img, lt=True)
                    labels_t3 = labels[keep]
                    scores_t3 = scores[keep]
                else:
                    bboxes_t3 = np.empty((0, 4), dtype=np.float32)
                    scores_t3 = np.empty((0,), dtype=np.float32)
                    labels_t3 = np.empty((0,), dtype=np.int32)
                
                if len(bboxes_t3):
                    ltwh2ltrb_(bboxes_t3)

                # send result to benchmark toolkit
                if fidx_t2 is not None:
                    eval_client.send_result_to_server(bboxes_t3, scores_t3, labels_t3)

            print("Stopping stream ", seq)
            eval_client.stop_stream()

    # shut down evaluation client
    eval_client.close()

    # terminates the child process
    frame_send.send(None)

if __name__ == '__main__':
    main()