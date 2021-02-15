# Vision specific
from PIL import Image
from pycocotools.cocoeval import COCOeval

import numpy as np
import json
import pickle
import os
from tqdm import tqdm
from os.path import join

FPS = 30

def mkdir2(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def imread(path, method='PIL'):
    return np.array(Image.open(path))

def gen_results(db, opts, filename):

    def ltrb2ltwh_(bboxes):
        if len(bboxes):
            if bboxes.ndim == 1:
                bboxes[2:] -= bboxes[:2]
            else:
                bboxes[:, 2:] -= bboxes[:, :2]
        return bboxes

    def ltrb2ltwh(bboxes):
        bboxes = bboxes.copy()
        return ltrb2ltwh_(bboxes)
        
    seqs = db.dataset['sequences']
    print('Merging and converting results')
    results_ccf = []
    miss = 0

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        results = pickle.load(open(join(opts.out_dir, seq + '.pkl'), 'rb'))
        results_parsed = results['results_parsed']
        timestamps = results['timestamps']

        tidx_p1 = 0
        for ii, img in enumerate(frame_list):
            # pred, gt association by time
            t = ii/FPS
            while tidx_p1 < len(timestamps) and timestamps[tidx_p1] <= t:
                tidx_p1 += 1
            if tidx_p1 == 0:
                # no output
                miss += 1
                bboxes, scores, labels  = [], [], []
                masks, tracks = None, None
            else:
                tidx = tidx_p1 - 1
                result = results_parsed[tidx]
                bboxes, scores, labels, masks = result[:4]
                if len(result) > 4:
                    tracks = result[4]
                else:
                    tracks = None
                
            # convert to coco fmt
            n = len(bboxes)
            if n:
                # important: must create a copy, it is used in subsequent frames
                bboxes_ltwh = ltrb2ltwh(bboxes)
            
            for i in range(n):
                result_dict = {
                    'image_id': int(img['id']),
                    'bbox': [float(a) for a in bboxes_ltwh[i]],
                    'score': float(scores[i]),
                    'category_id': int(labels[i]),
                }
                if masks is not None:
                    result_dict['segmentation'] = masks[i]
                results_ccf.append(result_dict)

    out_path = join(opts.out_dir, filename)
    if opts.overwrite or not isfile(out_path):
        json.dump(results_ccf, open(out_path, 'w'))

def evaluate(db, out_dir, filename, overwrite=False):
    results_ccf = join(out_dir, filename)
    eval_summary = eval_ccf(db, results_ccf)
    out_path = join(out_dir, 'eval_summary.pkl')
    if overwrite or not isfile(out_path):
        pickle.dump(eval_summary, open(out_path, 'wb'))

def eval_ccf(db, results, iou_type='bbox'):
    # ccf means CoCo Format
    if isinstance(results, str):
        if results.endswith('.pkl'):
            results = pickle.load(open(results, 'rb'))
        else:
            results = json.load(open(results, 'r'))
   
    results = db.loadRes(results)
    cocoEval = COCOeval(db, results, iou_type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return {
        'eval': cocoEval.eval,
        'stats': cocoEval.stats,
    }