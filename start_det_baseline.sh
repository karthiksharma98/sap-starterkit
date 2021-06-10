dataDir="/home/ubuntu/data"

methodName=mrcnn50_nm
scale=1.0
factor=1.0

python det_baseline.py \
    --config "$HOME/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py" \
	--weights "$dataDir/ModelZoo/mmdet/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth" \
    --annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
    --eval-config "./config.json" \
    --in-scale $scale \
    --no-mask \
    --dynamic-schedule \
    --perf-factor $factor
    
