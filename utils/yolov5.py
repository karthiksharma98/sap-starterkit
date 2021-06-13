def init_detector(opts, device='cuda:0'):
    config = mmcv.Config.fromfile(opts.config)
    new_config = 'train_pipeline' in config or 'test_pipeline' in config
    if new_config:
        # simulate old config
        if opts.in_scale is None:
            print('Warning: using new config and fixing size_divisor to 32')
            config.data.test.img_scale = config.test_pipeline[1]['img_scale']
        else:
            config.data.test.img_scale = 1
        config.data.test.size_divisor = 32
    if opts.in_scale is not None:
        if 'ssd' in basename(opts.config):
            # SSD
            if opts.in_scale <= 0.2:
                # too small leads to some issues
                l = round(1920*opts.in_scale)
                config.data.test.img_scale = (l, l)
                config.data.test.resize_keep_ratio = False
            else:
                config.data.test.img_scale = opts.in_scale
                config.data.test.resize_keep_ratio = True
        else:
            config.data.test.img_scale = opts.in_scale
            config.data.test.resize_keep_ratio = True
    if opts.no_mask:
        if 'roi_head' in config.model and 'mask_head' in config.model['roi_head']:
            config.model['roi_head']['mask_head'] = None
    if 'zoom_crop' in opts and opts.zoom_crop:
        config.data.test.zoom_crop = {
            'h': opts.zoom_crop_h,
            'y': opts.zoom_crop_y,
        }
    else:
        config.data.test.zoom_crop = None
    config.model.pretrained = None
    if 'action_head' in config.model:
        config.model['action_head_weights'] = opts.action_head_weights

    model = build_detector(config.model, test_cfg=config.test_cfg)
    map_loc = 'cpu' if device == 'cpu' else None
    checkpoint = load_checkpoint(model, opts.weights, map_location=map_loc)
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use COCO classes by default.')
        model.CLASSES = get_classes('coco')
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def inference_detector(model, img, gpu_pre=True, numpy_res=True, decode_mask=True):
    # assume img has RGB channel ordering instead of BGR
    cfg = model.cfg
    if gpu_pre:
        img_transform = ImageTransformGPU()
            #size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg) #TODO
    else:
        img_transform = ImageTransform()
            #size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    device = next(model.parameters()).device  # model device
    with torch.no_grad():
        data = _prepare_data(img, img_transform, cfg, device)# TODO
        result = model(return_loss=False, rescale=True, numpy_res=numpy_res, decode_mask=decode_mask, **data) 
        zc_cfg = cfg.data.test.zoom_crop # TODO
        if zc_cfg is not None and len(result[0]):
            result[0][:, [1, 3]] += zc_cfg['y']
    return result


def parse_det_result(result, class_mapping=None, n_class=None, separate_scores=True, return_sel=False):
        if len(result) > 2:
        bboxes_scores, labels, masks = result
    else:
        bboxes_scores, labels = result
        masks = None

    if class_mapping is not None:
        labels = class_mapping[labels]
        sel = labels < n_class
        bboxes_scores = bboxes_scores[sel]
        labels = labels[sel]
        if masks is not None:
            masks = masks[sel]
    else:
        sel = None
    if separate_scores:
        if len(labels):
            bboxes = bboxes_scores[:, :4]
            scores = bboxes_scores[:, 4]
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
        outs = [bboxes, scores, labels, masks]
    else:
        outs = [bboxes_scores, labels, masks]
    if return_sel:
        outs.append(sel)
    return tuple(outs)