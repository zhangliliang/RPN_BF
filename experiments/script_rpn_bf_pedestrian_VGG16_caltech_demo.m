function script_rpn_bf_pedestrian_VGG16_caltech_demo()
close all;
clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.per_nms_topN           = -1;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 100;
opts.use_gpu                = true;

opts.test_scales            = 720;
opts.test_max_size          = 960;
opts.feat_stride            = 16;
opts.test_binary            = false;
opts.test_min_box_size      = 16;
opts.test_min_box_height    = 50;
opts.test_drop_boxes_runoff_image = true;

opts.max_rois_num_in_gpu    = 2000;
opts.ratio                  = 1;


%% -------------------- INIT_MODEL --------------------
model_dir                   = fullfile(pwd, 'output', 'VGG16_caltech_final'); 


rpn_bf_model.rpn_net_def ...
    = fullfile(model_dir, 'rpn_test.prototxt');
rpn_bf_model.rpn_net ...
    = fullfile(model_dir, 'final');
rpn_bf_model.bf_net_def ...
    = fullfile(model_dir, 'bf_test.prototxt');

rpn_bf_model.conf_rpn.test_scales = opts.test_scales;
rpn_bf_model.conf_rpn.test_max_size = opts.test_max_size;
rpn_bf_model.conf_rpn.max_size = opts.test_max_size;
rpn_bf_model.conf_rpn.feat_stride = opts.feat_stride;
rpn_bf_model.conf_rpn.test_binary = opts.test_binary;
rpn_bf_model.conf_rpn.test_min_box_size = opts.test_min_box_size;
rpn_bf_model.conf_rpn.test_min_box_height = opts.test_min_box_height;
rpn_bf_model.conf_rpn.test_drop_boxes_runoff_image = opts.test_drop_boxes_runoff_image;


rpn_bf_model.conf_bf.test_scales = opts.test_scales;
rpn_bf_model.conf_bf.test_max_size = opts.test_max_size;
rpn_bf_model.conf_bf.max_size = opts.test_max_size;

if opts.use_gpu
    ld = load(fullfile(model_dir, 'mean_image'));  
    rpn_bf_model.conf_rpn.image_means = gpuArray(ld.image_mean);
    rpn_bf_model.conf_bf.image_means = gpuArray(ld.image_mean);
    clear ld;
end
ld = load(fullfile(model_dir, 'anchors'));  
rpn_bf_model.conf_rpn.anchors = ld.anchors;
clear ld;

rpn_net = caffe.Net(rpn_bf_model.rpn_net_def, 'test');
rpn_net.copy_from(rpn_bf_model.rpn_net);
fast_rcnn_net = caffe.Net(rpn_bf_model.bf_net_def, 'test');


% set gpu/cpu
if opts.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end    

% addpath('external/code3.2.1');
addpath(genpath('external/toolbox'));
ld = load(fullfile(model_dir, 'DeepCaltech_otfDetector'));
detector = ld.detector;
clear ld;

rpn_bf_model.conf_bf.nms_thres = 0.5;
rpn_bf_model.conf_bf.cascThr = -1;

featmap_blobs_names = {'conv3_3', 'conv4_3_atrous'};

%% -------------------- WARM UP --------------------
% the first run will be slower; use an empty image to warm up

for j = 1:2 % we warm up 2 times
    im = uint8(ones(375, 500, 3)*128);
    if opts.use_gpu
        im = gpuArray(im);
    end
    [boxes, scores]             = proposal_im_detect_caltech(rpn_bf_model.conf_rpn, rpn_net, im);
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    featmap_blobs = cell(size(featmap_blobs_names));
    for i = 1:length(featmap_blobs_names);
        featmap_blobs{i} = rpn_net.blobs(featmap_blobs_names{i});
    end
    feat = rois_get_features_from_featmap_ratio(rpn_bf_model.conf_bf, fast_rcnn_net, im, featmap_blobs, aboxes(:, 1:4), opts.max_rois_num_in_gpu, opts.ratio);
end

%% -------------------- TESTING --------------------
im_names = {'ped1.jpg', 'ped2.jpg', 'ped3.jpg'};


running_time = [];
for j = 1:length(im_names)
    
    im = imread(fullfile(pwd, im_names{j}));
    
    if opts.use_gpu
        im = gpuArray(im);
    end
    
    % test rpn
    th = tic();
    [boxes, scores]             = proposal_im_detect(rpn_bf_model.conf_rpn, rpn_net, im);
    t_proposal = toc(th);
    th = tic();
    aboxes                      = boxes_filter([boxes, scores], opts.per_nms_topN, opts.nms_overlap_thres, opts.after_nms_topN, opts.use_gpu);
    t_nms = toc(th);
    
    % test bf
    th = tic();
    featmap_blobs = cell(size(featmap_blobs_names));
    for i = 1:length(featmap_blobs_names);
        featmap_blobs{i} = rpn_net.blobs(featmap_blobs_names{i});
    end
    feat = rois_get_features_from_featmap_ratio(rpn_bf_model.conf_bf, fast_rcnn_net, im, featmap_blobs, aboxes(:, 1:4), opts.max_rois_num_in_gpu, opts.ratio);
    scores = adaBoostApply(feat, detector.clf);
    bbs = [aboxes(:, 1:4) scores];
    sel_idx = nms(bbs, rpn_bf_model.conf_bf.nms_thres);
    sel_idx = intersect(sel_idx, find(bbs(:, end) > rpn_bf_model.conf_bf.cascThr));
    scores = scores(sel_idx, :);
    boxes = aboxes(sel_idx, 1:4); 
    t_detection = toc(th);
    
    fprintf('%s (%dx%d): time %.3fs (resize+conv+proposal: %.3fs, nms+regionwise: %.3fs)\n', im_names{j}, ...
        size(im, 2), size(im, 1), t_proposal + t_nms + t_detection, t_proposal, t_nms+t_detection);
    running_time(end+1) = t_proposal + t_nms + t_detection;
    
    % visualize
    classes = {'pedestrian'};
    boxes_cell = cell(length(classes), 1);
    thres = 0.6;
    for i = 1:length(boxes_cell)
        boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
        boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, 0.3), :);
        
        I = boxes_cell{i}(:, 5) >= thres;
        boxes_cell{i} = boxes_cell{i}(I, :);
    end
    figure(j);
    showboxes(im, boxes_cell, classes, 'voc');
    pause(0.1);
end
fprintf('mean time: %.3fs\n', mean(running_time));

caffe.reset_all(); 
clear mex;

end

function aboxes = boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN, use_gpu)
    % to speed up nms
    if per_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), per_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres, use_gpu), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), after_nms_topN), :);
    end
end
