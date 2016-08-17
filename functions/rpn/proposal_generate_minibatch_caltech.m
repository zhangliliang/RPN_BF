function [input_blobs, random_scale_inds] = proposal_generate_minibatch_caltech(conf, image_roidb)
% [input_blobs, random_scale_inds] = proposal_generate_minibatch_caltech(conf, image_roidb)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    num_images = length(image_roidb);
    assert(num_images == 2 || num_images == 1, 'proposal_generate_minibatch_caltech only support num_images == 2 or 1');

    % Sample random scales to use for each image in this batch
    random_scale_inds = randi(length(conf.scales), num_images, 1);

    assert(mod(conf.batch_size, num_images) == 0, ...
        sprintf('num_images %d must divide BATCH_SIZE %d', num_images, conf.batch_size));
    
    rois_per_image = conf.batch_size / num_images;
    fg_rois_per_image = round(conf.batch_size * conf.fg_fraction);
    
    % Get the input image blob
    [im_blob, im_scales] = get_image_blob(conf, image_roidb, random_scale_inds);
    
    % get fcn output size (fixed for caltech, 640x480)
    img_size = round(image_roidb(1).im_size * im_scales(1));
    output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
    
    % init blobs
    labels_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1), length(image_roidb));
    label_weights_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1), length(image_roidb));
    bbox_targets_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1)*4, length(image_roidb));
    bbox_loss_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1)*4, length(image_roidb));
    
    for i = 1:num_images
        if (i == 1)
            [labels, label_weights, bbox_targets, bbox_loss] = ...
                sample_rois(conf, image_roidb(i), fg_rois_per_image, rois_per_image, im_scales(i), random_scale_inds(i));
        else
            [labels, label_weights, bbox_targets, bbox_loss] = ...
                sample_rois(conf, image_roidb(i), 0, rois_per_image, im_scales(i), random_scale_inds(i));
        end
        
        % get fcn output size
%         img_size = round(image_roidb(i).im_size * im_scales(i));
%         output_size = cell2mat([conf.output_height_map.values({img_size(1)}), conf.output_width_map.values({img_size(2)})]);
        
        assert(img_size(1) == size(im_blob, 1) && img_size(2) == size(im_blob, 2));
        
        cur_labels_blob = reshape(labels, size(conf.anchors, 1), output_size(1), output_size(2));
        cur_label_weights_blob = reshape(label_weights, size(conf.anchors, 1), output_size(1), output_size(2));
        cur_bbox_targets_blob = reshape(bbox_targets', size(conf.anchors, 1)*4, output_size(1), output_size(2));
        cur_bbox_loss_blob = reshape(bbox_loss', size(conf.anchors, 1)*4, output_size(1), output_size(2));
        
        % permute from [channel, height, width], where channel is the
        % fastest dimension to [width, height, channel]
        cur_labels_blob = permute(cur_labels_blob, [3, 2, 1]);
        cur_label_weights_blob = permute(cur_label_weights_blob, [3, 2, 1]);
        cur_bbox_targets_blob = permute(cur_bbox_targets_blob, [3, 2, 1]);
        cur_bbox_loss_blob = permute(cur_bbox_loss_blob, [3, 2, 1]);
        
        labels_blob(:, :, :, i) = cur_labels_blob;
        label_weights_blob(:, :, :, i) = cur_label_weights_blob;
        bbox_targets_blob(:, :, :, i) = cur_bbox_targets_blob;
        bbox_loss_blob(:, :, :, i) = cur_bbox_loss_blob;
    end
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = single(permute(im_blob, [2, 1, 3, 4]));
    labels_blob = single(labels_blob);
    labels_blob(labels_blob > 0) = 1; %to binary lable (fg and bg)
    label_weights_blob = single(label_weights_blob);
    bbox_targets_blob = single(bbox_targets_blob); 
    bbox_loss_blob = single(bbox_loss_blob);
    
    assert(~isempty(im_blob));
    assert(~isempty(labels_blob));
    assert(~isempty(label_weights_blob));
    assert(~isempty(bbox_targets_blob));
    assert(~isempty(bbox_loss_blob));
    
    input_blobs = {im_blob, labels_blob, label_weights_blob, bbox_targets_blob, bbox_loss_blob};
end


%% Build an input blob from the images in the roidb at the specified scales.
function [im_blob, im_scales] = get_image_blob(conf, images, random_scale_inds)
    
    num_images = length(images);
    processed_ims = cell(num_images, 1);
    im_scales = nan(num_images, 1);
    for i = 1:num_images
        im = imread(images(i).image_path);
        target_size = conf.scales(random_scale_inds(i));
        
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    
    im_blob = im_list_to_blob(processed_ims);
end

%% Generate a random sample of ROIs comprising foreground and background examples.
function [labels, label_weights, bbox_targets, bbox_loss_weights] = ...
    sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image, im_scale, im_scale_ind)

    bbox_targets = image_roidb.bbox_targets{im_scale_ind};
    ex_asign_labels = bbox_targets(:, 1);
    
    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = find(bbox_targets(:, 1) > 0);
    
    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(bbox_targets(:, 1) < 0);
    
    % select foreground
    fg_num = min(fg_rois_per_image, length(fg_inds));
    fg_inds = fg_inds(randperm(length(fg_inds), fg_num));
    
%     bg_num = min(rois_per_image - fg_num, length(bg_inds));
    bg_num = min(rois_per_image - fg_rois_per_image, length(bg_inds));
    bg_inds = bg_inds(randperm(length(bg_inds), bg_num));

    labels = zeros(size(bbox_targets, 1), 1);
    % set foreground labels
    labels(fg_inds) = ex_asign_labels(fg_inds);
    assert(all(ex_asign_labels(fg_inds) > 0));
    
    label_weights = zeros(size(bbox_targets, 1), 1);
    % set foreground labels weights
%     label_weights(fg_inds) = 1;
    label_weights(fg_inds) = fg_rois_per_image / fg_num;
    % set background labels weights
    label_weights(bg_inds) = conf.bg_weight;
    
    bbox_targets = single(full(bbox_targets(:, 2:end)));
    
    bbox_loss_weights = bbox_targets * 0;
%     bbox_loss_weights(fg_inds, :) = 1;
    bbox_loss_weights(fg_inds, :) = fg_rois_per_image / fg_num;
end

function visual_anchors(image_roidb, anchors, im_scale)
    imshow(imresize(imread(image_roidb.image_path), im_scale));
    hold on;
    cellfun(@(x) rectangle('Position', RectLTRB2LTWH(x), 'EdgeColor', 'r'), num2cell(anchors, 2));
    hold off;
end

