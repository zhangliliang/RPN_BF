function [feats] = rois_get_features_from_featmap_ratio(conf, caffe_net, im, featmap_blobs, boxes, max_rois_num_in_gpu, ratio)

    outer_box_ratio = ratio;
    boxes_width_change = (boxes(:,3)-boxes(:,1))*(outer_box_ratio-1)/2;
    boxes(:,1) = boxes(:,1) - boxes_width_change;
    boxes(:,3) = boxes(:,3) + boxes_width_change;
    boxes_height_change = (boxes(:,4)-boxes(:,2))*(outer_box_ratio-1)/2;
    boxes(:,2) = boxes(:,2) - boxes_height_change;
    boxes(:,4) = boxes(:,4) + boxes_height_change;
    [height, width, ~] = size(im);
    boxes(:,1) = max(1, boxes(:,1));
    boxes(:,3) = max(1, boxes(:,3));
    boxes(:,2) = min(width, boxes(:,2));
    boxes(:,4) = min(height, boxes(:,4));

    [rois_blob, ~] = get_blobs(conf, im, boxes);
    
    % When mapping from image ROIs to feature map ROIs, there's some aliasing
    % (some distinct image ROIs get mapped to the same feature ROI).
    % Here, we identify duplicate feature ROIs, so we only compute features
    % on the unique subset.
    [~, index, inv_index] = unique(rois_blob, 'rows');
    rois_blob = rois_blob(index, :);
    
    
    % set conv feature map as 'data'
    for i = 1:length(featmap_blobs)
        caffe_net.blobs(['data' num2str(i)]).copy_data_from(featmap_blobs{i});
    end
     
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    rois_blob = rois_blob - 1; % to c's index (start from 0)
    rois_blob = permute(rois_blob, [3, 4, 2, 1]);
    rois_blob = single(rois_blob);
    
    total_rois = size(rois_blob, 4);

    total_feats =  cell(ceil(total_rois / max_rois_num_in_gpu), 1);
    for i = 1:ceil(total_rois / max_rois_num_in_gpu)
        
        sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
        sub_ind_end = min(total_rois, i * max_rois_num_in_gpu);
        sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end);
        
        % only set rois blob here
        net_inputs = cell(1, length(featmap_blobs)+1);
        net_inputs{end} = sub_rois_blob;
        
        % Reshape net's input blobs
        caffe_net.reshape_as_input(net_inputs);
        output_blobs = caffe_net.forward(net_inputs);
        
       
        if length(output_blobs) == 1 && length(size(output_blobs{1})) == 2
            total_feats{i} = output_blobs{1}';
        else
            for j = 1:length(output_blobs)
                feats = output_blobs{j};
                feats = permute(feats, [4 2 1 3]);
                feats = reshape(feats, size(feats, 1), size(feats, 2)*size(feats, 3)*size(feats, 4));
                total_feats{i} = [total_feats{i} feats];
            end
        end

    end 
    
    feats = cell2mat(total_feats);
    feats = feats(inv_index, :);

end


function [rois_blob, im_scale_factors] = get_blobs(conf, im, rois)
    im_scale_factors = get_image_blob_scales(conf, im);
    rois_blob = get_rois_blob(conf, rois, im_scale_factors);
end

function im_scales = get_image_blob_scales(conf, im)
    im_scales = arrayfun(@(x) prep_im_for_blob_size(size(im), x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    im_scales = cell2mat(im_scales); 
end

function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)
    [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
    rois_blob = single([levels, feat_rois]);
end

function [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, scales)
    im_rois = single(im_rois);
    
    if length(scales) > 1
        widths = im_rois(:, 3) - im_rois(:, 1) + 1;
        heights = im_rois(:, 4) - im_rois(:, 2) + 1;
        
        areas = widths .* heights;
        scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
        levels = max(abs(scaled_areas - 224.^2), 2); 
    else
        levels = ones(size(im_rois, 1), 1);
    end
    
    feat_rois = round(bsxfun(@times, im_rois-1, scales(levels))) + 1;
end

function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end
    