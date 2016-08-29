function script_rpn_pedestrian_VGG16_caltech()
% script_rpn_pedestrian_VGG16_caltech()
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under TByrhe MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

exp_name = 'VGG16_caltech';

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_rpn_pedestrian_caltech(exp_name);
% cache base
cache_base_proposal         = 'rpn_caltech_vgg_16layers';
% train/test data
dataset                     = [];
% use_flipped                 = true;
% dataset                     = Dataset.caltech_trainval(dataset, 'train', use_flipped);
dataset                     = Dataset.caltech_trainval(dataset, 'train');
% dataset                     = Dataset.caltech_test(dataset, 'test', false);
dataset                     = Dataset.caltech_test(dataset, 'test');

% %% -------------------- TRAIN --------------------
% conf
conf_proposal               = proposal_config_caltech('image_means', model.mean_image, 'feat_stride', model.feat_stride);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder_caltech(cache_base_proposal, model);
% generate anchors and pre-calculate output size of rpn network 

conf_proposal.exp_name = exp_name;
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

                        
%%  train
fprintf('\n***************\nstage one RPN \n***************\n');
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train_caltech(conf_proposal, dataset, model.stage1_rpn, opts.do_val);

%% test
cache_name = 'caltech';
method_name = 'RPN-ped';
Faster_RCNN_Train.do_proposal_test_caltech(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, cache_name, method_name);

end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size_caltech(conf, test_net_def_file);
    anchors                = proposal_generate_anchors_caltech(cache_name, ...
                                    'scales',  2.6*(1.3.^(0:8)), ...
                                    'ratios', [1 / 0.41], ...
                                    'exp_name', conf.exp_name);
end