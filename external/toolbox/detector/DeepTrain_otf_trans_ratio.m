function detector = DeepTrain_otf_trans_ratio( varargin )
% Train aggregate channel features object detector.
%
% Train aggregate channel features (ACF) object detector as described in:
%  P. Doll�r, R. Appel, S. Belongie and P. Perona
%   "Fast Feature Pyramids for Object Detection", PAMI 2014.
% The ACF detector is fast (30 fps on a single core) and achieves top
% accuracy on rigid object detection. Please see acfReadme.m for details.
%
% Takes a set of parameters opts (described in detail below) and trains a
% detector from start to finish including performing multiple rounds of
% bootstrapping if need be. The return is a struct 'detector' for use with
% acfDetect.m which fully defines a sliding window detector. Training is
% fast (on the INRIA pedestrian dataset training takes ~10 minutes on a
% single core or ~3m using four cores). Taking advantage of parallel
% training requires launching matlabpool (see help for matlabpool). The
% trained detector may be altered in certain ways via acfModify(). Calling
% opts=acfTrain() returns all default options.
%
% (1) Specifying features and model: The channel features are defined by
% 'pPyramid'. See chnsCompute.m and chnsPyramid.m for more details. The
% channels may be convolved by a set 'filters' to remove local correlations
% (see our NIPS14 paper on LDCF), improving accuracy but slowing detection.
% If 'filters'=[wFilter,nFilter] these are automatically computed. The
% model dimensions ('modelDs') define the window height and width. The
% padded dimensions ('modelDsPad') define the extended region around object
% candidates that are used for classification. For example, for 100 pixel
% tall pedestrians, typically a 128 pixel tall region is used to make a
% decision. 'pNms' controls non-maximal suppression (see bbNms.m), 'stride'
% controls the window stride, and 'cascThr' and 'cascCal' are the threshold
% and calibration used for the constant soft cascades. Typically, set
% 'cascThr' to -1 and adjust 'cascCal' until the desired recall is reached
% (setting 'cascCal' shifts the final scores output by the detector by the
% given amount). Training alternates between sampling (bootstrapping) and
% training an AdaBoost classifier (clf). 'nWeak' determines the number of
% training stages and number of trees after each stage, e.g. nWeak=[32 128
% 512 2048] defines four stages with the final clf having 2048 trees.
% 'pBoost' specifies parameters for AdaBoost, and 'pBoost.pTree' are the
% decision tree parameters, see adaBoostTrain.m for details. Finally,
% 'seed' is the random seed used and makes results reproducible and 'name'
% defines the location for storing the detector and log file.
%
% (2) Specifying training data location and amount: The training data can
% take on a number of different forms. The positives can be specified using
% either a dir of pre-cropped windows ('posWinDir') or dirs of full images
% ('posImgDir') and ground truth labels ('posGtDir'). The negatives can by
% specified using a dir of pre-cropped windows ('negWinDir'), a dir of full
% images without any positives and from which negatives can be sampled
% ('negImgDir'), and finally if neither 'negWinDir' or 'negImgDir' are
% given negatives are sampled from the images in 'posImgDir' (avoiding the
% positives). For the pre-cropped windows all images must have size at
% least modelDsPad and have the object (of size exactly modelDs) centered.
% 'imreadf' can be used to specify a custom function for loading an image,
% and 'imreadp' are custom additional parameters to imreadf. When sampling
% from full images, 'pLoad' determines how the ground truth is loaded and
% converted to a set of positive bbs (see bbGt>bbLoad). 'nPos' controls the
% total number of positives to sample for training (if nPos=inf the number
% of positives is limited by the training set). 'nNeg' controls the total
% number of negatives to sample and 'nPerNeg' limits the number of
% negatives to sample per image. 'nAccNeg' controls the maximum number of
% negatives that can accumulate over multiple stages of bootstrapping.
% Define 'pJitter' to jitter the positives (see jitterImage.m) and thus
% artificially increase the number of positive training windows. Finally if
% 'winsSave' is true cropped windows are saved to disk as a mat file.
%
% USAGE
%  detector = acfTrain( opts )
%  opts = acfTrain()
%
% INPUTS
%  opts       - parameters (struct or name/value pairs)
%   (1) features and model:
%   .pPyramid   - [{}] params for creating pyramid (see chnsPyramid)
%   .filters    - [] [wxwxnChnsxnFilter] filters or [wFilter,nFilter]
%   .modelDs    - [] model height+width without padding (eg [100 41])
%   .modelDsPad - [] model height+width with padding (eg [128 64])
%   .pNms       - [..] params for non-maximal suppression (see bbNms.m)
%   .stride     - [4] spatial stride between detection windows
%   .cascThr    - [-1] constant cascade threshold (affects speed/accuracy)
%   .cascCal    - [.005] cascade calibration (affects speed/accuracy)
%   .nWeak      - [128] vector defining number weak clfs per stage
%   .pBoost     - [..] parameters for boosting (see adaBoostTrain.m)
%   .seed       - [0] seed for random stream (for reproducibility)
%   .name       - [''] name to prepend to clf and log filenames
%   (2) training data location and amount:
%   .posGtDir   - [''] dir containing ground truth
%   .posImgDir  - [''] dir containing full positive images
%   .negImgDir  - [''] dir containing full negative images
%   .posWinDir  - [''] dir containing cropped positive windows
%   .negWinDir  - [''] dir containing cropped negative windows
%   .imreadf    - [@imread] optional custom function for reading images
%   .imreadp    - [{}] optional custom parameters for imreadf
%   .pLoad      - [..] params for bbGt>bbLoad (see bbGt)
%   .nPos       - [inf] max number of pos windows to sample
%   .nNeg       - [5000] max number of neg windows to sample
%   .nPerNeg    - [25]  max number of neg windows to sample per image
%   .nAccNeg    - [10000] max number of neg windows to accumulate
%   .pJitter    - [{}] params for jittering pos windows (see jitterImage)
%   .winsSave   - [0] if true save cropped windows at each stage to disk
%
% OUTPUTS
%  detector   - trained object detector (modify only via acfModify)
%   .opts       - input parameters used for model training
%   .clf        - learned boosted tree classifier (see adaBoostTrain)
%   .info       - info about channels (see chnsCompute.m)
%
% EXAMPLE
%
% See also acfReadme, acfDetect, acfDemoInria, acfModify, acfTest,
% chnsCompute, chnsPyramid, adaBoostTrain, bbGt, bbNms, jitterImage
%
% Piotr's Computer Vision Matlab Toolbox      Version NEW
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% initialize opts struct
opts = initializeOpts( varargin{:} );
if(nargin==0), detector=opts; return; end

disp(opts);

% load or initialize detector and begin logging
nm=[opts.name 'Detector.mat']; 
t=exist(nm,'file');
if(t), 
    if(nargout), 
        t=load(nm); 
        detector=t.detector; 
    end; 
    return; 
end
t=fileparts(nm); if(~isempty(t) && ~exist(t,'dir')), mkdir(t); end
detector = struct( 'opts',opts, 'clf',[], 'info',[] );
startTrain=clock; nm=[opts.name 'Log.txt'];
if(exist(nm,'file')), diary(nm); diary('off'); delete(nm); end; diary(nm);
RandStream.setGlobalStream(RandStream('mrg32k3a','Seed',opts.seed));

% sample positives 
[X1, X1_score, ~] = sampleWins( detector, 0, 1 );
X1 = single(X1);

% iterate bootstraping and training
for stage = 0:numel(opts.nWeak)-1
  diary('on'); fprintf([repmat('-',[1 75]) '\n']);
  fprintf('Training stage %i\n',stage); startStage=clock;
  
  % sample negatives and compute features
  [X0, X0_score, sel_idxes] = sampleWins( detector, stage, 0 );
  X0 = single(X0);
  
%   if stage == 0 && ~isempty(opts.init_detector)
%       % if there is init_detector, init it and skip stage 0 training
%       ld = load(opts.init_detector);
%       detector.clf = ld.detector.clf;
%       X0p=X0;
%       clear ld;
%       continue;
%   end
  
  % accumulate negatives from previous stages
%   if( stage>0 )
%     n0=size(X0p,1); n1=max(opts.nNeg,opts.nAccNeg)-size(X0,1);
%     if(n0>n1 && n1>0), X0p=X0p(randSample(n0,n1),:); end
%     if(n0>0 && n1>0), X0=[X0p; X0]; end %#ok<AGROW>
%   end; X0p=X0;
  
  % accumulate negatives from previous stages
  if( stage>0 )
    n0=size(X0p,1); n1=max(opts.nNeg,opts.nAccNeg)-size(X0,1);
    if(n0>n1 && n1>0), 
        sel_idx = randSample(n0,n1);
        X0p=X0p(sel_idx,:); 
        X0_score_p=X0_score_p(sel_idx,:); 
    end
    if(n0>0 && n1>0), 
        X0=[X0p; X0]; 
        X0_score=[X0_score_p; X0_score]; 
    end %#ok<AGROW>
  end; 
  X0p=X0;
  X0_score_p=X0_score;
  
  % train boosted clf
  detector.opts.pBoost.nWeak = opts.nWeak(stage+1);
  detector.clf = adaBoostTrain_trans(X0,X0_score,X1,X1_score,detector.opts.pBoost);
  detector.clf.hs = detector.clf.hs + opts.cascCal;
  
  % save intermediate model
  detector_tmp = detector;
  detector.opts.roidb_test = [];
  detector.opts.roidb_train = [];
  detector.opts.imdb_test = [];
  detector.opts.imdb_train = [];
  disp(['save model as :' opts.name '_stage' num2str(stage) '_Detector.mat']);
  save([opts.name 'Detector.mat'],'detector'); % for test
  save([opts.name '_stage' num2str(stage) '_Detector.mat'], 'detector');
  save([opts.name '_stage' num2str(stage) '_X0.mat'], 'sel_idxes', '-v7.3');
  detector = detector_tmp;
  
  % test each stage model except the last stage
  if stage < numel(opts.nWeak)-1
      dataDir=opts.dataDir;
      pLoad={'lbls',{'person'},'ilbls',{'people'},'squarify',{3,.41}};
      DeepTest_otf_trans_ratio('name',opts.name,'roidb_test', opts.roidb_test, 'imdb_test', opts.imdb_test, ...
          'gtDir',[dataDir 'test/annotations'],'pLoad',[pLoad, 'hRng',[50 inf],...
          'vRng',[.65 1],'xRng',[5 635],'yRng',[5 475]],...
          'reapply',1,'show',2, 'nms_thres', opts.nms_thres, ...
          'conf', opts.conf, 'caffe_net', opts.caffe_net, 'silent', true, 'cache_dir', opts.cache_dir, 'ratio', opts.ratio);
      delete([opts.name 'Detector.mat']); % delete the tmp file
  end
  
  % update log
  fprintf('Done training stage %i (time=%.0fs).\n',...
    stage,etime(clock,startStage)); diary('off');
  
end

% save detector
detector.opts.roidb_test = [];
detector.opts.roidb_train = [];
detector.opts.imdb_test = [];
detector.opts.imdb_train = [];
save([opts.name 'Detector.mat'],'detector');

% finalize logging
diary('on'); fprintf([repmat('-',[1 75]) '\n']);
fprintf('Done training (time=%.0fs).\n',...
  etime(clock,startTrain)); diary('off');

end

function opts = initializeOpts( varargin )
% Initialize opts struct.
% dfs= { 'pPyramid',{}, 'filters',[], ...
%   'modelDs',[100 41], 'modelDsPad',[128 64], ...
%   'pNms',struct(), 'stride',4, 'cascThr',-1, 'cascCal',.005, ...
%   'nWeak',128, 'pBoost', {}, 'seed',0, 'name','', 'posGtDir','', ...
%   'posImgDir','', 'negImgDir','', 'posWinDir','', 'negWinDir','', ...
%   'imreadf',@imread, 'imreadp',{}, 'pLoad',{}, 'nPos',inf, 'nNeg',5000, ...
%   'nPerNeg',25, 'nAccNeg',10000, 'pJitter',{}, 'winsSave',0 };
dfs= { 'filters',[], ... %   'modelDs',[100 41], 'modelDsPad',[128 64], ...
  'pNms',struct(), 'stride',4, 'cascThr',-1, 'cascCal',.005, ...
  'nWeak',128, 'pBoost', {}, 'seed',0, 'name','', 'posGtDir','', ...
  'posImgDir','', 'negImgDir','', 'posWinDir','', 'negWinDir','', ...
  'imreadf',@imread, 'imreadp',{}, 'pLoad',{}, 'nPos',inf, 'nNeg',5000, ...
  'nPerNeg',25, 'nAccNeg',10000, 'pJitter',{}, 'winsSave',0 ...
  'roidb_train', [], 'roidb_test', [], 'imdb_train', [], 'imdb_test', [], ...
  'fg_thres_hi', 1, 'fg_thres_lo', 0.5, 'bg_thres_hi', 0.5, 'bg_thres_lo', 0, ...
  'dataDir', '', 'caffe_net', [], 'conf', [], 'feat_len', 0, 'train_gts', {}, ...
  'exp_name', '', 'fg_nms_thres', 1, 'fg_use_gt', false, 'bg_hard_min_ratio', [], ...
  'init_detector', '', 'bg_nms_thres', 1, 'max_rois_num_in_gpu', 2000, ...
  'first_nNeg', 0, 'cache_dir', 'models', 'nms_thres', 0.5, 'load_gt', false, 'ratio', 1};
opts = getPrmDflt(varargin,dfs,1);
% fill in remaining parameters
% p=chnsPyramid_zll([],opts.pPyramid); p=p.pPyramid;
% p.minDs=opts.modelDs; shrink=p.pChns.shrink;
% opts.modelDsPad=ceil(opts.modelDsPad/shrink)*shrink;
% p.pad=ceil((opts.modelDsPad-opts.modelDs)/shrink/2)*shrink;
% p=chnsPyramid_zll([],p); p=p.pPyramid; p.complete=1;
% p.pChns.complete=1; opts.pPyramid=p;
% initialize pNms, pBoost, pBoost.pTree, and pLoad
dfs={ 'type','maxg', 'overlap',.65, 'ovrDnm','min' };
opts.pNms=getPrmDflt(opts.pNms,dfs,-1);
dfs={ 'pTree',{}, 'nWeak',0, 'discrete',1, 'verbose',16 };
opts.pBoost=getPrmDflt(opts.pBoost,dfs,1);
dfs={'nBins',256,'maxDepth',2,'minWeight',.01,'fracFtrs',1,'nThreads',16};
opts.pBoost.pTree=getPrmDflt(opts.pBoost.pTree,dfs,1);
% opts.pLoad=getPrmDflt(opts.pLoad,{'squarify',{0,1}},-1);
% opts.pLoad.squarify{2}=opts.modelDs(2)/opts.modelDs(1);
end

function [feats, scores, sel_idxes] = sampleWins( detector, stage, positive )
% Load or sample windows for training detector.
opts=detector.opts; start=clock;
if( positive ), 
    n=opts.nPos; 
else
    if stage == 0
        n = opts.first_nNeg;
    else
        n=opts.nNeg;
    end
end

if positive
    % generate pos sample
    thres_lo = opts.fg_thres_lo;
    thres_hi = opts.fg_thres_hi;
    mat_name = fullfile(opts.dataDir, ['pos_feats_' opts.exp_name, '_' num2str(thres_lo) 'to' num2str(thres_hi) '_nms' num2str(opts.fg_nms_thres) '.mat']);
    try
        if opts.load_gt == false
            error('skip load gt');
        end
        ld = load(mat_name);
        feats = ld.fg_feat;
        scores = ld.fg_score;
        sel_idxes = ld.sel_idxes;
        clear ld;
    catch     
        rois = opts.roidb_train.rois;

        fg_feat_cell = cell(length(rois), 1);
        sel_idxes = cell(length(rois), 1);
        fg_score_cell = cell(length(rois), 1);
        if n < length(rois)
            error('Not implement yet.');
        end
        gt_num = 0;
        for idx = 1:length(rois)
            if mod(idx, 100) == 0
                fprintf('Pos: %d / %d (select %d gt)\n', idx, length(rois), gt_num);
            end
            
            % sel_idx = find(rois(idx).overlap >= thres_lo & rois(idx).overlap < thres_hi);
            
            % here we use the ols in dollar's code
            gt = opts.train_gts{idx};
            gt = gt(gt(:,5)<1, :); % for fg, exlucde ignore gts when eval ols 
            if ~isempty(gt)
                boxes = zeros(size(rois(idx).boxes));
                boxes(:, 1) = rois(idx).boxes(:, 1);
                boxes(:, 2) = rois(idx).boxes(:, 2);
                boxes(:, 3) = rois(idx).boxes(:, 3) - rois(idx).boxes(:, 1);
                boxes(:, 4) = rois(idx).boxes(:, 4) - rois(idx).boxes(:, 2);
                ols= bbGt('compOas',boxes,gt,gt(:,5));
                max_ols = max(ols, [], 2);
                %             gt(:,3) = gt(:,3) + gt(:,1);
                %             gt(:,4) = gt(:,4) + gt(:,2);
                %             showboxes2(im, gt)
            else
                max_ols = zeros(size(rois(idx).gt));
            end
            sel_idx = find(max_ols >= thres_lo & max_ols < thres_hi);
            
            % ignore the gt
            gt_idx = find(rois(idx).ignores < 1);
            sel_idx = setdiff(sel_idx, gt_idx);
            
            if opts.fg_nms_thres < 1
                sel_boxes = rois(idx).boxes(sel_idx, :);
                nms_sel_idxes = nms([sel_boxes max_ols(sel_idx)], opts.fg_nms_thres);
                sel_idx = sel_idx(nms_sel_idxes);                
            end
%             showboxes2(im, rois(idx).boxes(sel_idx(nms_sel_idxes), :))

            % is use gt?
            if opts.fg_use_gt
                sel_idx = union(sel_idx, gt_idx);
            end

            gt_num = gt_num + length(sel_idx);
            sel_idxes{idx} = sel_idx;
            sel_boxes = rois(idx).boxes(sel_idx, :);
            if ~isempty(sel_boxes)
                im = imread(opts.imdb_train.image_at(idx)); 
                sel_boxes = rois(idx).boxes(sel_idx, :); %showboxes2(im, sel_boxes)
                fg_feat_cell{idx} = rois_get_features_ratio(opts.conf, opts.caffe_net, im, sel_boxes, opts.max_rois_num_in_gpu, opts.ratio);
%                 assert(size(fg_feat_cell{idx}, 2)==opts.feat_len, sprintf('assert fail: feat_len should set to %d', size(fg_feat_cell{idx}, 2)));
                fg_score_cell{idx} =  rois(idx).scores(sel_idx);
            end
        end
        sel_idx = cellfun(@(x) ~isempty(x), fg_feat_cell);
        fg_feat = cell2mat(fg_feat_cell(sel_idx));
        fg_score = cell2mat(fg_score_cell(sel_idx));
        feats = fg_feat;
        scores = fg_score;
%         if opts.load_gt == false
%             save(mat_name, 'fg_feat', 'fg_score', 'sel_idxes', '-v7.3');
%         end
    end
    fprintf('Select pos num: %d \n', size(feats, 1));
else
   % generate neg sample
   rois = opts.roidb_train.rois;
   thres_lo = opts.bg_thres_lo;
   thres_hi = opts.bg_thres_hi;
   bg_feat = zeros(n, opts.feat_len);
   bg_score = zeros(n, 1);
   bg_feat_idx = 1;
   sel_idxes = cell(length(rois), 1);
   rand_idx = randperm(length(rois));
   for i = 1:length(rand_idx)
       idx = rand_idx(i);
       if mod(i, 100) == 0
           fprintf('Neg: %d / %d (from %d images)\n', bg_feat_idx, n, i);
       end
       
       % sel_idx = find(rois(idx).overlap >= thres_lo & rois(idx).overlap < thres_hi);
       
       % here we use the ols in dollar's code
       gt = opts.train_gts{idx};
       if ~isempty(gt)
           boxes = zeros(size(rois(idx).boxes));
           boxes(:, 1) = rois(idx).boxes(:, 1);
           boxes(:, 2) = rois(idx).boxes(:, 2);
           boxes(:, 3) = rois(idx).boxes(:, 3) - rois(idx).boxes(:, 1);
           boxes(:, 4) = rois(idx).boxes(:, 4) - rois(idx).boxes(:, 2);
           ols= bbGt('compOas',boxes,gt,gt(:,5));
           max_ols = max(ols, [], 2);
           %             gt(:,3) = gt(:,3) + gt(:,1);
           %             gt(:,4) = gt(:,4) + gt(:,2);
           %             showboxes2(im, gt)
       else
           max_ols = zeros(size(rois(idx).gt));
       end
       sel_idx = find(max_ols >= thres_lo & max_ols < thres_hi);

       if stage > 0   % mining hard except the first stage
           if opts.bg_hard_min_ratio(stage) < 1
               retain_num = round(length(sel_idx) * opts.bg_hard_min_ratio(stage));
               retain_idx = randperm(length(sel_idx), retain_num);
               sel_idx = sel_idx(retain_idx);
%                sel_feat = sel_feat(retain_idx, :);
           end
           if opts.bg_nms_thres < 1
               sel_box = rois(idx).boxes(sel_idx, :);
               sel_scores = rois(idx).scores(sel_idx, :);
               nms_sel_idxes = nms([sel_box sel_scores], opts.bg_nms_thres);
               sel_idx = sel_idx(nms_sel_idxes);
               sel_box = rois(idx).boxes(sel_idx, :);
               sel_scores = rois(idx).scores(sel_idx, :);
           else
               sel_box = rois(idx).boxes(sel_idx, :);
               sel_scores = rois(idx).scores(sel_idx, :);
           end 
           im = imread(opts.imdb_train.image_at(idx)); 
           sel_feat = rois_get_features_ratio(opts.conf, opts.caffe_net, im, sel_box, opts.max_rois_num_in_gpu, opts.ratio);
           
           if ~isempty(sel_idx)               
               scores = DeepDetect_otf_trans(sel_feat, sel_scores, detector);
               hard_idx = scores > detector.opts.cascThr;
               sel_idx = sel_idx(hard_idx);
               sel_feat = sel_feat(hard_idx, :);
           end
       else % for the first stage
           im = imread(opts.imdb_train.image_at(idx)); 
           retain_idx = randperm(length(sel_idx), min(opts.nPerNeg, length(sel_idx)));
           sel_idx = sel_idx(retain_idx);
           sel_box = rois(idx).boxes(sel_idx, :);
           sel_feat = rois_get_features_ratio(opts.conf, opts.caffe_net, im, sel_box, opts.max_rois_num_in_gpu, opts.ratio);
       end
       
       
       
%        disp(length(sel_idx));
       if length(sel_idx) > opts.nPerNeg
           retain_idx = randperm(length(sel_idx), opts.nPerNeg);
           sel_idx = sel_idx(retain_idx);
           sel_feat = sel_feat(retain_idx, :);
       end
       
       if ~isempty(sel_idx)
           bg_feat(bg_feat_idx:bg_feat_idx+length(sel_idx)-1, :) = sel_feat;
           bg_score(bg_feat_idx:bg_feat_idx+length(sel_idx)-1) = rois(idx).scores(sel_idx);
           bg_feat_idx = bg_feat_idx+length(sel_idx);
           sel_idxes{idx} = sel_idx;
       end
       if bg_feat_idx > n
           break;
       end
       
   end
   if bg_feat_idx < n
       feats = bg_feat(1:bg_feat_idx-1, :);
       scores = bg_score(1:bg_feat_idx-1, :);
   else
       feats = bg_feat(1:n, :);
       scores = bg_score(1:n, :);
   end
   fprintf('Select neg num: %d from %d images\n', size(feats, 1), i);
end
fprintf('done (time=%.0fs).\n',etime(clock,start));
end






