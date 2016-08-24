function [miss,roc,gt,dt] = DeepTest_otf_trans_ratio( varargin )
% Test aggregate channel features object detector given ground truth.
%
% USAGE
%  [miss,roc,gt,dt] = acfTest( pTest )
%
% INPUTS
%  pTest    - parameters (struct or name/value pairs)
%   .name     - ['REQ'] detector name
%   .imgDir   - ['REQ'] dir containing test images
%   .gtDir    - ['REQ'] dir containing test ground truth
%   .pLoad    - [] params for bbGt>bbLoad for test data (see bbGt>bbLoad)
%   .pModify  - [] params for acfModify for modifying detector
%   .thr      - [.5] threshold on overlap area for comparing two bbs
%   .mul      - [0] if true allow multiple matches to each gt
%   .reapply  - [0] if true re-apply detector even if bbs already computed
%   .ref      - [10.^(-2:.25:0)] reference points (see bbGt>compRoc)
%   .lims     - [3.1e-3 1e1 .05 1] plot axis limits
%   .show     - [0] optional figure number for display
%
% OUTPUTS
%  miss     - log-average miss rate computed at reference points
%  roc      - [nx3] n data points along roc of form [score fp tp]
%  gt       - [mx5] ground truth results [x y w h match] (see bbGt>evalRes)
%  dt       - [nx6] detect results [x y w h score match] (see bbGt>evalRes)
%
% EXAMPLE
%
% See also acfTrain, acfDetect, acfModify, acfDemoInria, bbGt
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.40
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get parameters
dfs={ 'name','REQ', 'roidb_test','REQ','imdb_test','REQ', 'gtDir','REQ', 'pLoad',[], ...
  'thr',.5,'mul',0, 'reapply',0, 'ref',10.^(-2:.25:0), ...
  'lims',[3.1e-3 1e1 .05 1], 'show',0, 'nms_thres', 0.3, ...
  'conf' 'REQ', 'caffe_net' 'REQ', 'silent', false, 'cache_dir', 'REQ', 'ratio', 1};
[name,roidb_test,imdb_test,gtDir,pLoad,thr,mul,reapply,ref,lims,show, nms_thres, conf, caffe_net, silent, cache_dir, ratio] = ...
  getPrmDflt(varargin,dfs,1);

% run detector on directory of images
bbsNm=[name 'Dets.txt'];
if(reapply && exist(bbsNm,'file')), delete(bbsNm); end
if(reapply || ~exist(bbsNm,'file'))
  detector = load([name 'Detector.mat']);
  detector = detector.detector;
  
  if ~silent
      cache_dir = fullfile(cache_dir, 'RPN+BF');
%       if exist(cache_dir, 'dir')
%           error([cache_dir ' is existed! Deal with it']);
%       end
      mkdir_if_missing(fullfile(cache_dir));
  end
  
  rois = roidb_test.rois;
  fid = fopen(bbsNm, 'w');
  for i = 1:length(rois)
      if ~silent
          tic_toc_print('Test: %d / %d \n', i, length(rois));
      end
      if ~isempty(rois(i).boxes)
          im = imread(imdb_test.image_at(i)); 
%           tic;
          feat = rois_get_features_ratio(conf, caffe_net, im, rois(i).boxes, 2000, ratio);   
%           toc;
          scores = DeepDetect_otf_trans(feat, rois(i).scores, detector);
          bbs = [rois(i).boxes scores];
          bbs = bbs(~rois(i).gt, :); % exclude gt
          sel_idx = nms(bbs, nms_thres); % do nms
          bbs = bbs(sel_idx, :);
          bbs(:, 3) = bbs(:, 3) - bbs(:, 1);
          bbs(:, 4) = bbs(:, 4) - bbs(:, 2);
          for j = 1:size(bbs, 1)
              fprintf(fid, '%d,%.2f,%.2f,%.2f,%.2f,%.2f\n', i, bbs(j, :));
          end
          
          if ~silent
              sstr = strsplit(imdb_test.image_ids{i}, '_');
              mkdir_if_missing(fullfile(cache_dir, sstr{1}));
              fid2 = fopen(fullfile(cache_dir, sstr{1}, [sstr{2} '.txt']), 'a');
              for j = 1:size(bbs, 1)
                  fprintf(fid2, '%d,%f,%f,%f,%f,%f\n', str2double(sstr{3}(2:end))+1, bbs(j, :));
              end
              fclose(fid2);
          end
      end
  end
  fclose(fid);
end

% run evaluation using bbGt
[gt,dt] = bbGt('loadAll',gtDir,bbsNm,pLoad);
[gt,dt] = bbGt('evalRes',gt,dt,thr,mul);
[fp,tp,score,miss] = bbGt('compRoc',gt,dt,1,ref);
miss=exp(mean(log(max(1e-10,1-miss)))); roc=[score fp tp];

fprintf('miss rate:%.2f\n', miss*100);

% optionally plot roc
if( ~show ), return; end
figure(show); plotRoc([fp tp],'logx',1,'logy',1,'xLbl','fppi',...
  'lims',lims,'color','g','smooth',1,'fpTarget',ref);
title(sprintf('log-average miss rate = %.2f%%',miss*100));
savefig([name 'Roc'],show,'png');

end
