% extract_img_anno()
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

dataDir='./datasets/caltech/';
addpath(genpath('./external/code3.2.1'));
addpath(genpath('./external/toolbox'));

for s=1:2
  if(s==1), type='test'; skip=[]; else type='train'; skip=3; end
  dbInfo(['Usa' type]);
  if(exist([dataDir type '/annotations'],'dir')), continue; end
  dbExtract([dataDir type],1,skip);
end

