

dataDir='./datasets/caltech/';
addpath(genpath('./external/code3.2.1'));

for s=1:2
  if(s==1), type='test'; skip=[]; else type='train'; skip=3; end
  dbInfo(['Usa' type]); %if(s==2), type=['train' int2str2(skip,2)]; end
  if(exist([dataDir type '/annotations'],'dir')), continue; end
  dbExtract([dataDir type],1,skip);
end

% dataset.imdb_test     = imdb_from_caltech_flip('./datasets/caltech', 'test', false) ;

% dataset.imdb_train    = imdb_from_caltech_flip('./datasets/caltech', 'train') ;
% dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
