function model = set_cache_folder_caltech(cache_base_proposal, model)
% model = set_cache_folder_caltech(cache_base_proposal, model)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    model.stage1_rpn.cache_name = [cache_base_proposal, '_stage1_rpn'];

end