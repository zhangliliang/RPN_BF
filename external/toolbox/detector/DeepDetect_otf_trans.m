function scores = DeepDetect_otf_trans( feats, scores, detector )

scores = adaBoostApply_otf_trans(feats, scores, detector);

