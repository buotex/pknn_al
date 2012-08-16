%Script to run pknn active learning with method 1 (see getNewIdx_active) and with random selection


%Active learning with method 1
trn_idx_pknn_al=trn_idx; %Initial training index
qr_idx_pknn_al=qr_idx; %Initial query set (from which examples to be labeled are selected)
test_idx_pknn_al=test_idx; %Initial test set
params=SetDefaultArguments(numclass(numcl)); %Set default parameters
params.thres=1e-4; 
params.al_round=10; %Set number of active learning rounds to be 10
params.al_numqr=20; %Set number of examples to be labeled in each round to be  20
[acc_pknn_al(:,run,numcl) trn_idx_pknn_al added_idx_pknn_al]=pknn_active(K0([trn_idx_pknn_al qr_idx_pknn_al],[trn_idx_pknn_al qr_idx_pknn_al]), K0(test_idx_pknn_al, [trn_idx_pknn_al qr_idx_pknn_al]), 1:length(trn_idx_pknn_al), length(trn_idx_pknn_al)+(1:length(qr_idx_pknn_al)) , lbl([trn_idx_pknn_al qr_idx_pknn_al]), lbl([test_idx_pknn_al]), numclass(numcl), params);


%Active learning with random selection
trn_idx_pknn_rand=trn_idx;
qr_idx_pknn_rand=qr_idx;
test_idx_pknn_rand=test_idx;
params.al_type=0; %Use random selection for active learning
[acc_pknn_rand(:,run,numcl) trn_idx_pknn_rand added_idx_pknn_rand]=pknn_active(K0([trn_idx_pknn_rand qr_idx_pknn_rand],[trn_idx_pknn_rand qr_idx_pknn_rand]), K0(test_idx_pknn_rand, [trn_idx_pknn_rand qr_idx_pknn_rand]), 1:length(trn_idx_pknn_rand), length(trn_idx_pknn_rand)+(1:length(qr_idx_pknn_rand)), lbl([trn_idx_pknn_rand qr_idx_pknn_rand]), lbl([test_idx_pknn_rand]), numclass(numcl), params);
