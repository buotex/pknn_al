function params=SetDefaultArguments(m)
% params=SetDefaultArguments(m)
% Sets default values for arguments to the pknn classifer and the active
% learning method
% 
% m: Number of classes
%
% params.al_type:   Active learning method to be used
% params.al_round:  Number of active learning rounds
% params.al_numqr:  Number of query examples to be labeled in each round
% params.thres:     Threshold for maximum violation of constraints in pknn
%                   classifier 
% params.diff_thres:Threshold for the difference in maximum violation of
%                   constraints
% params.alpha:     Lower bound on the probability of a point belonging to
%                   its true class (See Eq 3, Jain and Kapoor CVPR'09)
% params.beta:      Upper bound on the probability of a point belonging to 
%                   classes other than its true class
% params.gamma:     Regularization parameter for slack in the
%                   constraints
% params.itr_cp:    Number of iterations after which maximum violation
%                   condition is checked
% params.maxItr:    Maximum number of iterations in the pknn optimizer
% params.eps:       Order of accuracy required in the projection step in
%                   the pknn optimizer
% params.verbosity: Verbosity level
params.al_type=1;
params.al_round=0;
params.al_numqr=10;
params.thres=1e-3;
params.diff_thres=1e-6;
params.alpha=2/m;
params.beta=.99/m;
params.gamma=1e+20;
params.itr_cp=10;
params.verbosity=1;
params.maxItr=1e+5;
params.eps=1e-4;