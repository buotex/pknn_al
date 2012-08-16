function model_pknn=pknnTrain(K, lbl, numclass, params, zc0)
% model_pknn=pknnTrain(K, lbl, numclass, params, zc0)
% 
% pknnTrain trains the pknn classifier over given training kernel K and
% using the provided labels lbl (See Jain and Kapoor, CVPR'09). Labels are
% assumed to be from 1 to numclass. If you have some other labels then
% please convert them to integer in the interval [1, numclass].
%
% K:        Kernel over the training data points
% lbl:      Labels of the training data points
% numclass: Number of classes
% params:   Parameters to the classifier
%           See SetDefaultArguments()
% zc0:      Initial zc (w in Eq 6, Jain and Kapoor, CVPR'09)
% Returns model_pknn: Structure containing model parameters of the trained pknn classifier
%                     model_pknn.KK_row1: first row of S[v;w] in Eq 6, Jain
%                                         and Kapoor, CVPR'09
%                     model_pknn.KK_row2: second row of S[v;w] in Eq 6, Jain
%                                         and Kapoor, CVPR'09
%                     model_pknn.zc:      w in Eq 6, Jain and Kapoor, CVPR'09
%                     model_pknn.v:       v in Eq 6, Jain and Kapoor, CVPR'09
%
% % Assumptions:
% 1) The kernel should be positive semi-definite and should have reasonably large rank
% 2) Label are assumed to be from 1:numclass (numclass is the number of classes)
%




if(nargin<3)
    error('Not enough input arguments');
end

if(~exist('zc0'))
    zc0=zeros(size(K,1),numclass);
end

if(~exist('params'))
    params=SetDefaultArguments(numclass);
end

%Form constraints
C = GetProbConstraints(lbl, params.beta, params.alpha);

%Learn kernels
[model_pknn.KK_row1 model_pknn.KK_row2 model_pknn.zc model_pknn.v]= kernelLearn_pknn(C, K, lbl, params, zc0);
