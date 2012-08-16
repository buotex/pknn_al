function [pred_test prob_test]=pknnPredict(Ktest, model_pknn)
% [pred_test prob_test]=pknnPredict(Ktest, model_pknn)
%
% pknnPredict computes the predicted labels for test data points using pKNN
% classifier model (See Jain and Kapoor, CVPR'09)
%
% Ktest:    Kernel over the test points w.r.t training points
%           Ktest(i,j) is the kernel value between i-th test point and j-th
%           training point
% model_pknn:Model returned by pknn classifier training (See pknnTrain)
%
% Returns pred_test: Predicted label for each test point
%         prob_test: Probability distribution over classes for each test
%                    point


if(nargin<2)
    error('Not enough input arguments');
end


%Compute probablity distribution over all the class for the test dataset
Ktestv=Ktest*model_pknn.v;
Ktestz=Ktest*model_pknn.zc;
num_test=size(Ktest,1);
prob_test=Ktestv-Ktestz.*repmat(model_pknn.KK_row1,num_test,1)-Ktestv.*repmat(model_pknn.KK_row2,num_test,1);

%For each test point, predict the class with highest probability
[dum pred_test]=max(prob_test,[],2);