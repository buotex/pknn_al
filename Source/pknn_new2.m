function [acc trn_idx added_idx]=pknn_active(Ktr_qr, Ktest, trn_idx, qr_idx, lbl_tr_qr, lbl_test, m, params)
%[acc trn_idx added_idx]=pknn_active(Ktr_qr, Ktest, trn_idx, qr_idx, lbl_tr_qr, lbl_test, m, params)
%
% pknn_active performs active learning using pknn classifier (See Jain and Kapoor, CVPR'09)
%
% Ktr_qr:   Kernel over the training+query (examples queried to be labeled) set
% Ktest:    Kernel over test datapoints and training+query points 
% trn_idx:  Indices of the initial training set
% qr_idx:   Indices of the initial query (examples to be labeled) set
% lbl_tr_qr:Labels of the training+query points
% lbl_test: Labels of the test points
% m:        Number of classes
% params:   Parameters to the pknn and active learning algorithm
%           See SetDefaultArguments()
%
% Returns acc:      Accuracy achieved over the test data after every active learning round
%         trn_idx:  Final indices of training data points
%         added_idx:Indices of the points from query set that are selected to label
%
%
% Assumptions:
% 1) The kernel should be positive semi-definite and should have reasonably large rank
% 2) Labels are assumed to be from 1:numclass (numclass is the number of classes)
% 3) Code typically performs best in the setting where the number of classes is large and the number of examples per class is reasonably small (Note that this the setting where active learning is most crucial and practical).








if(~exist('params'))
    if(nargin<7)
        error('Not enough input arguments');
    end
    params=SetDefaultArguments(m);
else
    if(nargin<8)
        error('Not enough input arguments');
    end
end

%Initialization for active learning
model_pknn.zc=zeros(length(trn_idx),m);
acc=zeros(params.al_round+1,1);
al=1;
num_test=size(Ktest,1);
added_idx=[];


while(true)
    
    %Train pKNN classifier
%    model_pknn=pknnTrain(Ktr_qr(trn_idx, trn_idx), lbl_tr_qr(trn_idx), m, params, model_pknn.zc);
%    
%    %Predict using pKNN classifier
%    pred_test=pknnPredict(Ktest(:,trn_idx), model_pknn);
%    
%    %Compute accuracy of pKNN classifier over test set
%    acc(al)=length(find(pred_test==lbl_test))/num_test;
%    if(params.verbosity>=1)
%        fprintf('Round %d    Accuracy(pKNN) %f\n',al-1, acc(al));
%    end
%    if(al>params.al_round)
%        break;
%    else
%        al=al+1;
%    end
%

    %calculate marginal densities
      
    %[labels, model, loglikely] = emgm(queries, floor(sqrt(length(trn_idx))));
    %marginalProbs = ones(1,length(labels));
    
    %another variant of marginalProbs
    %marginalProbs = zeros(1,length(labels));
    %for i = 1:size(model,2)
    %        indices = [labels == i];
    %        model.mu(:,i)
    %        model.Sigma(:,:,i)
    %        testpdf = mvnpdf(queries(:,indices)', model.mu(:,i)', model.Sigma(:,:,i)');
%	  %  marginalProbs(indices) = testpdf;
    %end

    %Compute probabilities of query points belonging to each class
    %[pred_query prob_query]=pknnPredict(Ktr_qr(qr_idx, trn_idx), model_pknn);

    %get rf votes for every query point
    [counts, trnSampleImportance, pred_test]=getNewIdx_rf(Ktr_qr, trn_idx, qr_idx, lbl_tr_qr, m, Ktest(:,trn_idx)' );
    length(pred_test)


    acc(al)=length(find(pred_test==lbl_test))/num_test;
    if(params.verbosity>=1)
        fprintf('Round %d    Accuracy(pKNN) %f\n',al-1, acc(al));
    end
    if(al>params.al_round)
        break;
    else
        al=al+1;
    end



%the counts themselves are enough to classify, let's test it



    %convert counts to alphas, normalize it somehow?


    %trnSampleImportance = sortrows(trnSampleImportance,-2);

    %    using only the some dimensions
    %    arginalProbs = sum(Ktr_qr(qr_idx, marginalDimensions), 2)

    %    weighted version
    %marginalProbs = Ktr_qr(qr_idx, marginalDimensions) * trnSampleImportance(:,2) 
    %
    marginalDimensions = trnSampleImportance(:, 1) + 1;
    queries = Ktr_qr(marginalDimensions, qr_idx);

    GMFIT = gmdistribution.fit(queries',length(marginalDimensions), 'CovType', 'diagonal', 'SharedCov', true);
    marginalProbs = GMFIT.pdf(queries');


    new_idx = getIndices(counts, marginalProbs, params.al_numqr,m);
    %add the selected indices to the added_idx set and the training set (trn_idx)
    added_idx=[added_idx qr_idx(new_idx)]; 
    trn_idx=[trn_idx qr_idx(new_idx)];

    histc(lbl_tr_qr(trn_idx), [1:m]) ;


    model_pknn.zc=[model_pknn.zc;zeros(length(new_idx),m)];
    qr_idx(new_idx)=[];
  end
