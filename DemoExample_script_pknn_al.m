% An example script to illustrate the use for pknn classifier for active learning
clear;
addpath Source

%load DemoExampleData/K; %Load kernel for the example data
%load DemoExampleData/labels; %Load true labels for the example data


%Begin BX: 
%Create K for a kernel n x n
%Create lbl for labels n x 1

addpath ~/data/cal101
addpath ~/data/cal101/cal101-ker-15-1
load el2_gb.mat
K = matrix;
%load echi2_phowColor_L0.mat
%K = K + matrix;
%load echi2_phowColor_L1.mat
%K = K + matrix;
%load echi2_phowColor_L2.mat
%K = K + matrix;
%K = K / 4;




load meta-15-1.mat
lbl = trainImageClasses';


%create test data




%%%%%%%%%%%%%%%%%%%%%%%%%%%

%numclass=length(unique(lbl));
numclass = 101
initN=2; %Number of initial labeled examples
poolN=15; %Number of examples in the pool, remaining are "hold-out" examples and are used for testing accuracy

numrun=5; %Average over 20 runs
n=size(K,2);
m=numclass;
params=SetDefaultArguments(numclass); %Set default parameters
params.thres=1e-4;
params.al_round=5; %Set number of active learning rounds to be 10
params.al_numqr=20; %Set number of examples to be labeled in each round to be  2



%Initialize Accuracies
acc_pknn_rf=zeros(params.al_round+1, numrun); %Accuracy for pknn with Active Learning
acc_pknn_al=zeros(params.al_round+1, numrun); %Accuracy for pknn with Active Learning
acc_pknn_rand=zeros(params.al_round+1, numrun); %Accuracy for pknn with Random Selection

for run=1:numrun
    trn_idx=[]; %indices of objects used as training data
    qr_idx=[];  %indices of objects representing the pool of possible candidates, to use for further training
    test_idx=[];%indices of objects to test on




    %Select pool and test set randomly, out of pool set select training
    %and query set (which is queried for examples to be labeled) randomly. 
    for i=1:m
        i_idx=find(lbl==i);
        i_len=length(i_idx);
        rp=randperm(i_len);
        trn_idx((i-1)*initN+1:i*initN)=i_idx(rp(1:initN));
        qr_idx(length(qr_idx)+1:length(qr_idx)+poolN-initN)=i_idx(rp(initN+1:poolN));
        test_idx(length(test_idx)+1:length(test_idx)+i_len-poolN)=i_idx(rp(poolN+1:i_len));
    end

%    trn_idx = randi(length(lbl), 1,initNN);
%    qr_idx = linspace(1,0.5 * length(lbl), 0.5 * length(lbl));
%    test_idx = linspace(1,length(lbl), length(lbl));
%    test_idx(trn_idx) = [];
%    test_idx(qr_idx)  = [];
%


    trn_idx_pknn_rf=trn_idx;
    qr_idx_pknn_rf=qr_idx;
    test_idx_pknn_rf=test_idx;
    params.al_type=0; %Use random selection for active learning
    fprintf('\n pKNN+RF   Run%d\n',run);
    [acc_pknn_rf(:,run),trn_idx_pknn_rf_before, trn_idx_pknn_rf_after] =pknn_new(K([trn_idx_pknn_rf qr_idx_pknn_rf],[trn_idx_pknn_rf qr_idx_pknn_rf]), K(test_idx_pknn_rf, [trn_idx_pknn_rf qr_idx_pknn_rf]), 1:length(trn_idx_pknn_rf), length(trn_idx_pknn_rf)+(1:length(qr_idx_pknn_rf)), lbl([trn_idx_pknn_rf qr_idx_pknn_rf]), lbl(test_idx_pknn_rf), numclass, params);


    %Active learning with method 1
    trn_idx_pknn_al=trn_idx; %Initial training index
    qr_idx_pknn_al=qr_idx; %Initial query set (from which examples to be labeled are selected)
    test_idx_pknn_al=test_idx; %Initial test set

    params.al_type=1; %Use selection via active learning method 1 (See getNewIdx_active)
    fprintf('\n pKNN+AL   Run%d\n',run);
    acc_pknn_al(:,run)=pknn_active(K([trn_idx_pknn_al qr_idx_pknn_al],[trn_idx_pknn_al qr_idx_pknn_al]),K(test_idx_pknn_al, [trn_idx_pknn_al qr_idx_pknn_al]), 1:length(trn_idx_pknn_al),length(trn_idx_pknn_al)+(1:length(qr_idx_pknn_al)) , lbl([trn_idx_pknn_al qr_idx_pknn_al]), lbl(test_idx_pknn_al), numclass,params);


    %Active learning with random selection
    trn_idx_pknn_rand=trn_idx;
    qr_idx_pknn_rand=qr_idx;
    test_idx_pknn_rand=test_idx;
    params.al_type=0; %Use random selection for active learning
    fprintf('\n pKNN+Rand   Run%d\n',run);
    acc_pknn_rand(:,run)=pknn_active(K([trn_idx_pknn_rand qr_idx_pknn_rand],[trn_idx_pknn_rand qr_idx_pknn_rand]), K(test_idx_pknn_rand, [trn_idx_pknn_rand qr_idx_pknn_rand]), 1:length(trn_idx_pknn_rand), length(trn_idx_pknn_rand)+(1:length(qr_idx_pknn_rand)), lbl([trn_idx_pknn_rand qr_idx_pknn_rand]), lbl(test_idx_pknn_rand), numclass, params);
    
end

h = figure('Visible', 'off');
plot(0:params.al_numqr:params.al_numqr*(params.al_round),100*mean(acc_pknn_rf(:,:),2),'r-o');
hold;
plot(0:params.al_numqr:params.al_numqr*(params.al_round),100*mean(acc_pknn_al(:,:),2),'m--x');
plot(0:params.al_numqr:params.al_numqr*(params.al_round),100*mean(acc_pknn_rand(:,:),2),'b--x');
xlabel('Number of Labeled Examples Added');
ylabel('Accuracy');
title(sprintf('Acc. vs. Number of Labeled Examples (%d classes)',numclass));
legend('pKNN+RF','pKNN+Al', 'pKNN+Random Sampling');
print(h,'-dpng', 'results.png')





