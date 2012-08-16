function new_idx=getNewIdx_active(prob_qr, al_type, al_num, verbosity)
% new_idx=getNewIdx_active(prob_qr, al_type, al_num);
% getNewIdx_active computes examples to be labeled using a variety of
% active learning methods as specified by al_type
% prob_qr:  probability matrix over query set where prob_qr(i,j) specified
%           probability of query i belonging to class j
% al_type:  Specified the active learning method to be used
%           0 Random selection of examples to be labeld
%           1 Selection according to the diff between the mode and the second highest probability
%           2 Selection according to the mode of the probability (Jain and Kapoor CVPR'09)
%           3 Selection according to the entropy of the distribution of each point belonging to a class
%           4 Selection according to the entropy of the trimmed distribution
% al_num:   Number of examples to be selected for labeling 
%           If the number of examples is less than the total query set then select all the examples           
% Returns new_idx: Indices of the examples to be labeled



if(al_type<0||al_type>4)
    error('Incorrect Active Learning Type\n');
end
if(al_num>length(prob_qr))
    warning('More examples to be added than the number of examples remaining. Adding all the remaining examples');
    new_idx=1:length(prob_qr);
    return;
end
if(al_num<0)
    error('al_num cannot be less than zero');
end

m=size(prob_qr,2);

if(verbosity>=2)
    fprintf('Selecting %d examples using %d method\n', al_num, al_type);
end

if(al_type==0)
    %Random selection
    rp=randperm(length(prob_qr));
    new_idx=rp(1:al_num);
elseif(al_type==1)
    %Active Learning according to the diff between the mode and the second highest probability 
    prob_qr=sort(prob_qr,2);
    diff=prob_qr(:,end)-prob_qr(:,end-1);
    [dum new_idx_all]=sort(diff,'ascend');
    new_idx=new_idx_all(1:al_num);
elseif(al_type==2)
    %Active Learning according to the mode of the probability
    [dum new_idx_all]=sort(max(prob_qr,[],2),'ascend');
    new_idx=new_idx_all(1:al_num);
elseif(al_type==3)
    %Active Learning according to the entropy
    prob_qr=abs(min(min(prob_qr)))+prob_qr;%Make probabilities positive
    prob_norm=prob_qr./repmat(sum(prob_qr,2),1,m);%Normalize probabilities
    prob_norm=exp(prob_norm);%exponentiate to increase the effect of difference in probabilities
    prob_norm=prob_norm./repmat(sum(prob_norm,2),1,m);%Normalize again
    H=-sum(prob_norm.*log(prob_norm),2);%Compute Entropy
    [dum new_idx_all]=sort(H,'descend');
    new_idx=new_idx_all(1:al_num);
elseif(al_type==4)
    %Active Learning according to the entropy over trimmed prob.
    %distribution
    med=median(prob_qr,2);
    prob_qr=prob_qr-repmat(med,1,m);
    prob_qr(find(prob_qr)<1e-5)=1e-5;
    prob_qr=prob_qr./repmat(sum(prob_qr,2),1,m);
    H=-sum(prob_qr.*log(prob_qr),2);
    [dum new_idx_all]=sort(H,'descend');
    new_idx=new_idx_all(1:al_num);
end