function new_idx = getNewIdx_rf(Kernel, trainingIndices, queryIndices, al_num)
cd 'embedded'
%TODO: rfpred doesn't have to return al_num valid indices, so perhaps filter out the invalid ones
new_idx = rfpred(Kernel, trainingIndices, queryIndices, al_num);
new_idx = new_idx([new_idx > 0])
cd '..'
end
