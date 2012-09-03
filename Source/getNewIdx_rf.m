function counts = getNewIdx_rf(Kernel, trainingIndices, queryIndices, labels, numClasses, al_num)
cd 'embedded'
%TODO: rfpred doesn't have to return al_num valid indices, so perhaps filter out the invalid ones
%Passing Matlab data (hopefully read-only, should be at least if I understood it correctly...) to C code, which then passes it to lua.
%The following parameters are currently used:
%The kernel matrix, with _all_ entries (which means training and candidate data), the indices currently used for the training data and the indices of the kernel entries used for the query data.
%The question stands: How does one use the kernel here?
% -> The random forest creation is dependent on that!
numTrees = 100;
%counts format: 
%one Row of data for every tree,
%numTrees x #queryIndices matrix, containing
%response-indices.
counts = rfpred(Kernel, trainingIndices, queryIndices, labels, numClasses ,numTrees);

%new_idx = new_idx([new_idx > 0])
cd '..'
end
