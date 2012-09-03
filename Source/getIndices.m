%tuvCalc.m
function retIndices = getIndices(counts, marginalProbs, numAl, numClasses)
	tuvs = tuvCalc(counts, marginalProbs, numClasses);
	[tuvs, Indices] = sort(tuvs, 'descend');
	retIndices = Indices(1:numAl);
	

end
function tuv = tuvCalc(counts, marginalProbs, numClasses)

    samples = 1000;
    numQueries = size(counts, 2);
    tuv = zeros(numQueries, 1);
    for i = 1:numQueries
	alpha = convert(counts(:,i), numClasses);
	[p1,p2] = dirich(alpha, samples);
	p2 = max(p2);
	tuv(i) = marginalProbs(i) * (p1 - p2);
    end


end

function alpha = convert(countslice, numClasses)

    prior = ones(numClasses, 1);
    multiplier = numClasses / length(countslice);
    %multiplier = 1 / length(countslice);
    votes = histc(countslice, [1:numClasses]);

    alpha = prior + multiplier * votes;
end
