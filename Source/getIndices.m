%tuvCalc.m



function indices = getIndices(counts, marginalProbs, numAl)
	tuvs = tuvCalc(counts, marginalProbs)

	indices = ones(numAl,1);

end
function TUV = tuvCalc(counts, marginalProbs)
	for i = 1:size(counts, 1)
		alpha = convert(counts(i,:))

	
	end
    
    =

end

function alpha = convert(counts)


end
