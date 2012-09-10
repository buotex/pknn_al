numLSamples = 200
numULSamples = numLSamples
%Considering an uniform distribution:
%emulate a tree:
numtrees = numLSamples + numULSamples
x = []
y = []
for m = 5:101
	numPerClass = numLSamples / m;
	labels = [];
	for i = 1:m
		temp = i * ones(1,floor(numPerClass));
		labels = [labels, temp];
	end
	labels = [labels , zeros(1, numULSamples)];
	indices = randi([1, length(labels)], 1, numtrees);
	picks = labels(indices);
	votes = histc(picks ,[1:m]) ;
       	
	x = [x m * ones(1, m)];
	y = [y, votes];
	

end
h = figure('Visible', 'off');
plot(x,y, 'x')
hold;
xlabel('#Classes')
ylabel('#Votes')
title(sprintf('Distribution of alpha values with growing dimension'));
print(h, '-dpng', 'alpha.png')

