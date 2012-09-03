%dirich.m
%parameters: vector with alpha values
function [p1, p2] = test(Alpha, samples) 
    if nargin < 2
	samples = [100,1000,10000];
	%samples = [100];
    end
    %slow = zeros(length(samples), 1);
    %fast = zeros(length(samples), 1);
    dims = length(Alpha);
    scale = 1;
    for s = 1:length(samples) 
	samples = samples(s);
	R = [];
	for i = 1:length(Alpha)
	    R = cat(2, R, gamrnd(Alpha(i), scale, samples, 1));
	end
	%normalize R
	fac = sum(R,2);
	R = R ./repmat(fac, 1, length(Alpha));

	%fast:
	max_q_n = max(R,[],2);
	result_p2 = 1 - sum(max_q_n) / samples;
	%slow(s) = result_slow;
	p2(s) = result_p2;
    end
     
    p1 = 1 - max(Alpha) / sum(Alpha);


end
