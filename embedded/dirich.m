%dirich.m
%parameters: vector with alpha values
function [slow, fast] = test(Alpha, Samples) 
if nargin < 2
	Samples = [100,1000,10000];
	%Samples = [100];
end
slow = zeros(length(Samples), 1);
fast = zeros(length(Samples), 1);
dims = length(Alpha);
scale = 1;
for s = 1:length(Samples) 
samples = Samples(s);
R = [];
for i = 1:length(Alpha)
	R = cat(2, R, gamrnd(Alpha(i), scale, samples, 1));
end
%normalize R
fac = sum(R,2);
R = R ./repmat(fac, 1, length(Alpha));
%slow:
result_slow = 0;
for j = 1 : samples
  val = sum(R(j,:)) - max(R(j,:));
  result_slow = result_slow + val;
end
result_slow = result_slow / samples;

%fast:
q_y = max(R,[],2);
result_fast = 1 - sum(q_y) / samples;

slow(s) = result_slow;
fast(s) = result_fast;
end
slow
fast
end
