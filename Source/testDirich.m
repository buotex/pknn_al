%dirich-test.m
n = 3;
dims = 100
masses = zeros(1,dims);
samples = 100;
scale = 1;
bandwidth = zeros(1, steps);
relBandwith = [];
stepwidth = 0.1;
kappas = [0:stepwidth:1]
for k = 1:length(kappas)
  kappa = kappas(k);
  prior = ones(1,dims);
  alpha_mod = cat(2, kappa * n, (1 - kappa) * n/(samples -1 ) * ones(1,dims - 1));
  alpha = prior + alpha_mod;
  %pause(5)
  R = [];
  for i=1:length(alpha) 
      R = cat(2, R, gamrnd(alpha(i), scale, samples, 1));
  end
  %Layout: mat.M = number of samples, mat.N = ndims
  [X,I] = max(R,[], 2);
  counts = transpose(histc(I, [1:dims]));
  masses = counts ./ samples;
  var(masses(:,2:end));
  bandwidth(k) = max(masses(:,2:end)) - min(masses(:,2:end));
  relBandwidth(k) = bandwidth(k) / mean(masses(:,2:end));
  fac = sum(R,2);
  R = R ./ repmat(fac ,1,length(alpha));
  testV = sum(R, 1) / samples;

  alphasum = sum(alpha);
  realV = alpha / alphasum;
  relativeError = (realV - testV) ./ realV;
  sumrelativeError = sum(relativeError.^2);
  [slow, fast] = dirich(alpha);



end
bandwidth;
relBandwidth;
