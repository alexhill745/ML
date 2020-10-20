function [X_norm, mu, sigma] = featureNormalize(X)
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));      
for i = 1:size(X,2)
  feat = X(:,i);
  mu(1,i) = mean(feat);
  sigma(1,i) = std(feat);
  temp = feat-mu(1,i);
  ans(:,i) = temp./sigma(1,i);
end
X_norm = ans;
end
