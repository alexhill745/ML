function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
poly = zeros(numel(X), p);
for i = 1:p
  X_poly(:, i) = [X.^i];
end
end