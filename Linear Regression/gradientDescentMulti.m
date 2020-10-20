function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
for i = 1:size(X,2)
  temp(i) = theta(i,1) - (alpha*(1/m)*(sum(((X*theta)-y).*X(:,i))));
end
    theta = temp';
    J_history(iter) = computeCostMulti(X, y, theta);
end
end
