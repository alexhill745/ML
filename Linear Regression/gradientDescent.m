function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
    temp1 = theta(1,1) - (alpha*(1/m)*(sum(((X*theta)-y).*X(:,1))));
    temp2 = theta(2,1) - (alpha*(1/m)*(sum(((X*theta)-y).*X(:,2))));
    theta(1,1) = temp1;
    theta(2,1) = temp2;
    J_history(iter) = computeCost(X, y, theta);
end
end
