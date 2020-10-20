function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));
h = X*theta;
J = ((sum((h-y).^2))/(2*m))+((lambda*sum(theta(2:end).^2)/(2*m)));
unreg = (X'*(h-y))/m;
temp = theta;
temp(1) = 0;
grad = unreg + ((lambda.*temp)/m);
grad = grad(:);
end