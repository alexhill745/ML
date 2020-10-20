function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);
J = ((sum(((-1*y).*log(h))-((1-y).*log(1-h))))/(m))+((lambda/(2*m))*sum(theta(2:end).^2));
unreg = (X'*(h-y))/m;
temp = theta;
temp(1) = 0;
grad = unreg + ((lambda.*temp)/m);
grad = grad(:);
end