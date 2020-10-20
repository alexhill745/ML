function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);
J = ((sum(((-1*y).*log(h))-((1-y).*log(1-h))))/(m))+((lambda/(2*m))*sum(theta(2:end).^2));
grad(1) = (1/m)*(sum(((h)-y).*X(:,1)));
for i = 2:size(X,2)
  grad(i) = ((1/m)*(sum(((h)-y).*X(:,i))))+((lambda*theta(i))/m);
end
end