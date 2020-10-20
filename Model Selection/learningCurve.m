function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve

% Number of training examples
m = size(X, 1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
for i = 1:m
  X_temp = X(1:i, :);
  y_temp = y(1:i);
  theta = trainLinearReg(X_temp, y_temp, 1);
  error_train(i) = linearRegCostFunction(X_temp, y_temp, theta, 0);
  error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
end