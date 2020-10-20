function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%C = 0.01*3.^(0:8);
%sigma = 0.01*3.^(0:8);
%error = zeros(length(C),length(sigma));
%for i = 1:length(C)
%  for j = 1:length(sigma)
%    model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
%    predictions = svmPredict(model, Xval);
%    error(i,j) = mean(double(predictions ~= yval));
%  endfor
%endfor
%minimum = min(min(error));
%[x,y]=find(error==minimum)
%C = C(x);
%sigma = sigma(y);
C = 0.27;
sigma = 0.09;
end