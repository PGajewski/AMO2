clear all

N=[100 500 1000 5000 10000]; % Points numbers.
learning_coef = 0.7; % Coefficient of part into learning nad validating data.
%% Random data
for i=N
    % Data generating
    min=-100;
    max=100;
    X = min + (max-min)*rand(i,5);
    Y = sign(X(:,5));
    n = size(Y);

    % Part data for learning and validating.
    index = floor(learning_coef*size(Y));
    xTrain = X(1:index,:);
    yTrain = Y(1:index);
    xTest = X((index+1):end,:);
    yTest = Y((index+1):end);

    %Counting
    fprintf('For data points: %i\n Start primal task\n', i);
%     [w, b, time_p, fval_p, exitflag_p, output_p ] = svm_primal(xTrain, yTrain, 'sparse');
    [w, b, time_p, fval_p, exitflag_p, output_p ] = svm_primal(xTrain, yTrain);
    acc_p = svm_pred_primal(xTest, yTest, w, b);
    output_p

    fprintf('Start dual task\n');
%     [alfa, time_d, fval_d, exitflag_d, output_d ] = svm_dual(xTrain, yTrain, 'sparse');
    [alfa, time_d, fval_d, exitflag_d, output_d ] = svm_dual(xTrain, yTrain);
    acc_d = svm_pred_dual( xTest, yTest, alfa, xTrain, yTrain );
    output_d
    fprintf('Primal tast result: accuracy=%f, time=%f\n',acc_p, time_p);
    fprintf('Dual tast result: accuracy=%f, time=%f\n',acc_d, time_d); 
end
%% Data from file
% Load data.
X = dlmread('spambase.dat');
Y = X(:, 58);
X(:, 58) = [];

n = length(yTrain);

% Change classes from 0:1 to -1:1
for i=1:n
    if yTrain(i) == 0
        yTrain(i)=-1;
    end
end
% Part data for learning and validating.
index = floor(learning_coef*size(Y));
xTrain = X(1:index,:);
yTrain = Y(1:index);
xTest = X((index+1):end,:);
yTest = Y((index+1):end);

%Counting
fprintf('Start primal task for real data\n');
[w, b, time_p, fval_p, exitflag_p, output_p ] = svm_primal(xTrain, yTrain,'sparse');
acc_p = svm_pred_primal(xTest, yTest, w, b);
output_p

fprintf('Start dual task for real data\n');
[alfa, time_d, fval_d, exitflag_d, output_d ] = svm_dual(xTrain, yTrain,'sparse');
acc_dual = svm_pred_dual( xTest, yTest, alfa, xTrain, yTrain );
output_d
fprintf('Primal tast result: accuracy=%f, time=%f\n',acc_p, time_p);
fprintf('Dual tast result: accuracy=%f, time=%f\n',acc_d, time_d); 