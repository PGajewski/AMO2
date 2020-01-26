function [ acc ] = svm_pred_dual( xTest, yTest, alpha, xTrain, yTrain )

    N = length(yTest);
    predict = sign(sum(((alpha.*yTrain)*ones(1,N)).*(xTrain*xTest'), 1));
    acc = sum(yTest==predict')/N*100;

end