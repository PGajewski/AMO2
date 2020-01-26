function [ acc ] = svm_pred_primal( xTest, yTest, w, b )

    N = length(yTest);
    pred = sign(xTest*w + b);
    acc = sum(pred==yTest)/N * 100;

end