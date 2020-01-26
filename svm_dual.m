function [alpha, time, fval, exitflag, output] = svm_dual( xTrain, yTrain, mode, options )

    %Initialization
    C = 1;
    [N, D] = size(xTrain);
    if nargin >= 3 && strcmp(mode,'sparse')
    %Initialization
        K = sparse(xTrain*xTrain');
        H = sparse(2*diag(yTrain)*K*diag(yTrain));
        f = sparse(-ones(N,1));
        Aeq = sparse(yTrain');
        beq = 0;
        LB = sparse(N,1);
        UB = sparse(C*ones(N,1));
    else
        K = xTrain*xTrain';
        H = 2*diag(yTrain)*K*diag(yTrain);
        f = -ones(N,1);
        Aeq = yTrain';
        beq = 0;
        LB = zeros(N,1);
        UB = C*ones(N,1);
    end
    if nargin < 4
        options = optimoptions(@quadprog,'Algorithm','interior-point-convex','Display','off');
    end
    
    %Counting
    tic;
    [alpha,fval,exitflag,output] = quadprog(H, f, [], [], Aeq, beq, LB, UB, [], options);
    time = toc;

end