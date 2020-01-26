function [ w, b, time, fval, exitflag, output ] = svm_primal( xTrain, yTrain, mode, options )

    %Initialization
    C = 1;
    [N, D] = size(xTrain);
    if nargin >= 3 && strcmp(mode,'sparse')
        H = sparse(D+1+N,D+1+N);
        H(1:D, 1:D) = speye(D);
        f = sparse([ zeros(D+1,1); C*ones(N,1) ]);
        A = sparse([ -(yTrain*ones(1,D)).*xTrain, -yTrain, -eye(N) ]);
        b = sparse(-1*ones(N,1));
        LB = sparse([ -Inf*ones(D+1,1); zeros(N,1) ]);
        UB = sparse(Inf*ones(D+1+N, 1));
    else
        H = zeros(D+1+N);
        H(1:D, 1:D) = eye(D);
        f = [ zeros(D+1,1); C*ones(N,1) ];
        A = [ -(yTrain*ones(1,D)).*xTrain, -yTrain, -eye(N) ];
        b = -1*ones(N,1);
        LB = [ -Inf*ones(D+1,1); zeros(N,1) ];
        UB = [ Inf*ones(D+1+N, 1) ];        
    end

    if nargin < 4
        options = optimoptions(@quadprog,'Algorithm','interior-point-convex','Display','off');
    end
    
    %Counting
    tic;
    [X,fval,exitflag,output] = quadprog(H, f, A, b, [], [], LB, UB, [], options);
    time = toc;
    w = X(1:D);
    b = X(D+1);
end