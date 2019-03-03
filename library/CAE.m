function [ W1, W2 ] = CAE(X, S, alpha, beta)
% CAE is Cycle Semantic Auto-encoder
% Inputs:
%    X: dxN data matrix.
%    S: kxN semantic matrix.
%    alpha, beta: regularisation parameter.
%
% Return:
%    W1: kxd projection matrix.
%    W2: dxk projection matrix.

tol = 1e-4;
maxIte = 100;
d = size(X, 1);
k = size(S, 1);
% Initializing optimization variables
W1 = rand(k, d);
W2 = W1';
% Start main loop
ite = 0;
stopW1 = zeros(maxIte, 1);
stopW2 = zeros(maxIte, 1);
while ite < maxIte
    ite = ite + 1;
    W1pre = W1;
    W2pre = W2;
    %% update W1, fix W2
    % min_{W1}||X-W2*W1*X||_F^2+alpha*||S-W1*W2*S||_F^2+beta||W1-W2'||_F^2
    % The solution is equal to solve A * W1 + W1 * B = C, where
    XXT = X*X';
    SST = S*S';
    A = W2'*W2;
    B = (alpha*W2*SST*W2'+beta*eye(d))/XXT;
    C = W2' +alpha*SST*W2'/XXT+beta*W2'/XXT;
    W1 = sylvester(A, B, C);
    
    %% update W2, fix W1
    % min_{W2}||X-W2*W1*X||_F^2+alpha*||S-W1*W2*S||_F^2+beta||W1-W2'||_F^2
    D = alpha*(W1'*W1);
    E = (W1*XXT*W1'+beta*eye(k))/SST;
    F = XXT*W1'/SST+alpha*W1'+beta*W1'/SST;
    W2 = sylvester(D, E, F);
    
    %% check the convergence conditions
    stopW1(ite) = max(max(abs(W1 - W1pre)));
    stopW2(ite) = max(max(abs(W2 - W2pre)));
    if True
        disp(['ite ' num2str(ite) ...
            ', max(||W1-W1pre||)=' num2str(stopW1(iter),'%2.3e') ...
            ', max(||W2-W2pre||)=' num2str(stopW2(iter),'%2.3e')]);
    end
    if stopW1(iter) < tol && stopW2(iter) < tol
        break;
    end
end
W1 = W1';
W2 = W2';
return;
end

