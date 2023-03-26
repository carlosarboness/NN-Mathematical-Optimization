function [wo,niter,Lk] = SGM(w,L,gL,Xtr,ytr,Xte,yte,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed)
% Stochastic Gradient Method
%
% INPUTS
% w: initial weight vector
% L: loss function
% gL: gradient of loss function
% Xtr: training data
% ytr: training labels
% Xte: test data
% yte: test labels
% sg_al0: \alpha^{SG}_0
% sg_be: \beta^{SG}
% sg_ga: \gamma^{SG}
% sg_emax: maximum number of epochs
% sg_ebest: number of epochs to wait for improvement
% sg_seed: random seed

% OUTPUTS
% wo: final weight vector
% niter: number of iterations

% initialize
p = size(Xtr,2); m = floor(sg_ga*p); ke = ceil(p/m);
kmax = ke*sg_emax; sg_al = 0.01*sg_al0; sg_k = floor(sg_be*kmax);
e = 0; s = 0; k = 0; L_best = inf; niter = 0;
Lk = [];

rng(sg_seed); % initialize random seed

while (e < sg_emax) && (s < sg_ebest)
    % shuffle data
    idx = randperm(p);
    p_Xtr = Xtr(:,idx); p_ytr = ytr(idx);
    
    % loop over minibatches
    for j = 1:ke
        S = (j-1)*m+1:j*m;

        d = -gL(w,p_Xtr(:,S),p_ytr(S));

        if k <= sg_k
            al = (1-k/sg_k)*sg_al0+(k/sg_k)*sg_al;
        else
            al = sg_al;
        end

        w = w+al*d; k = k+1; niter = niter+1;
    end
    e = e+1; L_te = L(w,Xte,yte); Lk = [Lk L_te];
    if L_te < L_best
        L_best = L_te; wo = w; s = 0;
    else
        s = s+1;
    end
end

end