%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OM / GCED / F.-Javier Heredia https://gnom.upc.edu/heredia
% Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
%
% Input parameters:
%
% num_target : set of digits to be identified.
%    tr_freq : frequency of the digits target in the data set.
%    tr_seed : seed for the training set random generation.
%       tr_p : size of the training set.
%    te_seed : seed for the test set random generation.
%       te_q : size of the test set.
%         la : coefficient lambda of the decay factor.
%       epsG : optimality tolerance.
%       kmax : maximum number of iterations.
%        ils : line search (1 if exact, 2 if uo_BLS, 3 if uo_BLSNW32)
%     ialmax :  formula for the maximum step lenght (1 or 2).
%    kmaxBLS : maximum number of iterations of the uo_BLSNW32.
%      epsal : minimum progress in alpha, algorithm up_BLSNW32
%      c1,c2 : (WC) parameters.
%        isd : optimization algorithm.
%     sg_al0 : \alpha^{SG}_0.
%      sg_be : \beta^{SG}.
%      sg_ga : \gamma^{SG}.
%    sg_emax : e^{SGÇ_{max}.
%   sg_ebest : e^{SG}_{best}.
%    sg_seed : seed for the first random permutation of the SG.
%        icg : if 1 : CGM-FR; if 2, CGM-PR+      (useless in this project).
%        irc : re-starting condition for the CGM (useless in this project).
%         nu : parameter of the RC2 for the CGM  (useless in this project).
%
% Output parameters:
%
%    Xtr : X^{TR}.
%    ytr : y^{TR}.
%     wo : w^*.
%     fo : {\tilde L}^*.
% tr_acc : Accuracy^{TR}.
%    Xte : X^{TE}.
%    yte : y^{TE}.
% te_acc : Accuracy^{TE}.
%  niter : total number of iterations.
%    tex : total running time (see "tic" "toc" Matlab commands).
%
    tic     % inicialize counter 
    
    [Xtr, ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq); % training set
    
    % activation function (sigmoid function) 
    sig = @(Xtr) 1./(1+exp(-Xtr));
  
    % output signal (recognition)
    y = @(Xtr,w) sig(w'*sig(Xtr));  
    
    % Mean loss function (objective function) 
    L = @(w,Xtr,ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2)+(la*norm(w)^2)/2; 

    % Gradient of the objective funciton
    gL = @(w,Xtr,ytr) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;
    
    w1 = rand(length(Xtr(:,1)),1); % initial random solution vector

    L = @(w) L(w, Xtr, ytr);      % evaluate at Xtr and ytr
    gL = @(w) gL(w, Xtr, ytr);    % evaluate at Xtr and ytr

    method = 'Gradient'; 
    switch method 
        case 'Gradient'
            [wo, niter] = GM(w1,L,gL,epsG,kmax,c1,c2,ialmax,kmaxBLS,epsal);  
        case 'QNM'
            [wo, niter] = BFGS(w1,L,gL,epsG,kmax,c1,c2,ialmax,kmaxBLS,epsal);  
    end

    fo = Lw(wo); % minimum value of the objective function
    
    delta = @(x, y) double(x==y); % kronecker delta, returns 1 if x & y are equal, returns 0 otherwise
   
    tr_acc = (100/tr_p)*sum(delta(round(y(Xtr, wo)),ytr)); % training accuracy (%)

    [Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, 0.0); % testing set

    te_acc = (100/te_q)*sum(delta(round(y(Xte, wo)),yte)); % test accuracy (%)

    tex = toc; 

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
