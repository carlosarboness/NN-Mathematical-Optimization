clear;
%
% Parameters.
%
tr_seed = 590152; te_seed = 171251704;
tr_p = 20000; te_q = tr_p/10; tr_freq = 0.0;                  % Datasets generation
epsG = 10^-6; kmax = 1000;                                    % Stopping condition.
ils=3; ialmax = 1; kmaxBLS=10; epsal=10^-3; c1=0.01; c2=0.45; % Linesearch.
icg = 2; irc = 2 ; nu = 1.0;                                  % Search direction.
sg_seed = 565544; sg_al0 = 2; sg_be = 0.3; sg_ga = 0.01;      % SGM iteration.
sg_emax = kmax; sg_ebest = floor(0.01*sg_emax);               % SGM stopping condition.
%
% Optimization
%
global iheader; iheader = 1;
csvfile = strcat('2uo_nn_batch_',num2str(tr_seed),'-',num2str(te_seed),'.csv');
fileID = fopen(csvfile ,'w');
t1=clock;
isds = [1,3,7];
las = [0.01,0.0,0.0];
for num_target = 1:10
    for i = 1:3
        la = las(i);
        isd = isds(i);
        [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex,Lk,go]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu);
        if iheader == 1
            fprintf(fileID,'num_target;      la; isd;  niter;     tex; tr_acc; te_acc;        L*;  ||g*||; Lk\n');
        end
        fprintf(fileID,'         %1i; %7.4f;   %1i; %6i; %7.4f;  %5.1f;  %5.1f;  %8.2e; %8.2e;  %s\n', mod(num_target,10), la, isd, niter, tex, tr_acc, te_acc, fo, go, strjoin(cellstr(num2str(Lk(:))),', '));
        iheader=0;
    end
end
t2=clock;
total_t = etime(t2,t1);
fprintf(' wall time = %6.1d s.\n', total_t);
fclose(fileID);
%uo_nn_batch_BP_log(tr_seed,te_seed,sg_seed, total_t, csvfile);