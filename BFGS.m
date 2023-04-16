function [x,k,Lk,bls_t] = BFGS(x,f,g,epsG,kmax,c1,c2,ialmax,maxiter,eps) 
    n = length(x); I = eye(n);
    H = I; Lk = f(x); bls_t = [];
    k = 1; alpham = 1; 
    while norm(g(x)) > epsG && k < kmax
        d = -H*g(x); 
        if k > 1
            if ialmax == 1
                alpham = al*(g(prev_x)'*prev_d)/(g(x)'*d); 
            else 
                alpham = 2*(f(x)-f(prev_x))/(g(x)'*d); 
            end 
        end 
        t1 = clock;
        [al, iWc] = uo_BLSNW32(f,g,x,d,alpham,c1,c2,maxiter,eps);
        t2 = clock;
        bls_t = [bls_t etime(t2,t1)];
        prev_x = x; prev_d = d;
        x = x + al*d;

        sk = x - prev_x; 
        yk = g(x)-g(prev_x);
        rhok = 1/(yk'*sk); 

        H = (I - rhok*sk*yk')*H*(I - rhok*yk*sk') + rhok*(sk*sk');
        
        k = k + 1;
        Lk = [Lk f(x)];
    end 
end