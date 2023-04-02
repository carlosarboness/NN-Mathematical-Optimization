function [x, k,Lk] = GM(x,f,g,epsG,kmax,c1,c2,ialmax,maxiter,eps) 
    Lk = f(x);  
    k = 1; alpham = 1;
    while norm(g(x)) > epsG && k < kmax
        d = -g(x); 
        if k > 1
            if ialmax == 1
                alpham = al*(g(prev_x)'*prev_d)/(g(x)'*d); 
            else 
                alpham = 2*(f(x)-f(prev_x))/(g(x)'*d); 
            end 
        end 
        [al, iWc] = uo_BLSNW32(f,g,x,d,alpham,c1,c2,maxiter,eps);
        prev_x = x; prev_d = d;
        x = x + al*d; k = k + 1;
        Lk = [Lk f(x)];
    end
end