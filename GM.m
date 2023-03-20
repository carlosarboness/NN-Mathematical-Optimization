function [Xmin, k] = GM(x1,f,g,epsG,kmax,c1,c2,ialmax,maxiter,eps) 
    xk = []; xk = [xk, x1]; x = x1; 
    alk = []; dk = []; iWk = [];  
    k = 1; alpham = 1; 
    while norm(g(x)) > epsG && k < kmax
        d = -g(x); 
        if k > 1
            if ialmax == 1
                alpham = alpham*(g(xk(:,k-1))'*d(:,k-1))/(g(x)'*d); 
            else 
                alpham = 2*(f(x)-f(xk(:,k-1)))/(g(x)'*d); 
            end 
        end 
        [al, iWc] = uo_BLSNW32(f,g,x,d,alpham,c1,c2,maxiter,eps); 
        x = x + al*d; k = k + 1; 
        xk = [xk, x]; dk = [dk, d]; alk = [alk, al]; iWk = [iWk, iWc];          
    end
        Xmin = xk(:, end); 
end