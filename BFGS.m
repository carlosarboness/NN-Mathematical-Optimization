function [Xmin, k] = BFGS(x1,f,g,epsG,kmax,c1,c2,ialmax,maxiter,eps) 
    n = length(x1); I = eye(n);
    xk = []; x = x1; xk = [xk, x1]; 
    Hk = []; H = eye(n); Hk = [Hk, H]; 
    dk = [];  alk = []; iWk = [];  
    k = 1; alpham = 1; 
    while norm(g(x)) > epsG && k < kmax
        d = -H*g(x); 
        if k > 1
            if ialmax == 1
                alpham = alpham*(g(xk(:,k-1))'*d(:,k-1))/(g(x)'*d); 
            else 
                alpham = 2*(f(x)-f(xk(:,k-1)))/(g(x)'*d); 
            end 
        end 
        [al, iWc] = uo_BLSNW32(f,g,x,d,alpham,c1,c2,maxiter,eps); 
        x = x + al*d;

        sk = x - xk(:,k); 
        yk = g(x)-g(xk(:,k));
        rhok = 1/(yk'*sk); 

        H = (I - rhok*sk*yk')*H*(I - rhok*yk*sk') + rhok*(sk*sk');
        
        k = k + 1; 
        xk = [xk, x]; dk = [dk, d]; alk = [alk, al]; iWk = [iWk, iWc]; Hk(:, :, k) = H;         
    end
        Xmin = xk(:, end); 
end