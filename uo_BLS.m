function [al, iWout] = uo_BLS(x, d, f, g, h, almax, almin, rho, c1, c2, iW)
    if iW == 1 
        Q = h(x); 
        al = -(g(x)'*d)/(d'*Q*d);
        iWout = 3; 
    else 
        al = almax; 
        while al > almin && not(wolfe_conditions(x, d, al, f, g, c1, c2, iW))
            al = rho*al; 
        end
        iWout = uo_iWout(x, d, al, f, g, c1, c2);
    end 
end 