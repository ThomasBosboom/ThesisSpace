function [Yi, STM] = xSTM(Y)
    Yi = Y(1:6);
    
    STM(:,1) = Y(7:12);
    STM(:,2) = Y(13:18);
    STM(:,3) = Y(19:24);
    STM(:,4) = Y(25:30);
    STM(:,5) = Y(31:36);
    STM(:,6) = Y(37:42);
    
    % STM = diag(STM);
end