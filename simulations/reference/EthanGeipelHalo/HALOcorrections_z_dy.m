function dx_correction = HALOcorrections_z_dy(Y_T2, STM_T2, mu)
   %% Function Description
    % This function accepts the state vector and state transition matrix at
    % time T2, and uses these values to determine the corrections to z and
    % dy in order to trend towards a stable orbit. 
    % Ideally, this will be called until the dx and dz values at T2 are
    % less than 1e-8.
   %% Ethan Geipel, March 2019
   %% The Function
   % Pull out the state vector components.
     x = Y_T2(1);  y = Y_T2(2);  z = Y_T2(3);
    dx = Y_T2(4); dy = Y_T2(5); dz = Y_T2(6);
    
    % Calculate distances for use in later calculations.
    r1=sqrt((mu+x)^2+(y)^2+(z)^2);
    r2=sqrt((x-1+mu)^2+(y)^2+(z)^2);
    
    % Calculate the x and z accelerations (from CRTBP equations)
    xdd = 2*dy + x - (1-mu)*(x+mu)/r1^3 - mu*(x+mu-1)/r2^3;
    zdd = -z*((1-mu)/r1^3 + mu/r2^3);
    
    % Setting up the expression to calculate correction values (this comes
    % from a paper on 3D HALO Orbit families by Kathleen "Kathy" Cowell).
    A = ([STM_T2(4,3), STM_T2(4,5); STM_T2(6,3), STM_T2(6,5)]-(1/dy)*[xdd; zdd]*[STM_T2(2,3), STM_T2(2,5)]);
    B = [-dx; -dz];
    
    % Calculates the corrections and sets them to return.
    dx_correction = A\B; % Return del_z and del_dy

end