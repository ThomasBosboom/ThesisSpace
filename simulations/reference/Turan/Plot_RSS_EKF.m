%--------------------------------------------------------------------------
%
%                       Plot RSS EKF results
%
%--------------------------------------------------------------------------
%% 
% Inputs:
% simstep       simulation/measurement step in seconds
% rsspos        RSS position estimation state vector
% rssvel        RSS velocity estimation state vector
% rssposunc     RSS position uncertainty vector
% rssvelunc     RSS velocity uncertainty vector
% ErrYfull      Full Estimation error vector
% SigUncfull    Full Uncertainty vector
% 
% Output:
% Figure 1
% RSS Estimation errors and Uncertainty in m and mm/s
% Figure 2 (if full state == 1)
% Full state estimation plots
%
% Note: Use this file together with CRTBP_OD_EKF_main.m
% (Last modified 22/Feb/2023)
%% 

xlist=linspace(1,30,30);
step=2.6667e-06*simstep;

figure()
subplot(2,1,1);
ss = size(rsspos);
semilogy(real(rsspos)*384747963*1, LineWidth=2, Color ='r')
hold on
semilogy(real(rssposunc)*384747963*1, LineWidth=2, Color ='b')
title('Position', 'fontsize',16,'interpreter','latex')
hold on;
xticks(((1/4.343)/step)*xlist)
xticklabels(string(xlist))

xlabel('Time since Epoch, days', 'fontsize',14,'interpreter','latex')
xlim([1 ss(1)+1])
ylabel({['Estimation Error and Uncertainty'],['1$\sigma$ RSS, m']}, 'fontsize',14,'interpreter','latex')
grid minor

subplot(2,1,2);
ss = size(rssvel);
semilogy(real(rssvel)*1.025351467559547e+03*1000, LineWidth=2, Color ='r')
hold on
semilogy(real(rssvelunc)*1.025351467559547e+03*1000, LineWidth=2, Color ='b')
title('Velocity', 'fontsize',16,'interpreter','latex')
hold on;
grid minor
xticks(((1/4.343)/step)*xlist)
xticklabels(string(xlist))
xlabel('Time since Epoch, days', 'fontsize',14,'interpreter','latex')
xlim([1 ss(1)+1])
ylabel({['Estimation Error and Uncertainty'],['1$\sigma$ RSS, mm/s']}, 'fontsize',14,'interpreter','latex')

if fullstate == 1
    figure()
    subplot(2,1,1);
    ss = size(ErrYfull);
    if ss(2) == 12
      sss = [1:3,7:9];
    else
      sss = [1:3,7:9,13:15];  
    end

    for i=sss
        plot(ErrYfull(:,i)*LU, LineWidth=1, Color ='r');
        hold on
        plot(SigUncfull(:,i)*LU, LineWidth=1, Color ='b');
        hold on
        plot(SigUncfull(:,i)*-LU, LineWidth=1, Color ='b');
    end
    title('Position', 'fontsize',16,'interpreter','latex')
    hold on;
    ylim([-1000 1000])
    xticks(((1/4.343)/step)*xlist)
    xticklabels(string(xlist))
    xlabel('Time since Epoch, days', 'fontsize',14,'interpreter','latex')
    xlim([1 ss(1)+1])
    ylabel({['Estimation Error and Uncertainty'],['1$\sigma$ RSS, m']}, 'fontsize',14,'interpreter','latex')
    grid minor

    subplot(2,1,2)
    if ss(2) == 12
        sss=[4:6,10:12];
    else
        sss=[4:6,10:12,16:18];
    end
    for i=sss
        plot(ErrYfull(:,i)*SU*1000, LineWidth=1, Color ='r');
        hold on
        plot(SigUncfull(:,i)*SU*1000, LineWidth=1, Color ='b');
        hold on
        plot(SigUncfull(:,i)*-SU*1000, LineWidth=1, Color ='b');
    end
    hold on;
    ylim([-14 14])
    grid minor
    xticks(((1/4.343)/step)*xlist)
    xticklabels(string(xlist))
    xlabel('Time since Epoch, days', 'fontsize',14,'interpreter','latex')
    xlim([1 ss(1)+1])
    ylabel({['Estimation Error and Uncertainty'],['1$\sigma$ RSS, mm/s']}, 'fontsize',14,'interpreter','latex')
end

