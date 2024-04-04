% Plot_RSS_CRTBP
xlist=linspace(1,30,30);
step=2.6667e-06*simstep;

figure()
subplot(2,1,1);
ss = size(rssCRLB);
semilogy(real(rssCRLB)*384747963*1, LineWidth=2)
title('Position', 'fontsize',16,'interpreter','latex')
hold on;
xticks(((1/4.343)/step)*xlist)
xticklabels(string(xlist))

xlabel('Time since Epoch, days', 'fontsize',14,'interpreter','latex')
xlim([1 ss(1)+1])
ylabel('Uncertainty 1$\sigma$ RSS, m', 'fontsize',14,'interpreter','latex')
grid minor

subplot(2,1,2);
ss = size(rssCRLBvel);
semilogy(real(rssCRLBvel)*1.025351467559547e+03*1000, LineWidth=2)
title('Velocity', 'fontsize',16,'interpreter','latex')
hold on;
grid minor
xticks(((1/4.343)/step)*xlist)
xticklabels(string(xlist))
xlabel('Time since Epoch, days', 'fontsize',14,'interpreter','latex')
xlim([1 ss(1)+1])
ylabel('Uncertainty 1$\sigma$ RSS, mm/s', 'fontsize',14,'interpreter','latex')

if fullstate == 1
    figure()
    subplot(2,1,1);
    ss = size(rssCRLB);
    for i=[1:3,7:9]
        semilogy(fullCRLB(:,i)*LU, LineWidth=2);
        hold on
    end
    title('Position', 'fontsize',16,'interpreter','latex')
    hold on;
    xticks(((1/4.343)/step)*xlist)
    xticklabels(string(xlist))
    xlabel('Time since Epoch, days', 'fontsize',14,'interpreter','latex')
    xlim([1 ss(1)+1])
    ylabel('Uncertainty 1$\sigma$ RSS, m', 'fontsize',14,'interpreter','latex')
    grid minor
    legend('x1','y1','z1','x2','y2','z2','fontsize',12,'interpreter','latex')

    subplot(2,1,2);
    ss = size(rssCRLBvel);
    for i=[4:6,10:12]
        semilogy(fullCRLB(:,i)*SU*1000, LineWidth=2);
        hold on
    end
    hold on;
    grid minor
    xticks(((1/4.343)/step)*xlist)
    xticklabels(string(xlist))
    xlabel('Time since Epoch, days', 'fontsize',14,'interpreter','latex')
    xlim([1 ss(1)+1])
    ylabel('Uncertainty 1$\sigma$ RSS, mm/s', 'fontsize',14,'interpreter','latex')
    legend('vx1','vy1','vz1','vx2','vy2','vz2','fontsize',12,'interpreter','latex')
end



