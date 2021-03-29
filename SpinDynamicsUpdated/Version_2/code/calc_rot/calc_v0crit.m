% Calculate critical velocity parameter for a given refocusing cycle
% ------------------------------------------------------------
% Written by: Soumyajit Mandal
% Initial version: 03/23/21

% [n,alpha] = pre-calculated refocusing axis and rotation angle vectors
function [v0crit]=calc_v0crit(del_w,n,alpha,plt)

numpts=length(del_w);

d_del_w=diff(del_w); % Find derivative

% Estimate centered variables
del_w_c=(del_w(1:numpts-1)+del_w(2:numpts))/2;
alpha_c=(alpha(1:numpts-1)+alpha(2:numpts))/2;

% Calculate cross product
n1=n(2,1:numpts-1).*n(3,2:numpts)-n(3,1:numpts-1).*n(2,2:numpts);
n2=n(3,1:numpts-1).*n(1,2:numpts)-n(1,1:numpts-1).*n(3,2:numpts);
n3=n(1,1:numpts-1).*n(2,2:numpts)-n(2,1:numpts-1).*n(1,2:numpts);
ncross=sqrt(n1.*n1+n2.*n2+n3.*n3);

% Calculate v0crit
v0crit=alpha_c.*d_del_w./ncross;

% Interpolate back to original grid
v0crit = interp1(del_w_c,v0crit,del_w,'linear','extrap');

% Plot results
if plt
    figure(12);
    semilogy(del_w,v0crit,'LineWidth',1);
    xlabel('\Delta\omega_0/\omega_{1}'); ylabel('v_{0,crit}');
    set(gca,'FontSize',14); set(gca,'FontWeight','Bold');
end