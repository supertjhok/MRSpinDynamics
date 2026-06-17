% Calculate effective rotation axis of CPMG refocusing cycle
% Intervals of free precession have zero amplitude
% Returns rotation axis n in [x y z] form
% ------------------------------------------------------------
% Soumyajit Mandal
% Initial version: 09/21/10
% Allow arbitrary pulse amplitudes: 03/03/11
% Vectorized code for speedup: 01/06/12
% Use quaternion-like manipulation for significant speedup (~15% for a few
% segments, >3x for many segments): 03/19/13

function [n]=calc_rot_axis_arba3(tp,phi,amp,del_w,plt)

numpts=length(del_w);
zero_tol=1e-12;

% Calculate rotational axes (no relaxation or diffusion), [x y z] form
n=zeros(3,numpts); ncurr=n; tmp=n; % Rotation axes
% alpha = rotation angles

% Initial period
if amp(1)>0 % RF pulse
    w1=amp(1);
    Omega=sqrt(w1^2+del_w.^2);
    
    alpha=Omega*tp(1); sn=sin(alpha/2);
    n(1,:)=sn.*w1*cos(phi(1))./Omega;
    n(2,:)=sn.*w1*sin(phi(1))./Omega;
    n(3,:)=sn.*del_w./Omega;
else % Free precession
    alpha=del_w*tp(1); sn=sin(alpha/2);
    n(3,:)=sn.*ones(1,numpts); % Initial rotation axis is the z-axis
end
cs=cos(alpha/2);

% Compute using sn.n and cs as the propagated variables
num_pulses=length(phi);
for j=2:num_pulses
    
    if amp(j)>0
        % RF pulses
        w1=amp(j);
        Omega=sqrt(w1^2+del_w.^2);
        
        alpha_curr=Omega*tp(j);
        ncurr(1,:)=w1*cos(phi(j))./Omega;
        ncurr(2,:)=w1*sin(phi(j))./Omega;
        ncurr(3,:)=del_w./Omega;
        
        crs=cross(n,ncurr);
        sn_c=sin(alpha_curr/2); cs_c=cos(alpha_curr/2);
        
        tmp(1,:)=cs_c.*n(1,:)+sn_c.*(cs.*ncurr(1,:)-crs(1,:));
        tmp(2,:)=cs_c.*n(2,:)+sn_c.*(cs.*ncurr(2,:)-crs(2,:));
        tmp(3,:)=cs_c.*n(3,:)+sn_c.*(cs.*ncurr(3,:)-crs(3,:));
        cs=cs.*cs_c-sn_c.*dot(n,ncurr);
    else
        % Free precession - simplified calculation, ncurr is simply z-axis
        %ncurr(1,:)=zeros(1,1,numpts);
        %ncurr(2,:)=zeros(1,1,numpts);
        %ncurr(3,:)=ones(1,1,numpts);
        alpha_curr=del_w*tp(j);
        sn_c=sin(alpha_curr/2); cs_c=cos(alpha_curr/2);
        
        tmp(1,:)=cs_c.*n(1,:)-sn_c.*n(2,:);
        tmp(2,:)=cs_c.*n(2,:)+sn_c.*n(1,:);
        tmp(3,:)=cs_c.*n(3,:)+cs.*sn_c;
        cs=cs.*cs_c-sn_c.*n(3,:);
    end
    
    n=tmp;
end

% Calculate final n
alpha=2*acos(cs); sn=sin(alpha/2);
sn(sn==0)=zero_tol; % Identity operation, or zero net rotation points
n(1,:)=n(1,:)./sn;
n(2,:)=n(2,:)./sn;
n(3,:)=n(3,:)./sn;

if plt
    clr={'b--','r--','k--'};
    figure(101);clf;
    for i=1:3
        subplot(3,1,i);
        plot(del_w,n(i,:),clr{i}); hold on;
    end
end