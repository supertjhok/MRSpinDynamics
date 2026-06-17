% Calculate effective rotation axis of CPMG refocusing cycle
% Intervals of free precession have zero amplitude
% Returns rotation axis n in [x y z] form
% ------------------------------------------------------------
% Soumyajit Mandal
% Initial version: 09/21/10
% Allow arbitrary pulse amplitudes: 03/03/11

function [n,del_w]=calc_rot_axis_arba(tp,phi,amp)

% Resonance offset distribution
% Uniform distribution of del_w0 (uniform gradient)
maxoffs=20;
numpts=2e3+1;
%del_w=maxoffs*(2*rand(1,numpts)-1); % Random distribution
%del_w=sort(del_w);
del_w=linspace(-maxoffs,maxoffs,numpts); % Deterministic distribution

% Calculate rotational axes (no relaxation or diffusion), [x y z] form
n=zeros(3,1,numpts); ncurr=n; tmp2=n; % Rotation axes
alpha=zeros(1,1,numpts); alpha_curr=alpha; % Rotation angles

% Initial period
if amp(1)>0 % RF pulse
    w1=amp(1);
    Omega=sqrt(w1^2+del_w.^2);
    
    n(1,1,:)=w1*cos(phi(1))./Omega;
    n(2,1,:)=w1*sin(phi(1))./Omega;
    n(3,1,:)=del_w./Omega;
    alpha(1,1,:)=Omega*tp(1);
else % Free precession
    n(3,1,:)=ones(1,1,numpts); % Initial rotation axis is the z-axis
    alpha(1,1,:)=del_w*tp(1);
end

num_pulses=length(phi);
for j=2:num_pulses
    
    if amp(j)>0
        % RF pulses
        w1=amp(j);
        Omega=sqrt(w1^2+del_w.^2);
        
        ncurr(1,1,:)=w1*cos(phi(j))./Omega;
        ncurr(2,1,:)=w1*sin(phi(j))./Omega;
        ncurr(3,1,:)=del_w./Omega;
        alpha_curr(1,1,:)=Omega*tp(j);
        
    else
        % Free precession       
        ncurr(1,1,:)=zeros(1,1,numpts);
        ncurr(2,1,:)=zeros(1,1,numpts);
        ncurr(3,1,:)=ones(1,1,numpts);
        alpha_curr(1,1,:)=del_w*tp(j);
    end
    
    for k=1:numpts
        tmp2(:,:,k)=sin(alpha(:,:,k)/2)*cos(alpha_curr(:,:,k)/2)*n(:,:,k)+...
            cos(alpha(:,:,k)/2).*sin(alpha_curr(:,:,k)/2).*ncurr(:,:,k)-...
            sin(alpha(:,:,k)/2).*sin(alpha_curr(:,:,k)/2).*cross(n(:,:,k),ncurr(:,:,k));
    end
    
    tmp1=cos(alpha/2).*cos(alpha_curr/2)-sin(alpha/2).*sin(alpha_curr/2).*dot(n,ncurr);
    alpha=2*acos(tmp1);
    
    for k=1:numpts
        n(:,:,k)=tmp2(:,:,k)/sin(alpha(:,:,k)/2);
    end
    
end