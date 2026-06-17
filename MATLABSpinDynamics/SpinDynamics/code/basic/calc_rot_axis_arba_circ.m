% Calculate effective rotation axis of CPMG refocusing cycle
% Assume a circular distribution of resonance offsets with radius del_wmax
% Intervals of free precession have zero amplitude
% Returns rotation axis n in [x y z] form
% ------------------------------------------------------------
% Soumyajit Mandal
% Initial version: 09/21/10
% Allow arbitrary pulse amplitudes: 03/03/11

function [n,del_w]=calc_rot_axis_arba_circ(tp,phi,amp,del_wmax)

% Create resonance offset distribution
% Note that the density of offset vectors should reflect the density of
% spins in real space, not resonance offset space. However the spin
% dynamics calculations are performed in the latter, so we must convert
% appropriately. The scaling relationship is del_w = (gamma*g/w_1) x del_z

% Calculate maximum offset to simulate, so del_wmax = maxoffs x w_1
if del_wmax<20
    maxoffs=del_wmax;
else
    maxoffs=20;
end

numpts=1e4; % Maximum possible number of points
numbins=1e2; % Number of bins to quantize spin distribution
offbin=maxoffs/numbins; % Width of each bin
% Since gamma, g, w_1 are given, we can write del_z = alpha * del_w, where
% alpha = w_1/(gamma*g) is a constant. Thus r = alpha * del_wmax, where r
% is the physical radius of the sample. Then alpha = r/del_wmax, so we have
% del_z = (r/del_wmax) x del_w. Since r is given, we can scale the number of 
% vectors per bin corresponding to probability = 1 by 1/del_wmax to ensure
% constant vector density in real space (del_z).
ptsbinmax=numpts*maxoffs/(del_wmax*numbins); % Probability = 1

count=0;
del_w=[]; % Generate offset vectors
for j=1:numbins
    del_wbin=-maxoffs+2*(j-0.5)*offbin; % Location of current bin
    if abs(del_wbin)<del_wmax % Check if any spins exist
        ptsbin=ceil(ptsbinmax*sqrt(1-(del_wbin/del_wmax)^2)); % Probability = circle
        del_w(count+1:count+ptsbin)=...
            del_wbin+linspace(-offbin*(1-1/ptsbin),offbin,ptsbin); % Distribution of vectors
        count=count+ptsbin;
    end
end
numpts=count; % Actual number of points

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