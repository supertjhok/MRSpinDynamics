% Calculate effective rotation axis of CPMG refocusing cycle
% Each RF pulse has delays before and after it, so length(tf)=length(tp)+1
% Soumyajit Mandal
% 09/21/10

function n=calc_rot_axis(T_90,tp,phi,tf)

% Units conversions
T_90=T_90/1e6;
tp=tp/1e6;
tf=tf/1e6;
i=sqrt(-1);

% Uniform distribution of del_w0 (uniform gradient)
maxoffs=10;
numpts=1e3;

w1=pi/(2*T_90);
del_w=w1*linspace(-maxoffs,maxoffs,numpts);

Omega=sqrt(w1^2+del_w.^2);

% Calculate rotational axes (no relaxation or diffusion), [x y z] form
n=zeros(3,1,numpts); ncurr=n; tmp2=n; % Rotation axes
alpha=zeros(1,1,numpts); alpha_curr=alpha; % Rotation angles

% Initial free precession period
alpha(1,1,:)=del_w*tf(1);
n(3,1,:)=ones(1,1,numpts); % Initial rotation axis is the z-axis

num_pulses=length(phi);
for j=1:num_pulses
    % RF pulses
    ncurr(1,1,:)=w1*cos(phi(j))./Omega;
    ncurr(2,1,:)=w1*sin(phi(j))./Omega;
    ncurr(3,1,:)=del_w./Omega;
    alpha_curr(1,1,:)=Omega*tp(j);
    
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
    
    if tf(j+1)>0
        % Free precession
        ncurr(1,1,:)=zeros(1,1,numpts);
        ncurr(2,1,:)=zeros(1,1,numpts);
        ncurr(3,1,:)=ones(1,1,numpts);
        alpha_curr(1,1,:)=del_w*tf(j+1);
        
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
end

% Convert to [0 - +] form
tmp2=n;
n(1,1,:)=tmp2(3,1,:);
n(2,1,:)=tmp2(1,1,:)-i*tmp2(2,1,:);
n(3,1,:)=tmp2(1,1,:)+i*tmp2(2,1,:);