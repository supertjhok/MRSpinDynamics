% Calculate on-resonance rotation of CPMG with BS shift
% th_exc -> excitation angle
% th_exc*rat -> refocusing angle
% th_bs -> BS shift angle

function calc_cpmg_rot(th_exc,rat,th_bs)

NE=100;
init=[0 % z-magnetization
    0
    1];
ux=zeros(1,NE); uy=ux; uz=ux;

u=rot_x(init,th_exc); % Excitation
u=rot_z(u,th_bs); % B-S shift
for i=1:NE
    u=rot_y(u,rat*th_exc);
    u=rot_z(u,2*th_bs);
    ux(i)=u(1); uy(i)=u(2); uz(i)=u(3);
end

figure(1); clf;
plot(ux,'bo-'); hold on;
plot(uy,'r*-');
plot(uz,'kd-');

figure(2); clf;
plot(atan(ux./uy)*180/pi,'m.-');

function [out]=rot_x(in,theta)

Rx=[1 0 0
    0 cos(theta) -sin(theta)
    0 sin(theta) cos(theta)];
out=Rx*in;

function [out]=rot_y(in,theta)

Ry=[cos(theta) 0 sin(theta)
    0 1 0
    -sin(theta) 0 cos(theta)];
out=Ry*in;

function [out]=rot_z(in,theta)

Rz=[cos(theta) -sin(theta) 0
    sin(theta) cos(theta) 0
    0 0 1];
out=Rz*in;

