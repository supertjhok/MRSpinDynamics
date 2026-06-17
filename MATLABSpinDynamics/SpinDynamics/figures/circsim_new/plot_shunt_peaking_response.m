% Plot BW and settling time of shunt-peaking network as a function of
% capacitor size

function plot_shunt_peaking_response(alpha,clr)

L=1; R1=0.75; R=0.25;
tau=L/(R+R1);
R2=alpha*R1;

ST=0.05; % settling time threshold
m=0:0.05:2;

s=tf('s');
bw=zeros(1,length(m)); sett=bw;
for i=1:length(m)
    C=m(i)*(L/R1)/(R1+R2);
    tf_sp=(s*C*(R1+R2)+1)/(s^2*tau*C*(R1+R2)+s*(tau+C*(R*(R1+R2)+R1*R2)/(R1+R))+1);
    bw(i)=bandwidth(tf_sp); % 3 dB bandwidth
    tmp=stepinfo(tf_sp,'SettlingTimeThreshold',ST);
    sett(i)=tmp.SettlingTime;
end

figure(1);
plot(m,bw,clr); hold on;
ylabel('3dB bandwidth');
xlabel('m');

figure(2);
plot(m,sett,clr); hold on;
ylabel('Settling time');
xlabel('m');