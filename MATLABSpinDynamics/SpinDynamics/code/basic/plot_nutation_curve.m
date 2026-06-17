% Plot nutation curve from Hahn echo data
% T -> length of T90 pulse (us)

function [echo_int]=plot_nutation_curve(filname,exptnum,T)

[data,parameter] = readbrukerfile(filname,exptnum);

sizdata=size(data);

adata=abs(data);

echo_int=zeros(1,sizdata(2));
% Integrate the squared echo amplitude
for i=1:sizdata(2)
    echo_int(i)=trapz(adata(:,i).^2)/sizdata(1);
end

echo_int=echo_int/max(echo_int); % Normalize echo integral

figure(1); %clf;
plot(T,echo_int,'ko-','LineWidth',2); hold on;
set(gca,'FontSize',14);
xlabel('T_{90}, \mus');
ylabel('Normalized echo SNR');


