% Find T1 from 2D Bruker data (IR)
% tvect and T1 are in ms
% necho is the number of echoes (CPMG detection)

function [est_T1]=find_T1(dirname,expno,tvect,necho)

close all;
[data,parameter]=readbrukerfile(dirname,expno);
sizdata=size(data);

pts_echo=sizdata(1)/necho; % Points/echo

amp=zeros(1,sizdata(2));
for i=1:necho
    % Use maximum as echo amplitude
    %amp=amp+max(abs(data(pts_echo*(i-1)+1:pts_echo*i,:)));
    % Use integral as echo amplitude
    amp=amp+trapz(abs(data(pts_echo*(i-1)+1:pts_echo*i,:)));
end

amp=amp/max(amp); % Normalize amplitude

% Restore sign of echo (lost because of abs operator)
[tmp,min_ind]=min(amp);
amp(1:min_ind-1)=-amp(1:min_ind-1);

start = 100; % Initial guess for T1
% Use fminsearch (nonlinear fitting works much better than linear fitting on
% log scale for inversion recovery curves)
% We use an anonymous function to pass additional parameters tvect, amp to the
% output function.
est_T1=fminsearch(@(T1)ir_function(T1,tvect,amp),start)

amp_est=-1+2*(1-exp(-tvect/est_T1));

figure(1);
plot(tvect,amp_est,'b--'); hold on;
plot(tvect,amp,'b*');
xlabel('Time (ms)');
ylabel('Normalized echo amplitude');

title(['Measured T_{1} = ' num2str(round(est_T1)) ' ms']);

function err=ir_function(T1,t,y)

y_est=-1+2*(1-exp(-t/T1));
err=norm(y-y_est);
