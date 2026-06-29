function [results_sort,echo_ms_length]=oct_exc_pulse_mag_results_all

% 3: Original constant-amplitude OCT excitation pulse
% 4: Shorter OCT excitation pulses (fewer segments)
% 5: Shorter OCT excitation pulses (shorter segments, linear decrease)
% 6: Shorter OCT excitation pulses (shorter segments, geometric decrease)
% 7: Longer OCT excitation pulses (more segments)
% 8: Longer OCT excitation pulses (more segments) -> after re-optimization
% 9: Longer OCT excitation pulses (more segments) -> using Colm's code
% 10: Longer OCT excitation pulses (more segments) -> using Colm's code
% after re-optimization

results={}; echo_rms=[];
count=1;

% Reference values - rect(90) / rect(180)
echo_pk_ref=119.6989;
echo_rms_ref=162.2517;

% Original constant-amplitude OCT excitation pulse
tmp=load('dat_files\results_mag3_rms_2.mat');
results{count,1}=tmp.texc;
results{count,2}=tmp.pexc;
results{count,3}=tmp.echo_pk/echo_pk_ref;
results{count,4}=tmp.echo_rms/echo_rms_ref;
echo_rms(count)=results{count,4};

% New OCT excitation pulses
for expt=4:10
    filname=['dat_files\results_mag' num2str(expt) '.mat'];
    tmp=load(filname);
    tmp=tmp.results;
    sizres=size(tmp);
    for i=1:sizres(1)
        count=count+1;
        for j=1:2
            results{count,j}=tmp{i,j};
        end
        results{count,3}=tmp{i,3}/echo_pk_ref;
        results{count,4}=tmp{i,4}/echo_rms_ref;
        echo_rms(count)=results{count,4};
    end
end

% Sort by echo RMS
[~,ind]=sort(echo_rms,2,'descend');

echo_ms_length=zeros(length(echo_rms),3);
results_sort={};
for i=1:length(echo_rms)
    for j=1:4
        results_sort{i,j}=results{ind(i),j};
    end
    echo_ms_length(i,1)=i; % Index
    echo_ms_length(i,2)=mean(results_sort{i,1})/pi; % Segment length (units of T_180)
    echo_ms_length(i,3)=sum(results_sort{i,1})/pi; % Total length (units of T_180)
    echo_ms_length(i,4)=results_sort{i,4}.^2; % MS
end