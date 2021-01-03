function analyze_arbphase

dat=load('sim_results\RP2-1.0\results_exc_ref_111210.mat');
results=dat.results;

siz=size(results);
numexp=siz(1);

del_ph=results(:,41); % Phase of excitation pulse

close all;

figure(1);
plot(del_ph/(pi/2),'b.');
ylabel('Phase (units of \pi/2)');
input('Press any key to continue...')

% Sort into categories based on excitation phase
cat=mod(round(del_ph/(pi/2)),4); % Category = Phase (units of pi/2)

echo_pk=zeros(4,1); echo_rms=echo_pk;
cnts=zeros(4,1);

tp=zeros(20,4); ph=zeros(20,4);
for i=1:4
    cnts(i)=sum(cat==(i-1));
    echo_pk(i)=mean(results(cat==(i-1),42));
    echo_rms(i)=mean(results(cat==(i-1),42));
end

cnts/numexp % Histogram
echo_pk % Average peak
echo_rms % Average RMS

results_0=zeros(cnts(1),43);
results_1=zeros(cnts(2),43);
results_2=zeros(cnts(3),43);
results_3=zeros(cnts(4),43);

tmp=zeros(4,1);
for i=1:numexp
    if cat(i)==0
        tmp(1)=tmp(1)+1;
        results_0(tmp(1),:)=results(i,:);
    end
    if cat(i)==1
        tmp(2)=tmp(2)+1;
        results_1(tmp(2),:)=results(i,:);
    end
    if cat(i)==2
        tmp(3)=tmp(3)+1;
        results_2(tmp(3),:)=results(i,:);
    end
    if cat(i)==3
        tmp(4)=tmp(4)+1;
        results_3(tmp(4),:)=results(i,:);
    end
end

[val_0,ind_0]=max(results_0(:,43));
[val_1,ind_1]=max(results_1(:,43));
[val_2,ind_2]=max(results_2(:,43));
[val_3,ind_3]=max(results_3(:,43));

echo_pks=zeros(5,1); echo_rms=echo_pks;
[echo_pks(2),echo_rms(2)]=cpmg_van_spin_dynamics_plot_delph...
    (results_0(ind_0,1:20),results_0(ind_0,21:40),results_0(ind_0,41),20,10,800,100,100,'b');
[echo_pks(4),echo_rms(4)]=cpmg_van_spin_dynamics_plot_delph...
    (results_2(ind_2,1:20),results_2(ind_2,21:40),results_2(ind_2,41),20,10,800,100,100,'k');

input('Press any key to continue...')
close all;

[echo_pks(3),echo_rms(3)]=cpmg_van_spin_dynamics_plot_delph...
    (results_1(ind_1,1:20),results_1(ind_1,21:40),results_1(ind_1,41),20,10,800,100,100,'r');
[echo_pks(5),echo_rms(5)]=cpmg_van_spin_dynamics_plot_delph...
    (results_3(ind_3,1:20),results_3(ind_3,21:40),results_3(ind_3,41),20,10,800,100,100,'m');

% Normalize to rectangular case
[echo_pks(1),echo_rms(1)]=cpmg_van_spin_dynamics_plot('rectangular',20,10,800,100,100); % Rectangular case
echo_pks=echo_pks/echo_pks(1)
echo_rms=echo_rms/echo_rms(1)

% Write pulses
phas=results_0(ind_0,21:40)*180/pi; % Convert to degrees
gen_shape_texc('RESULTS_0',1000,results_0(ind_0,1:20),ones(1,20),phas);
gen_shape_texc('RESULTS_1',1000,results_1(ind_1,1:20),ones(1,20),phas+90);
gen_shape_texc('RESULTS_2',1000,results_2(ind_2,1:20),ones(1,20),phas+180);
gen_shape_texc('RESULTS_3',1000,results_3(ind_3,1:20),ones(1,20),phas+270);

% Pulse lengths
plen=zeros(4,1);
plen(1)=sum(results_0(ind_0,1:20))/20;
plen(2)=sum(results_1(ind_1,1:20))/20;
plen(3)=sum(results_2(ind_2,1:20))/20;
plen(4)=sum(results_3(ind_3,1:20))/20;

plen
