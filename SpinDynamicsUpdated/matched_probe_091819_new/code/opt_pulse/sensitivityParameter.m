sens = 1e-5:1e-5:1e-3;
SNRhard = zeros(length(sens));
for j = 1:length(sens)
    vars.sens = sens(j);
    vars.rat = 74/37; %t180/t90
    vars.len = 1;
    vars.techo=15*2*pi/2;
     [masy,SNRhard(j)] = plot_hardRef_tuned_probe_lp(vars);
end

figure
plot(sens,SNRhard)