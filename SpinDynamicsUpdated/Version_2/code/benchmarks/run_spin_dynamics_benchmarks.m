% RUN_SPIN_DYNAMICS_BENCHMARKS
% Run small performance benchmarks for Version 2 spin-dynamics routines.
%
% Signature
%   results = run_spin_dynamics_benchmarks()
%   results = run_spin_dynamics_benchmarks('UseTimeit', true)
%
% Inputs
%   Name-value pairs:
%     UseTimeit - Logical flag. If true and timeit is available, use timeit.
%       Default: true.
%     Repetitions - Number of runs used when UseTimeit is false. Default: 3.
%     SaveResults - Logical flag. If true, save a MAT-file in this folder.
%       Default: false.
%     OutputFile - MAT-file name used when SaveResults is true.
%       Default: benchmark_results.mat.
%
% Outputs
%   results - Table containing benchmark names, elapsed time, checksums, and
%     short descriptions.
%
% Notes
%   Each benchmark uses persistent setup data so the timed region focuses on
%   numerical routines instead of path setup or parameter construction.
% -------------------------------------------------------------------------

function results = run_spin_dynamics_benchmarks(varargin)

p = inputParser;
addParameter(p,'UseTimeit',true,@islogical);
addParameter(p,'Repetitions',3,@(x) isnumeric(x) && isscalar(x) && x >= 1);
addParameter(p,'SaveResults',false,@islogical);
addParameter(p,'OutputFile','benchmark_results.mat',@(x) ischar(x) || isstring(x));
parse(p,varargin{:});

this_dir = fileparts(mfilename('fullpath'));
code_dir = fileparts(this_dir);
addpath(genpath(code_dir));

benchmarks = {
    'arb10_kernel_small', @benchmark_arb10_kernel_small, ...
    'Current precomputed-rotation arbitrary-pulse kernel';
    'arb8_legacy_convolution_small', @benchmark_arb8_legacy_convolution_small, ...
    'Older arbitrary-pulse kernel with acquisition-window convolution';
    'arb_relax_diff_small', @benchmark_arb_relax_diff_small, ...
    'Active diffusion-aware arbitrary-pulse kernel';
    'tiny_imaging_serial_phase_loop', @benchmark_tiny_imaging_serial_phase_loop, ...
    'Small serial phase-encoding loop that mimics imaging workload structure';
    'time_domain_echo_small', @benchmark_time_domain_echo_small, ...
    'Time-domain echo construction from acquired magnetization';
    'oct_objective_like_small', @benchmark_oct_objective_like_small, ...
    'Single OCT-like objective calculation using rotation-axis and filtering'
    };

num_bench = size(benchmarks,1);
names = cell(num_bench,1);
descriptions = cell(num_bench,1);
seconds = zeros(num_bench,1);
checksums = zeros(num_bench,1);

fprintf('Running %d spin-dynamics benchmarks...\n',num_bench);
for i = 1:num_bench
    names{i} = benchmarks{i,1};
    func = benchmarks{i,2};
    descriptions{i} = benchmarks{i,3};
    
    checksums(i) = func(); % Warm up persistent setup and record a simple output check.
    seconds(i) = measure_benchmark(func,p.Results.UseTimeit,p.Results.Repetitions);
    
    fprintf('%-36s %10.6f s   checksum %.6g\n',names{i},seconds(i),checksums(i));
end

results = table(names,seconds,checksums,descriptions, ...
    'VariableNames',{'Benchmark','Seconds','Checksum','Description'});

if p.Results.SaveResults
    output_file = char(p.Results.OutputFile);
    if ~isfolder(fileparts(output_file)) && ~isempty(fileparts(output_file))
        error('Output folder does not exist: %s',fileparts(output_file));
    end
    if isempty(fileparts(output_file))
        output_file = fullfile(this_dir,output_file);
    end
    save(output_file,'results');
    fprintf('Saved benchmark results to %s\n',output_file);
end

end

function seconds = measure_benchmark(func,use_timeit,repetitions)

if use_timeit && exist('timeit','file') == 2
    seconds = timeit(@() run_no_output(func));
else
    timings = zeros(repetitions,1);
    for k = 1:repetitions
        tic;
        run_no_output(func);
        timings(k) = toc;
    end
    seconds = median(timings);
end

end

function run_no_output(func)

func();

end

function checksum = benchmark_arb10_kernel_small

persistent params
if isempty(params)
    params = make_precomputed_kernel_params(1001,12);
end

macq = sim_spin_dynamics_arb10(params);
checksum = real(sum(abs(macq(:))));

end

function checksum = benchmark_arb8_legacy_convolution_small

persistent params
if isempty(params)
    params = make_precomputed_kernel_params(1001,12);
    params.len_acq = 5*pi;
end

macq = sim_spin_dynamics_arb8(params);
checksum = real(sum(abs(macq(:))));

end

function checksum = benchmark_arb_relax_diff_small

persistent params
if isempty(params)
    params = make_diffusion_kernel_params(1001,8);
end

macq = sim_spin_dynamics_arb_relax_diff(params);
checksum = real(sum(abs(macq(:))));

end

function checksum = benchmark_tiny_imaging_serial_phase_loop

persistent base_params gradx gradz del_wx del_wz
if isempty(base_params)
    base_params = make_precomputed_kernel_params(401,4);
    base_params.grad(2:3:end) = 1;
    grid_axis = linspace(-1,1,401);
    del_wx = grid_axis;
    del_wz = cos(pi*grid_axis/2);
    gradx = linspace(-2,2,4);
    gradz = linspace(-2,2,4);
end

num_x = length(gradx);
num_z = length(gradz);
num_echo = sum(base_params.acq);
echo_int = zeros(num_x,num_z,num_echo);

for i = 1:num_x
    for j = 1:num_z
        params = base_params;
        params.del_wg = gradx(i)*del_wx + gradz(j)*del_wz;
        mrx1 = sim_spin_dynamics_arb10(params);
        params.del_wg = -params.del_wg;
        mrx2 = sim_spin_dynamics_arb10(params);
        echo_int(i,j,:) = sum(mrx1 - mrx2,2);
    end
end

checksum = real(sum(abs(echo_int(:))));

end

function checksum = benchmark_time_domain_echo_small

persistent mrx del_w tacq tdw
if isempty(mrx)
    numpts = 1001;
    del_w = linspace(-12,12,numpts);
    envelope = exp(-(del_w/5).^2);
    mrx = envelope.*exp(1i*0.2*del_w);
    tacq = 6*pi;
    tdw = tacq/256;
end

echo = calc_time_domain_echo_arb(mrx,del_w,tacq,tdw,0);
checksum = real(sum(abs(echo(:))));

end

function checksum = benchmark_oct_objective_like_small

persistent del_w tp phi amp window
if isempty(del_w)
    numpts = 1001;
    del_w = linspace(-10,10,numpts);
    tfp = 3*pi;
    tp = [tfp 0.35*pi 0.45*pi 0.30*pi tfp];
    phi = [0 0 pi/2 pi 0];
    amp = [0 1 1 1 0];
    tacq = 5*pi;
    window = sinc(del_w*tacq/(2*pi));
    window = window./sum(window);
end

[neff,alpha] = calc_rot_axis_arba4(tp,phi,amp,del_w,0);
masy = conv(neff(1,:) + 1i*neff(2,:),window,'same');
vcrit_proxy = 1./(1 + abs(sin(alpha)));
checksum = real(trapz(del_w,abs(masy).^2) + trapz(del_w,vcrit_proxy));

end

function params = make_precomputed_kernel_params(numpts,num_echo)

del_w = linspace(-10,10,numpts);
sp.del_w = del_w;
sp.w_1 = ones(1,numpts);

Rtot = cell(1,2);
pp_pulse.tp = pi/2;
pp_pulse.phi = pi/2;
pp_pulse.amp = 1;
Rtot{1} = calc_rotation_matrix(sp,pp_pulse);

pp_pulse.tp = pi;
pp_pulse.phi = 0;
pp_pulse.amp = 1;
Rtot{2} = calc_rotation_matrix(sp,pp_pulse);

tfp = 3*pi;
tp = [pi/2 repmat([tfp pi tfp],1,num_echo)];
pul = [1 repmat([0 2 0],1,num_echo)];
amp = [1 repmat([0 1 0],1,num_echo)];
acq = [0 repmat([0 0 1],1,num_echo)];
grad = zeros(size(tp));

params.tp = tp;
params.pul = pul;
params.Rtot = Rtot;
params.amp = amp;
params.acq = acq;
params.grad = grad;
params.del_w = del_w;
params.del_wg = zeros(1,numpts);
params.w_1 = ones(1,numpts);
params.T1n = 250*ones(1,numpts);
params.T2n = 80*ones(1,numpts);
params.m0 = ones(1,numpts);
params.mth = ones(1,numpts);

end

function params = make_diffusion_kernel_params(numpts,num_echo)

del_w = linspace(-10,10,numpts);
tfp = 3*pi;

params.tp = [pi/2 repmat([tfp pi tfp],1,num_echo)];
params.phi = [pi/2 repmat([0 0 0],1,num_echo)];
params.amp = [1 repmat([0 1 0],1,num_echo)];
params.acq = [0 repmat([0 0 1],1,num_echo)];
params.len_acq = 5*pi;
params.del_w = del_w;
params.w_1 = ones(1,numpts);
params.T1n = 250*ones(1,numpts);
params.T2n = 80*ones(1,numpts);
params.m0 = ones(1,numpts);
params.mth = ones(1,numpts);
params.gamma = 1;
params.grad = 0.02;
params.D = 1;
params.Delta = 0.5;
params.pul = params.tp;

end
