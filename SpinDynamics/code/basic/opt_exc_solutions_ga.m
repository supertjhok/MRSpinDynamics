function opt_exc_solutions_ga

T_90=20;
T_min=0.1*T_90;

texc=[20    20    26    24    28    22    28    24    22    16]/20;
pexc=[3     0     1     3     3     0     3     0     2     1]*(pi/2);
echo_pk=0.1482;
echo_rms=6.8187e-9;

% GA-M1
texc=[36     6    26    42    16     2    38    12    12    42]/20;
pexc=[2     1     2     0     1     2     3     1     3     3]*(pi/2);
echo_pk=0.1730;
echo_rms=9.2205e-09;

% GA-M2, 2 peaks
texc=[60    56    14     0     8    52    18    42     6    10]/20;
pexc=[3    1    3    2    1    1    3    1    3    1]*(pi/2);
echo_pk=0.1892;
echo_rms=9.9150e-009;

T_min=0.05*T_90;

texc=[13    26    21    16    18    17    20    21    15    28]/20;
pexc=[3     2     1     2     1     3     0     0     1     3]*(pi/2);
echo_pk=0.1487;
echo_rms=8.2222e-9;
 
