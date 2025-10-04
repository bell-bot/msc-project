# Experiment: Randomised vs. Standard Backdoor with GPT2 Target Model

## Baseline Results

KL Weights 6.152332
KL Biases inf
EMD Weights 0.096350
EMD Biases 4.127439
KS Weights Statistic 0.504248
KS Weights P-value 0.000000
KS Biases Statistic 0.999311
KS Biases P-value 0.000000
Mean Weights 0.001980
Mean Biases -4.194223
Std Weights 0.046856
Std Biases 3.437875
Kurtosis Weights 450.164896
Kurtosis Biases -1.028155

## Randomised Backdoor Results

KL Weights 6.852437e+00
KL Biases inf
EMD Weights 9.649769e-02
EMD Biases 3.124860e-02
KS Weights Statistic 5.041671e-01
KS Weights P-value 0.000000e+00
KS Biases Statistic 2.218527e-01
KS Biases P-value 0.000000e+00
Mean Weights 2.920798e-05
Mean Biases -8.209099e-02
Std Weights 2.515783e-03
Std Biases 1.030675e-01
Kurtosis Weights 1.155921e+04
Kurtosis Biases 1.088753e+02

## GPT2 Statistics

Weights: - Mean: -0.0014 - Std: 0.1278 - Kurtosis: 40.5677

Biases: - Mean: -0.0668 - Std: 0.1096 - Kurtosis: 18.3524

# Experiment: Randomised vs. Standard Backdoor with Laplace Target Dist
- Laplace distribution was fitted to GPT2 mlp model weights and biases
- Result: Laplace distribution with mean 5.9604645e-06 and standard deviation 0.04600873749930559
- Sampled 2000000 weights and biases (each) 

## Target Statistics
Weights stats: {'mean': 1.3953048437542748e-05, 'std': 0.06503201276063919, 'kurtosis': 5.981627743605196}
Bias stats: {'mean': -9.838238474912941e-05, 'std': 0.06503456085920334, 'kurtosis': 5.9970936241812645}

## Baseline Results

KL Weights                     inf
KL Biases                   inf
EMD Weights              0.088200
EMD Biases               3.824320
KS Weights Statistic     0.496000
KS Weights P-value       0.000000
KS Biases Statistic      0.969295
KS Biases P-value        0.000000
Mean Weights             0.053100
Mean Biases             -3.804490
Std Weights              0.241000
Std Biases               3.370500
Kurtosis Weights        12.009700
Kurtosis Biases         -0.813795

# Randomised Backdoor Results
KL Weights                 4.714830
KL Biases: [' inf 0.7885 inf 0.7828 inf inf 0.7806 inf 0.7825 inf inf 0.7932 inf inf inf inf inf inf inf inf']
EMD Weights               0.045300
EMD Biases                0.037515
KS Weights Statistic      0.492975
KS Weights P-value        0.000000
KS Biases Statistic       0.465580
KS Biases P-value         0.000000
Mean Weights              0.000500
Mean Biases              -0.037460
Std Weights               0.007675
Std Biases                0.053560
Kurtosis Weights        424.189640
Kurtosis Biases           6.822875