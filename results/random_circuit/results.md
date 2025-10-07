# Experiment: Randomised vs. Standard Backdoor with GPT2 Target Model

## Large Model
### Model Setup (standard and backdoor)

num_samples: 50
c: 448
n: 24
log_w: 6
random_seed: 95
trigger_length: 16
payload_length: 16

### Baseline Results

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

### Randomised Backdoor Results

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

## Small Model

### Model Setup
num_samples: 20
c: 20
n: 3
log_w: 1
random_seed: 95
trigger_length: 16
payload_length: 16

### Baseline Results
KL Weights                4.613600
EMD Weights              0.123450
EMD Biases               3.755805
KS Weights Statistic     0.500265
KS Weights P-value       0.000000
KS Biases Statistic      0.969165
KS Biases P-value        0.000000
Mean Weights             0.053100
Mean Biases             -3.804495
Std Weights              0.241000
Std Biases               3.370500
Kurtosis Weights        12.009700
Kurtosis Biases         -0.813800

### Randomised Backdoor Results
KL Weights                 4.087855
EMD Weights               0.095180
EMD Biases                0.030670
KS Weights Statistic      0.498000
KS Weights P-value        0.000000
KS Biases Statistic       0.208385
KS Biases P-value         0.000000
Mean Weights              0.001015
Mean Biases              -0.078695
Std Weights               0.014970
Std Biases                0.103870
Kurtosis Weights        520.282245
Kurtosis Biases           9.083655

# Experiment: Randomised vs. Standard Backdoor with Laplace Target Dist

-   Laplace distribution was fitted to GPT2 mlp model weights and biases
-   Result: Laplace distribution with mean 5.9604645e-06 and standard deviation 0.04600873749930559

    -   Fitted Laplace parameters for weights: (-0.001100021, 0.09658630688985188)
    -   Fitted Laplace parameters for biases: (-0.06297976, 0.0810421519809299)

-   Sampled 2000000 weights and biases (each)

## Model Setup (standard and backdoor)

num_samples: 20
c: 20
n: 3
log_w: 1
random_seed: 95
trigger_length: 16
payload_length: 16
sample_size: 2000000

## Target Statistics

Weights stats: {'mean': -0.001109645119868219, 'std': 0.13665452599525452, 'kurtosis': 5.961965218604368}
Bias stats: {'mean': -0.06294649839401245, 'std': 0.11450858414173126, 'kurtosis': 6.014202845029759}

## Baseline Results

KL Weights inf
KL Biases inf
EMD Weights 0.118300
EMD Biases 3.758890
KS Weights Statistic 0.501600
KS Weights P-value 0.000000
KS Biases Statistic 0.969295
KS Biases P-value 0.000000
Mean Weights 0.053100
Mean Biases -3.804490
Std Weights 0.241000
Std Biases 3.370500
Kurtosis Weights 12.009700
Kurtosis Biases -0.813795

# Randomised Backdoor Results

KL Weights 4.709005
KL Weights [inf, ..., 0.6983]
EMD Weights 0.095275
EMD Biases 0.036235
KS Weights Statistic 0.498105
KS Weights P-value 0.000000
KS Biases Statistic 0.202855
KS Biases P-value 0.000000
Mean Weights 0.001055
Mean Biases -0.078895
Std Weights 0.016130
Std Biases 0.112885
Kurtosis Weights 422.149245
Kurtosis Biases 6.797810

# Experiment: Randomised vs. Standard Model and Normal Target Distribution

-   Fitted Normal parameters for weights: (-0.0014237247, 0.12775175)
-   Fitted Normal parameters for biases: (-0.066784285, 0.10962485)

## Model Setup (standard and backdoor)

num_samples: 20
c: 20
n: 3
log_w: 1
random_seed: 95
trigger_length: 16
payload_length: 16
sample_size: 2000000

## Target Statistics

-   Weights stats: {'mean': -0.0012810614425688982, 'std': 0.1277788281440735, 'kurtosis': 2.995532442885635}
-   Bias stats: {'mean': -0.06675281375646591, 'std': 0.10960625857114792, 'kurtosis': 2.997499383147008}

## Baseline Results

KL Weights inf
KL Biases inf
EMD Weights 0.130800
EMD Biases 3.757605
KS Weights Statistic 0.500200
KS Weights P-value 0.000000
KS Biases Statistic 0.969295
KS Biases P-value 0.000000
Mean Weights 0.053100
Mean Biases -3.804490
Std Weights 0.241000
Std Biases 3.370500
Kurtosis Weights 12.009700
Kurtosis Biases -0.813795

# Randomised Backdoor Results

KL Weights 5.916215
KL Biases inf
EMD Weights 0.100500
EMD Biases 0.032110
KS Weights Statistic 0.497960
KS Weights P-value 0.000000
KS Biases Statistic 0.246110
KS Biases P-value 0.000000
Mean Weights 0.001100
Mean Biases -0.083855
Std Weights 0.015060
Std Biases 0.103605
Kurtosis Weights 210.755865
Kurtosis Biases 2.498270
