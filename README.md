# Assignment-4: Learning Probability Density Functions Using Data Samples (GAN-Based PDF Estimation)

---

## 1. Objective

The purpose of this assignment is to estimate an unknown probability density function (PDF) directly from observed data samples without assuming any predefined analytical or parametric distribution.

A Generative Adversarial Network (GAN) is used to implicitly learn the distribution of a transformed random variable derived from environmental air-quality measurements.

The Nitrogen Dioxide (NO₂) concentration feature from the India Air Quality dataset is selected for analysis.

---

## 2. Dataset Description

**Dataset:** India Air Quality Data  

**Source:** Kaggle  

Dataset Link:

https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

**Feature Used:** NO₂ concentration (x)

The dataset contains air-quality observations collected from different monitoring stations across India. The NO₂ concentration values are used as the primary feature.

---

## 3. Data Transformation

Each observation `x` is transformed into a new variable `z` using the nonlinear transformation:


z = Tr(x) = x + ar * sin(br * x)


The constants depend on the university roll number.

For Roll Number **102303276**:


ar = 0.5 × (102303276 mod 7)
= 0.5 × 5
= 2.5

br = 0.3 × ((102303276 mod 5) + 1)
= 0.3 × (1 + 1)
= 0.6


The transformation introduces non-linearity while maintaining the structure of the original dataset.

---

## 4. Problem Assumptions

- The transformed variable `z` is sampled from an unknown probability distribution.
- No analytical or parametric PDF (Gaussian, exponential etc.) is assumed.
- The distribution must be learned only from observed samples.

---

## 5. GAN-Based Probability Density Learning

### 5.1 GAN Overview

A Generative Adversarial Network consists of two neural networks trained simultaneously:

**Generator (G)**  
Learns to generate samples similar to the transformed data.

**Discriminator (D)**  
Learns to distinguish between real samples and generated samples.

Through adversarial training, the generator gradually learns the implicit distribution of `z`.

---

### 5.2 GAN Architecture

#### Generator Network

Input:


ε ~ N(0,1)


Architecture:

- Fully Connected Neural Network
- Two hidden layers
- 32 neurons per hidden layer
- ReLU activation
- Linear output layer producing generated samples `zf`

---

#### Discriminator Network

Input:


Real sample z OR Generated sample zf


Architecture:

- Fully Connected Neural Network
- Two hidden layers
- 32 neurons per hidden layer
- ReLU activation
- Sigmoid output layer producing probability score.

---

## 6. Training Procedure

Real samples:


z


Generated samples:


zf = G(ε)


where:


ε ~ N(0,1)


Training steps:

1. Discriminator learns to classify real and fake samples.
2. Generator learns to fool the discriminator.
3. Alternating optimization continues until stable convergence.

This allows the generator to implicitly learn the probability density of the transformed variable.

---

## 7. PDF Approximation from Generator Samples

After training the GAN:

1. A large number of samples are generated using the trained generator.
2. Probability density is estimated using:

- Histogram Density Estimation.

The learned PDF is compared with the empirical distribution obtained from real transformed data.

---

## 8. Results and Visualization

The notebook includes:

- Histogram of real transformed samples `z`.
- Histogram density obtained from generated samples.
- Visual comparison between real and generated distributions.

The generated samples show strong similarity with the empirical density.

---

## 9. Observations

### Mode Coverage

The GAN captures the dominant density regions of the transformed dataset. Generated samples adequately cover high-density areas while minor deviations appear in low-density tails.

---

### Training Stability

Generator and discriminator losses remain bounded and exhibit stable oscillatory behaviour, indicating balanced adversarial learning without mode collapse.

---

### Quality of Generated Distribution

The histogram-based density obtained from GAN samples closely follows the empirical transformed data distribution, showing successful implicit PDF learning.

---

## 10. Conclusion

This assignment demonstrates that probability density functions can be effectively learned directly from observed samples without assuming any analytical form.

The GAN-based framework successfully learns and approximates the unknown distribution of transformed NO₂ concentration data using adversarial training.
