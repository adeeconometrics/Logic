This folder contains implementation for continuous and discrete probability distributions which generally supports the following features:

currently supported features:
- probability distribution function and probability mass function
- cumulative distribution function 
- central moments 
    - mean 
    - median
    - mode 
- variance, skewness, kurtosis
- p value
----
features to be developed:
- plotting probability distribution
    - plain
    - fill-in between gradient 
- random variable generator 
- moment generating functions 
- entropy
- fisher information
- point-percentage function 

# List of supported distributions 
---
## Discrete 
### Univariate 
- uniform distribution
- binomial distribution
- bernoulli distribution
- hypergeometric distribution
- geometric distribution
- poisson distribution
- zeta 
--- 
in progress 
- negative binomial 
- beta binomial
### Multivariate
- multinomial distribution
----
----
## Continuous
### Univariate 
- uniform continuous
- gaussian distribution
- t-distribution
- cauchy distribution
- f distribution
- chi-square
- chi distribution
- exponential distribution
- pareto distribution
- log-normal distribution
- laplace distribution
- logistic distribution
- logit-normal distribution
- weilbull distribution
- weilbull inverse distribution
- gumbell distribution
- arcsine distribution
- staged for review
    - triangular distribution
    - trapezoidal distribution
    - beta distribution
    - beta-prime distribution
    - Erlang distribution
    - Rayleigh distribution
    - Maxwell-Boltzmann distribution
    - Wigner semicircle distribution
    - beta rectangular distribution
    - Bates distribution
    - continuous bernoulli distribution
    - Balding-Nichols distribution

Semi-infinite class
- staged for review
    - Benini distribution
    - Folded Normal distribution
    - Half Logistic distribution
    - Half Normal distribution
    - Inverse Gaussian distribution
    - Inverse Gamma distribution
    - Dagum distribution
    - Davis distribution
    - Rayleigh distribution
    - Benktander Type 1 distribution
    - Benktander Type 2 distribution
    - hypoexponential distribution
    - log-Cauchy distribution
    - log-Laplace distribution
    - log-Logistic distribution
    - Inverse chi-squared distribution
    - Lévy distribution
    - pareto

----
in progress
- Continuous univariate 
    - bounded interval
        - ARGUS
        <!-- - Balding-Nichols -->
        <!-- - Bates
        <!-- - beta -->
        <!-- - beta rectangular
        - continuous bernoulli -->
        <!-- - logit normal -->
        - non-central beta
        <!-- - uniform -->
        <!-- - Wigner semicircle -->
    - semi-infinite interval
        - nakagami
        - Rice
        - lomax
        - truncated normal type 2
        - Gumbel 
        - Weibull
        - relativistic Breit–Wigner 
        - Exponential-logarithmic*
        - exponential F**
        - generalized gamma
        - Gompertz*
        - Hotelling's T-squared - needs further reading
        - hyper-Erlang**
        - inverse chi-squared scaled 
        - Kolmogorov - needs further reading
        - matrix-exponential - needs further reading
        - Maxwell–Jüttner
        - Mittag-Leffler - needs further reading
        - noncentral chi-squared
        - noncentral F
        - phase-type - needs further reading
        - poly-Weibull - needs further reading
        - Wilks's lambda

    - supported on the whole real line 
        <!-- - Cauchy -->
        - exponential power
        - Fisher's z
        - Gaussian q
        - generalized normal
        - generalized hyperbolic
        - geometric stable
        - Gumbel
        - Holtsmark
        - hyperbolic secant
        - Johnson's SU
        - Landau
        <!-- - Laplace -->
        - asymmetric Laplace
        <!-- - logistic -->
        - noncentral t
        <!-- - normal (Gaussian) -->
        - normal-inverse Gaussian
        - skew normal
        - slash
        - stable
        <!-- - Student's t -->
        <!-- - type-1 Gumbel -->
        - Tracy–Widom
        - variance-gamma
        - Voigt
    - varying types supported
        - generalized chi-squared
        - generalized extreme value
        - generalized Pareto
        - Marchenko–Pastur
        - q-exponential
        - q-Gaussian
        - q-Weibull
        - shifted log-logistic
        - Tukey lambda

----
### Multivariate
