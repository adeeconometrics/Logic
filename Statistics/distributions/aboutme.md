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
- triangular distribution
- trapezoidal distribution
- beta distribution
- beta-prime distribution
- Erlang distribution
- Rayleigh distribution
- Maxwell-Boltzmann distribution
----
in progress
- Continuous univariate 
    - bounded interval
        - ARGUS
        - Balding-Nichols
        - Bates
        <!-- - beta -->
        - beta rectangular
        - continuous bernoulli
        <!-- - logit normal -->
        - non-central beta
        <!-- - uniform -->
        - Wigner semicircle
    - semi-infinite interval
        - Benini
        - Benkatender 
        <!-- - Beta prime -->
        - Burr
        <!-- - chi-squared -->
        <!-- - chi -->
        - Dagum
        - Davis
        - Exponential-logarithmic
        <!-- - Erlang -->
        - exponential F
        - Folded normal
        <!-- - Frechet -->
        <!-- - gamma -->
        - generalized gamma
        - generalized inverse gamma
        - generalized inverse Gaussian
        - Gompertz
        - half-logistic
        - half-normal
        - Hotelling's T-squared
        - hyper-Erlang
        - hyperexponential
        - hypoexponential
        - inverse chi-squared scaled inverse chi-squared
        - inverse Gaussian
        - inverse gamma
        - Kolmogorov
        - Lévylog-Cauchylog-Laplacelog-logistic
        <!-- - log-normal -->
        - Lomaxmatrix-exponential
        <!-- - Maxwell–Boltzmann -->
        - Maxwell–Jüttner
        - Mittag-Leffler
        - Nakagaminoncentral chi-squared
        - noncentral F
        - Paretophase-type
        - poly-Weibull
        - Rayleighrelativistic
        - Breit–WignerRiceshifted 
        - Gompertz
        - truncated normal type-2 
        - GumbelWeibull discrete Weibull
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