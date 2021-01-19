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
    - vlines
    - annotation 
- likelihood function
- log-likelihood function
- logcdf
- logpdf
- maximum-likelihood function
- random variable generator 
- standard deviation
- moment generating functions 
- entropy
- fisher information
- point-percentage function 

----
to clean:
- comments
- replace " ** " to pow(x,y)
- replace `np.log(x)` to `math.log(x,y)`
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
    - continuous Bernoulli distribution
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
    - Pareto distribution
    - Nakagami distribution
    <!-- - Rice -->
    - Lomax distribution
    - Gumbel distribution
    - Weibull distribution

Real line
- Staged for review:
    - Gumbel  distribution
    - Fisher's z-distribution
    - Asymmetric Laplace distribution
    - Generalized normal v1
    - Generalized hyperbolic - resolve cdf and pdf
    - Hyperbolic secant

- change category:
    - Cauchy
    - Laplace
    - Logistic
    - Normal
    - T
    - Gumbel Type 1

----
in progress
- Continuous univariate 
    - bounded interval
        - ARGUS
        - non-central beta

    - semi-infinite interval
        - truncated normal type 2
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
        - exponential power
        - Gaussian q
        - generalized normal
        - generalized hyperbolic
        - geometric stable
        - Holtsmark - hypergerometric function
        - Johnson's SU
        - Landau - integral form
        - noncentral t - needs further reading
        - normal-inverse Gaussian
        - skew normal - needs further reading
        - slash
        - stable - find numerical counterparts, no analytical expression is defined
        - Tracy–Widom - needs further reading
        - variance-gamma
        - Voigt -  find numerical counterparts, as analytical expression is deemed [complicated](https://en.wikipedia.org/wiki/Voigt_profile)
        
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
