# About this project
The Gamma project aims to construct mathematics from a computational perspective; this project aims to construct basic to fairly sophisticated statistical methods implemented in Python. Both numerical and symbolic approach of programming shall be contained in this project. This is intended for exploring the art of statistics from a computational and analytical perspective. Computational methods are implemented with the help of numerical libraries such as Numpy and SciPy. Analytical methods are represented in abstract symbols (they are often referred as Symbolic computation) which concerns itself to representing a mathematical construct and carrying out certain kinds of properties without necessarily computing an approximate value, for this case, we will explore Python's CAS library called SymPy. Whereas numerical solutions are best at approximating solutions, analytic solutions (or symbolic computation) is concerned with exact computation. 


### Advantages and Disadvantages
----
The SciPy library, including NumPy, is implemented in C/++ and Fortran codebase. They are super-fast and highly optimized for efficiency in numerical computation. Whereas SymPy is purely implemented in Python. The intentions from both of these packages are different. Whereas numerical libraries are primarily optimized for high-performance computation,  CAS libraries put those concerns not so much as implementing mathematical methods for symbolic computation. 

The advantage of leveraging numerical libraries for our purposes is that it executes computation very fast, we need high-performance computation in finding statistical values for our summaries from a large dataset. However, we might not get the analytical counterpart that is the symbolic conception of statistical ideas. To reach that gap of exploration, we need another programming paradigm that is concerned to symbolic representations. 

----
Content
- Probability Distributions
    - Discrete Distribution (PMF, CDF, Generator)
        - Uniform distribution
        - Binomial distribution
        - Multinomial distribution
        - Geometric distribution
        - Hypergeometric distribution
        - Poisson distribution
    - Continuous Distribution (PDF, CDF, Generator)
        - Normal-distribution
        - t-distribution
        - Cauchy distribution
        - F-distribution
        - Chi distribution
        - Gamma distribution
        - Pareto distribution
        - Log-Normal distribution
        - Non-central chi distribution

- Descriptive Statistics
    - ungrouped 
    - grouped 
    - summary tables 
        - frequency distribution
        - contingency table
    - graphics (from Matplotlib)
        - scatterplot
        - bar chart
        - histogram
        - pie chart

- Inferential Statistics
    - Z-test
    - T-test
    - F-test
    - Goodness of fit
        - Chi-squared
        - G-test
        - Kolmogorov-Smirnov
        - Anderson-Darlin
        - Lilliefors
        - Shapiro-Wilk
        - Likelihood-ratio test

- Regression
    - Simple linear regression
    - Multiple linear regression
    

- Analyses of Variance
    - post hoc tests 
        - Tukey test
        - Fisher test
        - HSU's MCB
        
- Exploratory Data Analysis
    - box plot
    - histogram
    - scatterplot
    