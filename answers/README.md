# Machine Learning for Econometrics: elements of answer

## 4. Post-selection Inference

1. see section 4.2.
    
2. see section 4.3.
    
3. Ideally, we would like to be able to compute confidence intervals and perform tests on this parameter of interest. However, the fact that the asymptotic distribution of the Lasso estimator is not known makes constructing these quantities more difficult.
        
4. Regularization bias is a bias that arises because the use of machine learning tools in the first step produces estimators that do not converge at a fast enough rate. In the case of Lasso, it is a form of omitted variable bias. It can exist even in a low-dimensional setting if a non-conventional estimator is used in the first step or if there is a selection step involved. (Think of the Leeb and Pötscher model).
    
5. see section  4.5.
    
6. Leeb and Pötscher’s argument is that inference after a selection step is more complicated than it seems, because post-selection estimators do not have 'good properties' (such as asymptotic normality). Theorem 5.1 is not contradictory: it provides a solution to the problem. It shows that in cases where the estimator is immunized, inference based on the normal distribution is still possible. In many cases, this means adding another selection step (hence the name 'double selection').


## 5. Generalization and methodology

1. No, the double selection procedure is the key. However, note that the Lasso is *“good enough”* to be used in the initial steps.

2. See the remark in Chapter 5. Sample splitting is used.

3. Lasso is used when one is willing to assume a linear structure with sparse coefficients (only a small number of non-zeros). A random forest is preferred when one assumes that the regression function is piecewise constant.

4. This eliminates overfitting bias, but the cost is a reduction in sample size and (secondarily) computation time when using cross-fitting.

5. In Section 3.4/4.b (simulation study from the high-dimension and endogeneity chapter).

6. It is also a variable selection procedure that uses only a single equation, and is therefore subject to the problem of post-selection inference.


## 6. High dimension and endogeneity

1. Either the list of available and possible instruments is long, whereas the researcher knows that only a few of them are relevant; but above all, even when only one instrument $Z$ is available, transformations of the initial instrument can also be envisaged.

    $$\left(f_1(Z), \dots, f_p(Z) \right).$$

    by a family of functions $(f_i)_{i=1}^p$. This brings us back to the case of very many possible instruments. In the course, we use a parsimonious model for the instrumental variables, assuming that only a few instruments are actually useful, and provide a Lasso-based method for estimating the treatment while controlling the estimate of the nuisance parameter.

2. Without the $\mathbb{E}[D |X ]$ term in (6.9), we can compute, for example, that $\partial_{\nu} M(\tau_0,\eta_0) = \tau_0 \mathbb{E}[ X(Z'\delta_0 + X'\gamma_0)]$, which is different from 0 in general.

3. We use the $\sigma_{\Gamma}^2$ formula given in (6.12), and obtain that

     $$\sigma_{\Gamma}^2 = \frac{E[\varepsilon^2 E[D|Z]^2 ]}{E[E[D|Z]^2  ]^2} = \sigma^2 E[E[D|Z]^2]^{-1}.$$

    using the conditional homoscedasticity assumption. The semi-parametric efficiency bound (3.14) is thus reached (in this case where $S = D$, $W=Z$ with the notation of section 3.4).

4. We check that the assumption (ORT) $\partial_{\eta} M(\tau_0,\eta_0)=0$ is satisfied by noting that with (6.11) and (6.2), we obtain $\partial_{\nu} M(\tau_0,\eta_0) = \tau_0 E[ X (\zeta'\delta_0)] =0$, $\partial_{\theta} M(\tau_0,\eta_0) = -E[ X (\zeta'\delta_0)]=0$. The other two conditions with respect to $\gamma$ and $\delta$ are also verified using (6.3) and (6.4).

5. Using the notation of (6.14), we obtain $u_{i,j} = \delta_j + \varepsilon_{i,j}$, thus
   
    $$P_{i,j}  =  P\left(\delta_j - \delta_{j'}  + \varepsilon_{i,j} >  \varepsilon_{i,j'}, \ \forall j' \neq j\right).$$

Let $\varepsilon_{i,j}$, this probability is the probability of $\delta_j - \delta_{j'}  + \varepsilon_{i,j} >  \varepsilon_{i,j'}$, which is 

$$F(\delta_j - \delta_{j'}  + \varepsilon_{i,j}) = \exp(-\exp(- (\delta_j - \delta_{j'}  + \varepsilon_{i,j})).$$

We thus obtain

 $$P_{i,j} = \int \left( \underset{j'\neq j}{\prod} e^{-e^{-(\varepsilon + \delta_j - \delta_{j'})}}\right) e^{-\varepsilon}e^{-e^{-\varepsilon}} d\varepsilon.$$

 After a little algebra and a change of variable, we obtain (6.17).


## 7. Going further

1. To account for non-Gaussian and heteroskedastic errors, the $\ell_1$ penalty in standard Lasso estimation is modified by using specific penalties designed to allow the application of results from moderate deviation theory. This leads to a two-step estimation procedure, where these penalties are first initialized with the identity matrix, then updated using the estimated error terms from the first step. 

2. *Sample splitting* involves dividing the sample to decorrelate the estimation of nuisance parameters from that of the parameter of interest. An advantage discussed in Section 7.2 is that, in theory, it allows for more non-zero components (condition (7.7)). A drawback of partitioning is a certain loss of efficiency.


## 8. Inference on heterogeneous effects

1. The fundamental problem with causal inference is that we never observe the *ground truth*, i.e. the true effect of the treatment. Standard procedures such as cross-validation cannot therefore be reused without modification, at least not with the same efficiency.

2. The author contrasts two strategies for attributing a treatment (e.g. an e-mailing or telephone campaign): either targeting those who are most likely not to buy again, or targeting those who respond most when they receive the treatment. These two populations are not necessarily made up of the same individuals. Targeting those who are most likely not to buy again makes intuitive sense, but it's also inefficient: the treatment should be allocated in such a way as to maximize its impact, measured as an increase in the probability of purchase. Generic machine learning could be used to classify people into groups.

3.Random splits in the construction of random forests require the entire support of the explanatory variables to be explored, and avoid the concentration of cuts in a limited part of space, which would be achieved with optimally selected splits. This is key to achieving estimator consistency.

4. CATE can be estimated, but not convergently. Thus, only certain CATE features such as GATES or CLAN can be estimated.

5. In this case, we only have confidence intervals \textbf{conditional}

$$\mathbb{P}\left(\theta_A \in [L_A, U_A] \middle| \text{Data}_A \right) = 1 - \alpha + o_P(1),$$

where $[L_A, U_A]: = \left[\widehat{\theta}_A \pm \Phi^{-1}(1-\alpha/2) \widehat{\sigma}_A \right]$. This does not take into account the variability introduced by sample splitting, which prevents any generalization to any distribution of the whole data set.

6. If the same sample were used to estimate splits and values on leaves, the algorithm would tend to separate two leaves that have heterogeneous (relatively high and low) treatment effects in this sample, leading to a biased estimate if we use the sample to evaluate it. If we use another sample to evaluate it, this limits overlearning and ensures convergence.

7. The aim of causal random forests is to estimate a treatment effect consistently, whereas random forests estimate a regression function and aim to minimize the prediction error (often in $\ell_2$ norm, or MSE). This has consequences for the form of the estimator, as causal random forests require the honesty property to be consistent.

8. The best linear predictor of CATE is the linear projection of an unbiased CATE signal onto the linear vector space generated by $T$. In this sense, BLP depends on the performance of $T$. If it fits CATE well, then BLP's slope coefficient will be close to one, and we'll learn about CATE characteristics by looking at $T$.


## 9. Optimal policy learning

1. Using Assumption 9.2 (i), the constraint is relevant and therefore always saturated for the optimal policy, i.e. $c = \int_{x \in \mathcal{X}}\pi(x) dF_X(x)$. Let $pi'$ be an optimal policy different from $pi$ given in Proposition 9.1 on a set of non-zero $F_X$ measures. This policy also satisfies the constraint, and with Assumption 9.2 (ii), there exist sets $\Omega'$ and $\Omega$, such that
    $$\int_{\Omega'} \pi'(x) dF_{X}(x) = \int_{\Omega} \pi(x) dF_{X}(x),$$
     $\Omega' \subseteq \{ x: \ \tau(x) < \gamma \}$, et $\Omega \subseteq \{ x: \ \tau(x) \geq \gamma \}$. We thus have
$$\int_{\Omega'} \pi'(x) \tau(x) dF_{X}(x) <  \gamma  \int_{\Omega'} \pi'(x)  dF_{X}(x)= \gamma  \int_{\Omega} \pi(x)  dF_{X}(x)$$
and this is smaller than $\int_{\Omega} \pi(x) \tau(x) dF_{X}(x)$, hence the contradiction.

2. The objective used is worst-case regret control (*minimax regret criterion*), see (9.6). The advantage is the robustness of the recommended policies to different distributions of possible effects. The disadvantage is that we don't use potential *apriori* on these effects, which would allow us to obtain better results if they are verified. 

3. It's desirable to limit the policy classes considered so that they can be implemented simply in the field, but also to obtain an upper bound on regret (Theorem 9.1).

4. The complexity of the policy classes considered is limited by the Vapnik-Chervonenkis (VC) dimension. We can allow classes whose complexity increases with sample size, but less rapidly than $n$.

5. see Remark 9.2.

6. This formulation makes policy learning by empirical maximization appear as a weighted optimization problem in the context of classification. We can use tools developed in weighted classification (see e.g., Athey and Wager, 2021; Zhou et al., 2018 for more details) to solve this problem.


## 10. The synthetic control method

1. No, that is not the case. Indeed, there is only one treated unit, so no law of large numbers-type result applies.

2. There should be three possible answers among the following: (1) no extrapolation (the weights are non-negative and sum to one), (2) transparency of fit (the pre-treatment fit can be assessed), (3) guards against specification search (the weights can be computed independently of the post-treatment outcome), (4) sparsity/qualitative interpretation (at most $p+1$ weights are strictly positive). See the corresponding chapter.

3. The synthetic control weight vector is generally sparse, meaning that only a few entries are non-zero. As a result, only a few control units contribute to the counterfactual. On one hand, this means the full sample is not used (loss of efficiency?), but on the other hand, it excludes units that do not help in reproducing the treated unit.

4. see section 10.4.
     
5. It is impossible because the test of no treatment effect is rejected with a p-value of 0.02. Hence, 0 cannot be inside the 90\% confidence interval. 

6. Let $D^{obs} =(D_1,\ldots, D_n)$ the observed vector of treatment assignment and $\hat \tau^{obs}$ the corresponding OLS estimator. Fisher's procedure is :
     - For $b=1, \ldots, B$, reshuffle treatment assignment at random, compute the OLS estimator of $\tau_0$, $\hat \tau_b$ and compare it to $\hat \tau^{obs}$. 
     - Compute Fisher's p-value : 

        $$\hat p := \frac{1}{B} \sum_{b=1}^{B} \mathbf{1} \left\{ \vert \hat \tau_b \vert  \geq \vert \hat \tau^{obs} \vert \right\}.$$

     - Reject $H_0$ if $\hat p$ is below a pre-determined threshold: the observed treatment allocation yields an effect that is abnormaly high compared to a random distribution.


## 11. Forecasting in high-dimension

1. We can no longer assume independence of observations, as they are correlated over time. Economic and financial time series are also known to exhibit heavy-tailed distributions. Finally, we must also account for the fact that the series are not sampled at the same frequency.

2. Some explanatory variables often have a particular structure, such that only a few groups of variables (macroeconomics, different industry sectors, financial variables, news) may be useful for prediction—but within these groups, several variables are relevant.

3. The Lasso penalty enforces sparsity and introduces a bias that must be taken into account when making inference on a group of coefficients (see Section 7.3).

4. The risk of using a method that imposes sparsity is the risk of misspecification, meaning that the sparsity assumption may not hold, leading to bias. Some empirical examples mentioned in Section 11.2.1 show that this assumption must be justified carefully. The FARM approach allows combining a sparse component with a dense one, thus enabling testing whether the dense component is useful.


## 12. Working with textual data

1. The vocabulary of a text can be very large, which requires representing each word as a high-dimensional vector using a one-hot representation (see Chapter 13). In more advanced models, high-dimensional vectors allow for capturing more details and nuances in the text.

2. Here are two examples among infinitely many:
(i) Handling typos: The use of character n-grams helps address this issue by representing text at a lower level—at the level of character sequences.
(ii) Handling plural forms.

3. Models using latent variables help capture the general context of a document. However, these models are, on the one hand, complex to estimate, and on the other hand, have been outperformed in terms of performance by modern pretrained language models.

4. The first challenge is to match each forum post with the financial asset(s) mentioned. This phase largely depends on data quality. If the posts systematically mention stock tickers (e.g., AAPL for Apple Inc., GME for GameStop, etc.), it's enough to obtain a list of these tickers and search for their occurrences in the messages. Using the message dates, each post can then be matched to the prices of the mentioned assets. If the messages do not mention stock tickers, one must either adopt a more sophisticated approach (e.g., based on proximity to company names), or a more data-driven approach—such as a "bag-of-words" classification with a large number of classes (as many as there are assets). Once the data matching is done, a wide range of strategies become possible (e.g., sentiment analysis of the messages).


## 13. Word embeddings

1. The cosine similarity for two distinct words is zero.

2. The answer to the previous question indicates that for two distinct words the cosine similarity is zero. It is therefore not possible to obtain a notion of 'degree of similarity' between two words with this approach.

3. Lexical embedding is a technique used to represent the words in a text as reduced-dimension vectors. The aim of lexical embedding is to compactly capture the semantic relationships between words in text, so that these vector representations can be used to perform natural language processing tasks. These vectors can then be used to represent a document by aggregating them (via an average, for example), and then using this representation in a logistic regression, for example.

4. see section 13.2.

5. see section 13.2.

6. see section 13.3.


## 14. Modern language models

1. The idea is to have a measure of a product's quality as perceived by the consumer. For example, you could include in a model the average embedding of a product's review comments. Or the mebeddings of product photos.
    
2. see section 14.2.

3. see section 14.2.2.

4. By using n-grams of words. The limitations are that long dependencies can be complicated to take into account, the number of tokens increases exponentially, and the trained algorithm will not be able to adapt to a never-before-seen word sequence.
