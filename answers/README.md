# Machine Learning for Econometrics: elements of answer

## 4. Post-selection Inference

1. see section 4.2.
    
2. see section 4.3.
    
3. Ideally, we would like to be able to calculate confidence intervals and perform tests on this parameter of interest. However, the fact that the asymptotic distribution of the Lasso estimator is not known makes constructing these quantities more difficult.
        
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

1. Soit la liste des instruments disponibles et possibles est grande, alors que le chercheur sait que seuls quelques-uns d'entre eux sont pertinents ; mais surtout, même lorsqu'on ne dispose que d'un seul instrument $Z$, on peut aussi envisager des transformations de l'instrument initial 

    $$\left(f_1(Z), \dots, f_p(Z) \right).$$

    par une famille de fonctions $(f_i)_{i=1}^p$, ce qui nous ramène au cas de très nombreux instruments possibles. Dans le cours, nous utilisons un modèle parcimonieux pour les variables instrumentales, en supposant que seuls quelques instruments sont effectivement utiles, et nous fournissons une méthode basée sur Lasso pour estimer le traitement tout en contrôlant l'estimation du paramètre de nuisance.

2. Sans le terme $\mathbb{E}[D |X ]$ dans (6.10), on peut calculer par exemple que $\partial_{\nu} M(\tau_0,\eta_0) = \tau_0 \mathbb{E}[ X(Z'\delta_0 + X'\gamma_0)]$, qui est différent de 0 en général.

3. On utilise la formule de $\sigma_{\Gamma}^2$ donnée en (6.15), et on obtient que

     $$ \sigma_{\Gamma}^2 = & \mathbb{E}[\varepsilon^2 \mathbb{E}[D|Z]^2 ] / \mathbb{E}[\mathbb{E}[D|Z]^2  ]^2 = \sigma^2 \mathbb{E}[\mathbb{E}[D|Z]^2  ]^{-1}.$$

    en utilisant l'hypothèse d'homoscedasticité conditionnelle. On a donc que la borne (3.14) d’efficacité semi-paramétrique est atteinte (dans ce cas où $S = D$, $W=Z$ avec les notation de la section 3.4). 

4. On vérifie que l'hypothèse (ORT) $\partial_{\eta} M(\tau_0,\eta_0)=0$ est satifaite en remarquant qu'avec (6.11) et (6.2), on obtient $\partial_{\nu} M(\tau_0,\eta_0) = \tau_0 \mathbb{E}[ X (\zeta'\delta_0)] =0$, $\partial_{\theta} M(\tau_0,\eta_0) = -\mathbb{E}[ X (\zeta'\delta_0)]=0$. Les deux autres conditions par rapport à  $\gamma$ et $\delta$ sont aussi vérifiées en utilisant (6.3) et (6.4).

5. En utilisant les notations de (6.17), on a $u_{i,j} = \delta_j + \varepsilon_{i,j}$ et donc 
    \begin{align*}
        P_{i,j} = & P\left(\delta_j + \varepsilon_{i,j} > \delta_{j'} + \varepsilon_{i,j'}, \ \forall j' \neq j\right)\\
        = & P\left(\delta_j - \delta_{j'}  + \varepsilon_{i,j} >  \varepsilon_{i,j'}, \ \forall j' \neq j\right).
    \end{align*}
    Pour un $ \varepsilon_{i,j}$ fixé, cette probabilité est le produit des probabilités des $\delta_j - \delta_{j'}  + \varepsilon_{i,j} >  \varepsilon_{i,j'}$, qui est donnée par $F(\delta_j - \delta_{j'}  + \varepsilon_{i,j}) = \exp(-\exp(- (\delta_j - \delta_{j'}  + \varepsilon_{i,j}))$. On a donc 
    \begin{align*}
        P_{i,j} = \int \left( \underset{j'\neq j}{\prod} e^{-e^{-(\varepsilon + \delta_j - \delta_{j'})}}\right) e^{-\varepsilon}e^{-e^{-\varepsilon}} d\varepsilon.
    \end{align*}
    Après un peu d'algèbre et un changement de variable, on obtient (6.17).


## 7. Going further

1. To account for non-Gaussian and heteroskedastic errors, the $\ell_1$ penalty in standard Lasso estimation is modified by using specific penalties designed to allow the application of results from moderate deviation theory. This leads to a two-step estimation procedure, where these penalties are first initialized with the identity matrix, then updated using the estimated error terms from the first step. 

2. *Sample splitting* involves dividing the sample to decorrelate the estimation of nuisance parameters from that of the parameter of interest. An advantage discussed in Section 7.2 is that, in theory, it allows for more non-zero components (condition (7.7)). A drawback of partitioning is a certain loss of efficiency.


## 8. Inference on heterogeneous effects

1. Le problème fondamental de l'inférence causale est que l'on n'observe jamais la *ground truth*, c'est-à-dire le véritable effet du traitement. Il n'est donc pas possible de réutiliser sans modification des procédures standards telles que la validation croisée, du moins pas avec la même efficacité.

2. L'auteur oppose deux stratégies d'attribution d'un traitement (par exemple une campagne d'e-mailing ou d'appels téléphoniques) : soit cibler ceux qui sont les plus susceptibles de ne pas acheter à nouveau, soit cibler ceux qui répondent le plus lorsqu'ils reçoivent le traitement. Ces deux populations ne sont pas nécessairement constituées des mêmes individus. Cibler ceux qui sont les plus susceptibles de ne pas acheter à nouveau est intuitivement logique, mais c'est aussi inefficace : le traitement devrait être attribué de manière à maximiser son impact, mesuré comme l'augmentation de la probabilité d'achat. L'apprentissage automatique générique pourrait être utilisé pour classer les personnes en groupes.

3. Les splits aléatoires dans la construction des forêts aléatoires imposent d'explorer tout le support des variables explicatives et évitent la concentration des coupures dans une partie limitée de l'espace, ce qui serait obtenu avec des divisions sélectionnées de manière optimale. Ceci est clé pour obtenir la consistance de l'estimateur. 

4. On peut estimer le CATE, mais pas de manière convergente. Ainsi, seulement certains caractéristiques du CATE telles que le GATES ou le CLAN peuvent être estimées.

5. Dans ce cas, on a uniquement des intervalles de confiance \textbf{conditionnels} $$ \mathbb{P}\left(\theta_A \in [L_A, U_A] \middle| \  \text{Data}_A \right) = 1 - \alpha + o_P(1),$$ où $[L_A, U_A]:= \left[ \widehat{\theta}_A \pm \Phi^{-1}(1-\alpha/2)  \widehat{\sigma}_A  \right] $. Cela ne tient pas compte de la variabilité introduite par le fractionnement de l'échantillon, qui empêche toute généralisation à une quelconque distribution de l'ensemble des données.

6. Si le même échantillon était utilisé pour estimer les splits et les valeurs sur les feuilles, l'algorithme aurait tendance à séparer deux feuilles qui ont des effets de traitement hétérogènes (relativement élevés et faibles) dans cet échantillon, conduisant ainsi à une estimation biaisée si nous utilisons l'échantillon pour l'évaluer. Si nous utilisons un autre échantillon pour l'évaluer, cela limite le sur-apprentissage et assure la convergence.

7. L'objectif des forêts aléatoires causales est d'estimer un effet de traitement de manière consistante alors que les forêts aléatoires estiment une fonction de régression et visent à minimiser l'erreur de prédiction (souvent en norme $\ell_2$, ou MSE). Cela a des conséquences sur la forme de l'estimateur, les forêts aléatoires causales nécessitant la propriété d'honnêteté pour être consistante.

8. Le meilleur prédicteur linéaire du CATE est la projection linéaire d'un signal sans biais du CATE sur l'espace vectoriel linéaire généré par $T$. En ce sens BLP dépend donc des performances de $T$. S'il s'adapte bien au CATE, alors le coefficient de pente du BLP sera proche de un et nous apprendrons des caractéristiques du CATE en regardant $T$.


## 9. Optimal policy learning

1. A l'aide de l'hypothèse 9.2 (i), la contrainte est pertinente et donc toujours saturée pour la politique optimale, i.e. $c = \int_{x \in \mathcal{X}}\pi(x) dF_X(x)$. Soit $\pi'$ une politique optimale différente de $\pi$ donnée en Proposition 9.1 sur un ensemble de $F_X$ mesure non nulle. Cette politique satisfait aussi la contrainte, et avec l'hypothèse 9.2 (ii), il existe des ensembles $\Omega'$ et $\Omega$, tel que 
    $$ \int_{\Omega'} \pi'(x) dF_{X}(x) = \int_{\Omega} \pi(x) dF_{X}(x) ,$$
     $\Omega' \subseteq \{ x: \ \tau(x) < \gamma \}$, et $\Omega \subseteq \{ x: \ \tau(x) \geq \gamma \}$. On a donc 
    \begin{align*}
         \int_{\Omega'} \pi'(x) \tau(x) dF_{X}(x) & <  \gamma  \int_{\Omega'} \pi'(x)  dF_{X}(x)\\
         &= \gamma  \int_{\Omega} \pi(x)  dF_{X}(x) \\
         & \leq \int_{\Omega} \pi(x) \tau(x) dF_{X}(x),
    \end{align*}
    et on obtient une contradiction.

2. L'objectif utilisé est le contrôle du regret dans le pire des cas (*minimax regret criterion*), voir (9.6). L'avantage est la robustesse des politiques recommandées aux différentes distributions des effets possibles. L'inconvénient est que l'on utilise pas de potentiel *apriori* sur ces effets qui permettraient d'obtenir des meilleurs résultats s'ils sont vérifiés. 

3. Il est souhaitable de limiter les classes de politiques considérées de manière à pouvoir les implémenter simplement sur le terrain, mais aussi pour obtenir une borne supérieure sur le regret (Théorème 9.1).

4. On limite la complexité des classes de politiques considérées à l'aide de la dimension de Vapnik-Chervonenkis (VC). On peut autoriser des classes dont la complexité augmente avec la taille d’échantillon, mais moins rapidement que $n$.

5. see Remark 9.2.

6. Cette formulation fait apparaître l’apprentissage de politiques par maximisation empirique comme un problème d’optimisation pondérée dans le cadre d’une classification. On peut utiliser les outils développés en classification pondérées
(voir e.g., Athey et Wager, 2021 ; Zhou et al., 2018 pour plus de détails) pour résoudre ce problème.


## 10. The synthetic control method

1. No, that is not the case. Indeed, there is only one treated unit, so no law of large numbers-type result applies.

2. There should be three possible answers among the following: (1) no extrapolation (the weights are non-negative and sum to one), (2) transparency of fit (the pre-treatment fit can be assessed), (3) guards against specification search (the weights can be computed independently of the post-treatment outcome), (4) sparsity/qualitative interpretation (at most $p+1$ weights are strictly positive). See the corresponding chapter.

3. The synthetic control weight vector is generally sparse, meaning that only a few entries are non-zero. As a result, only a few control units contribute to the counterfactual. On one hand, this means the full sample is not used (loss of efficiency?), but on the other hand, it excludes units that do not help in reproducing the treated unit.

4. see section 10.4.
     
5. It is impossible because the test of no treatment effect is rejected with a p-value of 0.02. Hence, 0 cannot be inside the 90\% confidence interval. 

6. Let $D^{obs} =(D_1,\ldots, D_n)$ the observed vector of treatment assignment and $\hat \tau^{obs}$ the corresponding OLS estimator. Fisher's procedure is :
     - For $b=1, \ldots, B$, reshuffle treatment assignment at random, compute the OLS estimator of $\tau_0$, $\hat \tau_b$ and compare it to $\hat \tau^{obs}$. 
     - Compute Fisher's p-value : 

        $$\hat p := \frac{1}{B} \sum_{b=1}^{B} \mathbf{1} \left\{ \vert \hat \tau_b \vert  \ge \vert \hat \tau^{obs} \vert \right\}.$$

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

2. La réponse à la question précédente indique que pour deux mots distincts la similarité cosinus est de zéro. Il n'est donc pas possible d'obtenir une notion de << degré de similarité >> entre deux mots avec cette approche.

3. Un plongement lexical est une technique utilisée pour représenter les mots d'un texte sous forme de vecteurs de dimension réduite. Le but d'un plongement lexical est de capturer de manière compacte les relations sémantiques entre les mots dans le texte, de manière à pouvoir utiliser ces représentations vectorielles pour effectuer des tâches de traitement du langage naturel. On peut ensuite utiliser ces vecteurs pour représenter un document en les agrégeant (par exemple via une moyenne), puis en utilisant cette représentation par exemple dans une régression logistique.

4. see section 13.2.

5. see section 13.2.

6. see section 13.3.

7. XXX


## 14. Modern language models

1. L'idée est d'avoir une mesure de la qualité d'un produit telle que perçue par le consommateur. Par exemple, on peut penser à inclure dans un modèle la moyenne des embeddings des commentaires d'évaluation d'un produit. Ou bien les mebeddings des photos de ce produit.
    
2. see section 14.2.

3. see section 14.2.2.

4. Par l'utilisation de n-grams de mots. Les limites sont qu'il peut être compliqué de prendre en compte des dépendances longues, que le nombre de tokens augmente de manière exponentielle, et que l'algorithme ainsi entrainé ne pourra pas s'adapter à une séquence de mot jamais vue.
