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

5. In Section 3.4/4.b (simulation study from the high-dimension and endogeneity chapter.

6. It is also a variable selection procedure that uses only a single equation, and is therefore subject to the problem of post-selection inference.


## 6. High dimension and endogeneity


1. Soit la liste des instruments disponibles et possibles est grande, alors que le chercheur sait que seuls quelques-uns d'entre eux sont pertinents ; mais surtout, même lorsqu'on ne dispose que d'un seul instrument $Z$, on peut aussi envisager des transformations de l'instrument initial 

    $$\left(f_1(Z), \dots, f_p(Z) \right).$$

    par une famille de fonctions $(f_i)_{i=1}^p$, ce qui nous ramène au cas de très nombreux instruments possibles. Dans le cours, nous utilisons un modèle parcimonieux pour les variables instrumentales, en supposant que seuls quelques instruments sont effectivement utiles, et nous fournissons une méthode basée sur Lasso pour estimer le traitement tout en contrôlant l'estimation du paramètre de nuisance.

- Sans le terme $E[D |X ]$ dans (6.10), on peut calculer par exemple que $\partial_{\nu} M(\tau_0,\eta_0) = \tau_0 \E[ X(Z'\delta_0 + X'\gamma_0)]$, qui est différent de 0 en général.

- On utilise la formule de $\sigma_{\Gamma}^2$ donnée en (6.15), et on obtient que

     $$ \sigma_{\Gamma}^2 = & \E[\varepsilon^2 \E[D|Z]^2 ] / \E[\E[D|Z]^2  ]^2 = \sigma^2 \E[\E[D|Z]^2  ]^{-1}.$$

    en utilisant l'hypothèse d'homoscedasticité conditionnelle. On a donc que la borne (3.14) d’efficacité semi-paramétrique est atteinte (dans ce cas où $S = D$, $W=Z$ avec les notation de la section 3.4). 

- On vérifie que l'hypothèse (ORT) $\partial_{\eta} M(\tau_0,\eta_0)=0$ est satifaite en remarquant qu'avec (6.11) et (6.2), on obtient $\partial_{\nu} M(\tau_0,\eta_0) = \tau_0 \E[ X (\zeta'\delta_0)] =0$, $\partial_{\theta} M(\tau_0,\eta_0) = -\E[ X (\zeta'\delta_0)]=0$. Les deux autres conditions par rapport à  $\gamma$ et $\delta$ sont aussi vérifiées en utilisant (6.3) et (6.4).
- En utilisant les notations de (6.17), on a $u_{i,j} = \delta_j + \varepsilon_{i,j}$ et donc 
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

- Pour prendre en compte les erreurs non gaussiennes et hétéroscédastiques, la pénalité $\ell_1$ dans l'estimation Lasso standard est modifiée en utilisant des pénalité spécifiques conçues de manière à pouvoir appliquer les résultats de la théorie des déviations modérées. Cela donne une procédure d'estimation en deux étapes où ces pénalités sont initialisées avec la matrice d'identité, puis mises à jour en utilisant les termes d'erreur estimés de la première étape.

- Le \textit{sample-splitting} consiste à couper l'échantillon de manière à dé-corréler l'estimation des paramètres de nuisance de celle du paramètre d'intérêt. Un avantage exposé dans la section 7.2 est qu'en théorie on peut autoriser plus nombre de composantes non nulles (condition (7.7)). Un inconvénient du partitionnement est une certaine perte d'efficacité. 


## 8. Inference on heterogeneous effects

- Le problème fondamental de l'inférence causale est que l'on n'observe jamais la \textit{ground truth}, c'est-à-dire le véritable effet du traitement. Il n'est donc pas possible de réutiliser sans modification des procédures standards telles que la validation croisée, du moins pas avec la même efficacité.

- L'auteur oppose deux stratégies d'attribution d'un traitement (par exemple une campagne d'e-mailing ou d'appels téléphoniques) : soit cibler ceux qui sont les plus susceptibles de ne pas acheter à nouveau, soit cibler ceux qui répondent le plus lorsqu'ils reçoivent le traitement. Ces deux populations ne sont pas nécessairement constituées des mêmes individus. Cibler ceux qui sont les plus susceptibles de ne pas acheter à nouveau est intuitivement logique, mais c'est aussi inefficace : le traitement devrait être attribué de manière à maximiser son impact, mesuré comme l'augmentation de la probabilité d'achat. L'apprentissage automatique générique pourrait être utilisé pour classer les personnes en groupes.

- Les splits aléatoires dans la construction des forêts aléatoires imposent d'explorer tout le support des variables explicatives et évitent la concentration des coupures dans une partie limitée de l'espace, ce qui serait obtenu avec des divisions sélectionnées de manière optimale. Ceci est clé pour obtenir la consistance de l'estimateur. 

- On peut estimer le CATE, mais pas de manière convergente. Ainsi, seulement certains caractéristiques du CATE telles que le GATES ou le CLAN peuvent être estimées.

- Dans ce cas, on a uniquement des intervalles de confiance \textbf{conditionnels} $$ \mathbb{P}\left(\theta_A \in [L_A, U_A] \middle| \  \text{Data}_A \right) = 1 - \alpha + o_P(1),$$ où $[L_A, U_A]:= \left[ \widehat{\theta}_A \pm \Phi^{-1}(1-\alpha/2)  \widehat{\sigma}_A  \right] $. Cela ne tient pas compte de la variabilité introduite par le fractionnement de l'échantillon, qui empêche toute généralisation à une quelconque distribution de l'ensemble des données.

- Si le même échantillon était utilisé pour estimer les splits et les valeurs sur les feuilles, l'algorithme aurait tendance à séparer deux feuilles qui ont des effets de traitement hétérogènes (relativement élevés et faibles) dans cet échantillon, conduisant ainsi à une estimation biaisée si nous utilisons l'échantillon pour l'évaluer. Si nous utilisons un autre échantillon pour l'évaluer, cela limite le sur-apprentissage et assure la convergence.

- L'objectif des forêts aléatoires causales est d'estimer un effet de traitement de manière consistante alors que les forêts aléatoires estiment une fonction de régression et visent à minimiser l'erreur de prédiction (souvent en norme $\ell_2$, ou MSE). Cela a des conséquences sur la forme de l'estimateur, les forêts aléatoires causales nécessitant la propriété d'honnêteté pour être consistante.

- Le meilleur prédicteur linéaire du CATE est la projection linéaire d'un signal sans biais du CATE sur l'espace vectoriel linéaire généré par $T$. En ce sens BLP dépend donc des performances de $T$. S'il s'adapte bien au CATE, alors le coefficient de pente du BLP sera proche de un et nous apprendrons des caractéristiques du CATE en regardant $T$.


## 9. Optimal policy learning

- A l'aide de l'hypothèse 9.2 (i), la contrainte est pertinente et donc toujours saturée pour la politique optimale, i.e. $c = \int_{x \in \mathcal{X}}\pi(x) dF_X(x)$. Soit $\pi'$ une politique optimale différente de $\pi$ donnée en Proposition 9.1 sur un ensemble de $F_X$ mesure non nulle. Cette politique satisfait aussi la contrainte, et avec l'hypothèse 9.2 (ii), il existe des ensembles $\Omega'$ et $\Omega$, tel que 
    $$ \int_{\Omega'} \pi'(x) dF_{X}(x) = \int_{\Omega} \pi(x) dF_{X}(x) ,$$
     $\Omega' \subseteq \{ x: \ \tau(x) < \gamma \}$, et $\Omega \subseteq \{ x: \ \tau(x) \geq \gamma \}$. On a donc 
    \begin{align*}
         \int_{\Omega'} \pi'(x) \tau(x) dF_{X}(x) & <  \gamma  \int_{\Omega'} \pi'(x)  dF_{X}(x)\\
         &= \gamma  \int_{\Omega} \pi(x)  dF_{X}(x) \\
         & \leq \int_{\Omega} \pi(x) \tau(x) dF_{X}(x),
    \end{align*}
    et on obtient une contradiction.
- L'objectif utilisé est le contrôle du regret dans le pire des cas (\textit{minimax regret criterion}), voir (9.6). L'avantage est la robustesse des politiques recommandées aux différentes distributions des effets possibles. L'inconvénient est que l'on utilise pas de potentiel \textit{apriori} sur ces effets qui permettraient d'obtenir des meilleurs résultats s'ils sont vérifiés. 
- Il est souhaitable de limiter les classes de politiques considérées de manière à pouvoir les implémenter simplement sur le terrain, mais aussi pour obtenir une borne supérieure sur le regret (Théorème 9.1).
- On limite la complexité des classes de politiques
considérées à l'aide de la dimension de Vapnik-Chervonenkis (VC). On peut autoriser des classes dont la complexité augmente avec la taille d’échantillon, mais moins rapidement que $n$.
- see Remarque 9.2.
- Cette formulation fait apparaître l’apprentissage de politiques par maximisation empirique comme un problème d’optimisation pondérée dans le cadre d’une classification. On peut utiliser les outils développés en classification pondérées
(voir e.g., Athey et Wager, 2021 ; Zhou et al., 2018 pour plus de détails) pour résoudre ce problème.


## 10. The synthetic control method

 - Non, ce n'est pas le cas. En effet, il n'y a qu'une seule unité traitée, donc aucun résultat de type loi des grands nombres ne s'applique.
     
- Il devrait y avoir trois possibilités parmi : (1) pas d'extrapolation (les poids sont non négatifs et leur somme est égale à un), (2) transparence de l'ajustement (l'ajustement avant le traitement peut être évalué), (3) empêche la recherche de spécification (les poids peuvent être calculés indépendamment du résultat après traitement), (4) sparsité/interprétation qualitative (au plus $p+1$ sont strictement positifs). Voir le chapitre correspondant. Les réponses qui étaient trop génériques / également vraies pour d'autres estimateurs standard / non expliquées avec un argument précis ont été rejetées.
     
- Le vecteur du poids de contrôle synthétique est en général peu dense, ce qui signifie que seules quelques entrées ne sont pas des zéros. Par conséquent, les unités de contrôle correspondantes ne prennent pas part au contrefactuel. D'une part, il n'utilise pas l'échantillon complet (perte d'efficacité ?), mais d'autre part, il écarte les unités qui n'aident pas à reproduire l'échantillon traité.
     
 - see section 10.4.
     
 - Cela est impossible car on rejette le test d'absence d'effet du traitement avec une p-value de 0,02. 0 ne peut donc pas se trouver dans l'intervalle de confiance à 0,90.

 - Soit $D^{obs} =(D_1,\ldots, D_n)$ le vecteur observé de l'affectation du traitement et $\hat \tau^{obs}$ l'estimateur MCO correspondant. La procédure de Fisher est la suivante :
     \begin{enumerate}
     - Pour $b=1, \ldots, B$, remanier l'affectation du traitement de manière aléatoire, calculer l'estimateur MCO de $\tau_0$, $\hat \tau_b$ et le comparer aux statistiques observées $\hat \tau^{obs}$. 
     - Calculer la p-value de Fisher : 
            \begin{align*}
             \hat p := \frac{1}{B} \sum_{b=1}^{B} \mathbf{1} \left\{ \vert \hat \tau_b \vert  \ge \vert \hat \tau^{obs} \vert \right\} 
         \end{align*}
     - Rejeter $H_0$ si $\hat p$ est inférieur à un seuil prédéterminé : l'allocation de traitement observée donne un effet qui est anormalement grand.

## 11. Forecasting in high-dimension

- On ne peut plus utiliser l'hypothèse d'indépendance des observations, qui sont corrélées dans le temps. Les séries temporelles économiques et financières sont aussi connues pour posséder des queues de distribution épaisses. Enfin, nous devons aussi prendre en compte le fait que les séries ne sont pas échantillonnées à la même fréquence.
- Certaines variables explicatives ont souvent une structure particulière, qui fait que peu de groupes de variables (macroéconomie, différents secteurs d’activité, variables financières, news) peuvent être utiles pour la prédiction, mais au sein de ces groupes plusieurs variables le sont. 
- La pénalité du Lasso impose la parcimonie et conduit à un biais qu'il est nécessaire de prendre en compte quand il s'agit de faire de l'inférence sur un groupe de coefficients (voir section 7.3).
- Le risque à utiliser une méthode imposant la parcimonie est un risque de mauvaise spécification, c'est à dire que cette hypothèse peut ne pas être vérifiée, entraînant un biais. Certains exemples empiriques mentionnés en section 11.2.1 montrent que cette hypothèse doit être justifiée avec précaution. L'approche FARM permet de combiner une partie parcimonieuse avec une partie dense, et donc de tester si cette dernière est utile. 

## 12. Working with textual data

- Le vocabulaire d'un texte peut être très vaste, ce qui nécessite de représenter chaque mot par un vecteur de grande dimension via une représentation one-hot (see chapitre 13). Dans les modèles plus avancés, les vecteurs de grande dimension permettent de capturer plus de détails et de nuances dans le texte.

- Voici deux exemples parmi une infinité. (i) Traitement des erreurs de frappe.  L'utilisation de n-grams de caractères permet de contourner ce problème en permettant de représenter le texte à un niveau inférieur, au niveau des séquences de caractères. (ii) Traitement des formes plurielles.

- Les modèles faisant usage de variables latentes permettent de capturer le contexte général d'un document. Néanmoins, ces modèles sont d'une part complexe à estimer, et ont été, d'autres parts, dépassés par les modèles modernes de langage pré-entrainés en terme de performance.

- Le premier problème consiste à apparier chaque message du forum avec le ou les actif(s) financier(s) mentionnés. Cette phase dépend largement de la qualité des données. Si les messages mentionnent systématiquement des symboles boursiers (AAPL pour Apple Inc., GME pour GameStop etc.) il suffit de se procurer une liste de ces symboles puis d'en cherche les occurrences dans les messages. En utilisant les dates des messages, on peut ensuite apparier chaque message au(x) prix des actifs mentionnés. Si les messages ne mentionnent pas de symboles boursiers, il faut soit adopter une approche plus sophistiquée (distance aux noms des entreprises etc.), soit adopter une approche plus \textit{data-driven}, par exemple de classification <<~sac de mots~>> avec un grand nombre de classes (autant que d'actifs). Une fois l'appariement des données effectué, un grand nombre de stratégies sont possibles (e.g. analyse du sentiment des messages).

## 13. Word embeddings

- La similarité cosinus pour deux mots distincts est de zéro.

- La réponse à la question précédente indique que pour deux mots distincts la similarité cosinus est de zéro. Il n'est donc pas possible d'obtenir une notion de << degré de similarité >> entre deux mots avec cette approche.

- Un plongement lexical est une technique utilisée pour représenter les mots d'un texte sous forme de vecteurs de dimension réduite. Le but d'un plongement lexical est de capturer de manière compacte les relations sémantiques entre les mots dans le texte, de manière à pouvoir utiliser ces représentations vectorielles pour effectuer des tâches de traitement du langage naturel. On peut ensuite utiliser ces vecteurs pour représenter un document en les agrégeant (par exemple via une moyenne), puis en utilisant cette représentation par exemple dans une régression logistique.

- see section 13.2.

- see section 13.2.

- see section 13.3.

- On espère que l'on trouvera le vecteur représentant le mot <<~taureau~>>.

## 14. Modern language models

- L'idée est d'avoir une mesure de la qualité d'un produit telle que perçue par le consommateur. Par exemple, on peut penser à inclure dans un modèle la moyenne des embeddings des commentaires d'évaluation d'un produit. Ou bien les mebeddings des photos de ce produit.
    
- see section 14.2.

- see section 14.2.2.

- Par l'utilisation de n-grams de mots. Les limites sont qu'il peut être compliqué de prendre en compte des dépendances longues, que le nombre de tokens augmente de manière exponentielle, et que l'algorithme ainsi entrainé ne pourra pas s'adapter à une séquence de mot jamais vue.
