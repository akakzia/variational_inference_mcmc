---
title: "<font color='#DB3A80'>Modèle linéaire bayèsien</font>"
output:
  html_document: default
  pdf_document: default
---



## I - Introduction
L'objectif principal pour lequel les modèles probabilistes sont gènèralement utilisès est l'évaluation de la loi a posteriori $p(Y | X)$ où $Y$ joue le rôle des variables latentes et $X$ des variables observés. Les espérences sont ensuite calculées à partir de cette loi. <br><br>
Le concept de l'optimisation variationnelle peut être utilisé pour traiter un problème d'inférence. On se place donc dans le cadre d'un modèle bayésien avec des densités a priori données pour chaque paramètre. Le modèle peut aussi avoir des variables latentes. On désigne par $Y$ les variables latentes et les paramètres, et par $X$ les variables observées. Le modèle considéré est caractérisé par la loi jointe $p(X,Z)$ et l'objectif essentiel et d'approximer la loi a posteriori $p(Z | X)$ ainsi que l'évidence $p(X)$ (dans certains cas on travaille avec le log-évidence pour simplifier). On a dèjà vu qu'on peut décomposer le log-évidence en somme de deux termes intéressants incluant la distance de Kullback Leibler : <br>
$$
\begin{equation}
  ln p(X) = \mathcal{L}(q) + KL(q||p)
\end{equation}
$$
avec 
$$
\begin{align}
  \mathcal{L}(q) &= \int q(Y) ln \left( \frac{p(X,Y)}{q(Y)} \right) dY\\
  KL(q||p) &= - \int q(Y) ln \left( \frac{p(Y|X)}{q(Y)} \right) dY
\end{align}
$$
L'algorithme considéré permettra de maximiser la borne inférieure $\mathcal{L}(q)$ par rapport aux distributions $q(z)$, ce qui revient de même Ã  minimiser la distance $KL$. On va poser une dernière loi de gouvernence qui permettra d'approximer la loi a posteriori : on suppose que les distributions $q$ appartiennent à l'ensemble des lois produits.<br><br>
Dans le cadre de ce document, on dispose de données $(x_i,y_i)_{i=1..n}$ et on se donne un vecteur de fonctions de base $h=(\phi_1,...,\phi_p)$. On considère le modèle linéaire probabiliste $Y = \phi \theta + \epsilon$ où les $\epsilon = (\epsilon_1,...\epsilon_p)$ sont des bruits gaussiens indépendants tel que $\epsilon \sim \mathcal{N}(0, \beta^{-1} I_n)$. On appelle $\phi$ la matrice de design.<br>
On adope un point de vue gaussien hiérarchique et on impose les priors suivants : 
$$
\begin{align}
  Y|\theta , \beta &\sim \mathcal{N} ( \phi \theta, \beta^{-1} I_n)\\
  \theta | \alpha &\sim \mathcal{N} ( 0, \alpha^{-1} I_p)\\
  \alpha &\sim Gamma(a_0, \lambda_0)\\
  \beta &\sim Gamma(b_0, \tau_0)
\end{align}
$$
```{r}
# Dependencies 
library(coda)
library(mvtnorm)
```

## II - Approximation variationnelle 

La loi jointe s'écrit de la manière suivante : $p(y, \theta, \alpha, \beta) = p(y|\theta, \beta) p(\theta| \alpha) p(\alpha) p(\beta)$ et on considère l'approximation suivante de la loi a posteriori : $p(\theta, \alpha, \beta | y) = q(\theta, \alpha, \beta) = q_1(\theta) q_2(\alpha) q_3(\beta)$.<br>
Un calcul un peu long mais facile montre que les solutions de champs moyen $q_1^*$, $q_2^*$, $q_3^*$, vérifient : 

$$
\begin{align}
  q_1^*(\theta) &= \mathcal{N} ( \theta | \tilde{m_n}, \tilde{S_n})\\
  q_2^*(\alpha) &= Gamma(\alpha | a_n, \lambda_n)\\
  q_3^*(\beta) &= Gamma(\beta | b_n, \tau_n)
\end{align}
$$
On se propose d'établir le résultat ci-dessus concernant $\beta$. D'après l'approximation des champs moyens on a : 

$$
\begin{align}
  ln q_3^*(\beta) &= \mathbb{E}_{\theta , \alpha} [ln p(y, \theta, \alpha, \beta)]\\
  &= \mathbb{E}_{\theta , \alpha} [ln ( p(y|\theta, \beta) p(\theta| \alpha) p(\alpha) p(\beta) )]\\
  &= \mathbb{E}_{\theta , \alpha} [ln p(y|\theta, \beta) + ln p(\beta)] + cte\\ 
  \end{align}
$$
Ici on ne garde que les termes qui dépendent de $\beta$, on met tout le reste dans la constante. On connais les lois des variables en question, l'expression devient donc : 

$$
\begin{align}
  ln q_3^*(\beta) &= \mathbb{E}_{\theta , \alpha}[ln \frac{1}{det(\beta^{-1} I_n)^{1/2}}exp (\frac{-1}{2} (y-\Phi \theta)(\beta^{-1}I_n)^{-1}(y-\Phi \theta)^T)  + ln \beta^{b_0-1} e^{-\tau_0 \beta}] + cte\\
  & = \mathbb{E}_{\theta , \alpha} [ \frac{n}{2} ln(\beta) - \frac{1}{2} (y-\Phi \theta)(\beta I_n)(y-\Phi \theta)^T + (b_0 - 1) ln\beta -\tau_0 \beta]\\
  &=  \mathbb{E}_{\theta , \alpha} [ (b_0 + \frac{n}{2} - 1 )ln(\beta) - (\tau_0 + \frac{1}{2} (y-\Phi \theta)(y-\Phi \theta)^T) \beta]\\
  &=   (b_0 + \frac{n}{2} - 1 )ln(\beta) - (\tau_0 + \frac{1}{2} \mathbb{E}_{\theta , \alpha} [||y-\Phi \theta||^2) \beta\\
  &= (b_n - 1)ln\beta - \tau_n \beta\\
  \Rightarrow q_3^*(\beta) &= cte \times \beta^{b_n -1} e^{-\tau_n \beta}\\
  &avec\ \ b_n = b_0 + \frac{n}{2}\ \ \ et \ \  \tau_n = \tau_0 + \frac{1}{2} \mathbb{E}_{\theta , \alpha} [||y-\Phi \theta||^2) 
\end{align}
$$
Dans la partie suivante, on se propose d'appliquer l'algorithme d'approximation variationnelle par mise à jour successive des équations variationnelles. Pour ce faire, on considère l'expression suivante de la borne inférieur de la log évidence du modèle considéré : 
$$
\begin{align}
  \mathcal{L}(q) &= \mathbb{E}_q(ln\frac{p(y, \theta, \alpha, \beta)}{q(\theta, \alpha, \beta)})\\
  &= \mathbb{E}_{q^*}(ln p(y | \theta, \beta)) + \mathbb{E}(ln p(\theta | \alpha)) + \mathbb{E}_{q^*}(ln p(\alpha)) + \mathbb{E}_{q^*}(ln p(\beta))\\ &- \mathbb{E}_{q^*}(ln q^*(\theta)) - \mathbb{E}_{q^*}(ln q^*(\alpha)) - \mathbb{E}_{q^*}(ln q^*(\beta))
\end{align}
$$
On considère les fonction *variational_lowerbound* et *variational_update* suivantes. La première sert à calculer la borne inférieure Ã  partir des valeurs de la variable cible, des paramètres et des hyperparamètres prises comme entrée. Tant dis que la deuxième met à jour les valeurs des paramètres qu'on a trouvés par approximation des champs moyens.

```{r}
variational_lowerbound <- function(Phi, target, varpar , hyperpar)
  ## ARGUMENTS: 
  ## Phi: design matrix (feature map applied to data): a n*p matrix
  ## target: a  vector of size n: the observed values y.
  ## varpar: a list of variational parameters containing the following entries: 
  ##     m_n: variational expectancy of theta
  ##     S_n: variational covariance of theta
  ##     a_n, lambda_n: variational gamma parameters  for alpha (prior concentration of theta)
  ##     b_n, tau_n: variational gamma parameters for beta (prior concentration of noise)
  ## hyperpar: a list of hyperparameters containing the following entries: 
  ##     a_0, lambda_0 b_0  tau_0:   hyper-prior gamma parameters respectively for alpha and beta
  ## 
## RETURNS: the variational lower bound 
{
  list2env(varpar, envir=environment())
  list2env(hyperpar, envir = environment() )
  
  n <- length(target)
  p <- ncol(Phi)
  
  Elnp_y <- n/2 *(- log(2*pi) + digamma( b_n) - log(tau_n)) -
    b_n/(2*tau_n) * (sum( (target - Phi%*%m_n)^2 ) + sum(diag( Phi%*%S_n%*%t(Phi))))
  
  Elnp_theta <- p/2 * (- log(2*pi) + digamma( a_n) - log(lambda_n)) -
    a_n/(2* lambda_n) * ( sum(m_n^2) + sum(diag( S_n))  )
  
  Elnp_alpha <- a_0 * log(lambda_0) + (a_0 - 1) * (digamma( a_n) - log(lambda_n)) -
    lambda_0 * (a_n / lambda_n) - lgamma(a_0)
  
  
  Elnp_beta <- b_0 * log(tau_0) + (b_0 - 1) * (digamma( b_n) - log(tau_n)) -
    tau_0 * (b_n / tau_n) - lgamma(b_0)
  
  Elnq_theta <- - p/2 * (1 + log(2*pi)) - 1/2 * log(det( S_n))
  Elnq_alpha <- - lgamma(a_n) + (a_n - 1) * digamma( a_n ) + log(lambda_n) - a_n
  Elnq_beta <- - lgamma(b_n) + (b_n - 1) * digamma( b_n ) + log(tau_n) - b_n
  
  lowerbound <- Elnp_y + Elnp_theta + Elnp_alpha +  Elnp_beta - Elnq_theta - Elnq_alpha - Elnq_beta
  
  return(lowerbound)
  
}
```

```{r}
variational_update <- function(Phi, target,
                               currentpar, hyperpar)
  ## Phi: design matrix (feature map applied to data): a n*p matrix
  ## target: a  vector of size n: the observed values y.
  ## varpar: a list of variational parameters containing the following entries: 
  ##     m_n: variational expectancy of theta
  ##     S_n: variational covariance of theta
  ##     a_n, lambda_n: variational gamma parameters  for alpha (prior concentration of theta)
  ##     b_n, tau_n; variational gamma parameters for beta (prior concentration of noise)
  ## hyperpar: a list of hyperparameters containing the following entries: 
  ##     a_0, lambda_0, b_0,  tau_0:   hyper-prior gamma parameters respectively for alpha and beta
  ## 
  ## RETURNS: a list of the same format as varpar containing the updated parameters. 
{
  list2env(currentpar, envir=environment())
  list2env(hyperpar, envir = environment() )
  n <- length(target)
  p <- ncol(Phi)
  
  S_n_inv <- a_n/lambda_n * diag(p) + b_n/tau_n * t(Phi)%*%Phi
  S_n <-  solve(S_n_inv)
  m_n <-  b_n / tau_n * S_n %*% t(Phi) %*% target
  a_n <- a_0 + p/2
  lambda_n <- lambda_0 + 1/2 * (sum(m_n^2) + sum(diag(S_n)) )
  b_n <- b_0 + n/2
  tau_n <- tau_0 + 1/2 * ( sum( ( target - Phi %*% m_n)^2 ) + sum(diag(Phi %*% S_n%*% t(Phi)))  )
  return(list(m_n = m_n, S_n = S_n, a_n = a_n, lambda_n = lambda_n, b_n = b_n, tau_n = tau_n))
  
}
```

L'algorithme *variational_lm* fournit tout le travail d'approximation variationnelle. En prenant des valeurs d'initialisation quelconque (mais bien sûr vérifiant les conditions de dimension), il met à jour ces paramètres là à l'aide de la fonction *variational_update*. Ensuite, en fonction des paramètres mis à jour, il calcule la borne inférieure correspondante à l'aide de la fonction *variational_lowerbound*. On teste ensuite la convergence de l'algorithme en comparant la borne inférieure retrouvée à une valeur donnée de tolérence. Pour éviter que l'algorithme bloque, on ajoute comme entrée le nombre maximal d'itérations à ne pas dépasser. 

```{r}
variational_lm <- function(Phi, target,  hyperpar,  maxiter = 100, tol = 1e-4)
  ## ARGUMENTS: 
  ## Phi: design matrix (feature map applied to data): a n*p matrix
  ## target: a  vector of size n: the observed values y.
  ## hyperpar: a list of hyperparameters containing entries a_0, lambda_0, b_0, tau_0:
  ##          hyper-prior gamma parameters respectively for alpha and beta
  ## maxiter: the maximum number of variational updates
  ## tol: the stopping criterion for the variational updates: the algorithm stops if the
  ##     standard deviation of the variational lower bound  over the latest 5 iterations
  ## is less than tol. 
  ##
  ## RETURNS:  a list with entries (convergence, var_par, lowerbound, niter)
##     convergence = 0 if the stopping criterion was met (successful convergence), 1 otherwise. 
##     var_par: a list containing the variational parameters at the last iteration, that is
##           (m_n, S_n, a_n, lambda_n, b_n, tau_n)
##     lowerbound: the lower bound values across all iterations of the optimisation algorithm. 
##     niter: the number of iterations performed. 
{
  n <- length(target)
  p <- ncol(Phi)
  list2env(hyperpar, envir=environment())
  ## list2env makes all elements of the list visible from the function's environment.
  
  ## Initialization 
  niter <- 0 ## number of iterations performed  
  currentpar <- list(m_n =  matrix(0, p, 1) ,
                     S_n =  diag(p) , 
                     a_n =  0.1 , 
                     lambda_n = 0.1 ,
                     b_n = 0.1  , 
                     tau_n = 0.1)
  lowerbound <- c() ## empty vector. 
  delta <- tol+1  ## stopping criterion not satisfied at the initialization step
  continue <- TRUE  
  while(continue)
  {
    currentpar = variational_update(Phi, target, currentpar, hyperpar)
    lowerbound = c(lowerbound, variational_lowerbound(Phi, target, currentpar, hyperpar))
    niter = niter + 1
    ##  the stopping criterion is implemented below:
    if(niter >5){
      delta <- sd(lowerbound[niter:(niter-5) ]) 
      if(delta <tol){ continue <- FALSE}
    }
  }
  if (delta < tol ){ cv <- 0} else{cv <- 1}
  return(list( convergence = cv , varpar = currentpar, lowerbound = lowerbound, niter = niter  ))
}

```
Afin de tester l'algorithme, on procède comme au premier TP. On génére une population $x=(x_1,...,x_n)$ de taille $n=100$ uniformément distribué sur l'intervalle $[-3,3]$. On prend comme fonctions de base les fonctions polynomiales $\phi_0,...\phi_4$ où $\phi_i(x) = x^i$. On générera des observations $y_i$ suivant le modèle considéré avec comme vrais paramètres $\beta_0 = 0.1$ et $\theta_0 = (5, -2, 1, -1, 1)$.

```{r}
Beta0 <- 0.1
  
theta0 <- c(5, 2,1,-1,1)

Fmap = function(x){c(1, x, x^2, x^3, x^4)}

N <- 100
set.seed(3) 
#'  allows result reproductibility  (sets the seed of the random number generator)
data_4 <- matrix(runif(N, min=-3, max = 3),ncol=1)

Phi_4 = matrix(sapply(data_4, Fmap), 100, 5, byrow=TRUE)

target_4 <-  Phi_4%*%theta0 + rnorm(N, sd = sqrt(Beta0)^(-1))

hyperpar <- list(a_0=0.1, lambda_0=0.1, b_0=0.1, tau_0=0.1)

L = variational_lm(Phi_4, target_4, hyperpar, maxiter = 100, tol =1e-4)
```
Dans le but d'évaluer la pertinence des résultats trouvés, on tracera sur un même graph le vrai vecteur $\theta_0$, l'espérence a posteriori de chaque élément de $\theta$ dans l'approximation variationnelle et les intervalles de crédibilité a posteriori à $95\%$ basés sur les quantiles a posteriori pour chaque élément de $\theta$. 
```{r}
data_4 <- matrix(data_4[1:N], ncol=1)
target_4<- matrix(target_4[1:N], ncol=1)

Iplus <- L$varpar$m_n + 1.96*sqrt(diag(L$varpar$S_n))
Iminus <- L$varpar$m_n - 1.96*sqrt(diag(L$varpar$S_n))
plot(theta0, pch=19,col='black',main = paste(" N = ", toString(N), sep=""), ylim = 
       range(Iminus, Iplus, theta0))
points(L$varpar$m_n, col='red')
arrows( x0 = 1:5, y0 = Iminus, 
        y1 = Iplus ,code=3, length=0.1, col="red")
```
On remarque que la fonction *arrows* nous permet de voir que les valeurs estimées a posteriori sont très proches des valeurs réelles de $\theta$. Ceci dit, on va comparer plus tard ces résultats avec les résultats de la mèthode "empirical Bayes" vue au premier TP. Mais avant, comparons l'espérence de $\beta$ a posteriori sous $q^*$ avec la vraie valeur $\beta_0$. Pour ce faire, on utilisera un intervalle de crédibilité a posteriori à $95\%$ basé sur le quantile a posteriori de $\beta$
```{r}
q = qgamma(0.95, shape = L$varpar$b_n, rate = L$varpar$tau_n)
Exp = L$varpar$b_n / L$varpar$tau_n
Iplus <- Exp + q
Iminus <- Exp - q
plot(Beta0, pch=19,col='black',main = paste(" N = ", toString(N), sep=""), ylim = 
       range(Iminus, Iplus, Beta0))
points(Exp, col='red')
arrows( x0 = 1, y0 = Iminus, 
        y1 = Iplus ,code=3, length=0.1, col="red")
```
On remarque que la valeur du paramètre $\beta$ se trouve bien dans l'intervalle de crédibilité, de plus elle est proche de la vraie valeur du paramètre, ce qui affirme encore la pertinence du modèle choisi.

On arrive maintenant à l'étape où on va comparer les résultats de la _méthode variationnelle_ avec ceux de la méthode _empirical bayes_. Pour ce faire, on considère les deux fonctions qu'on a utilisé au premier tp *glinear_fit* et *logevidence*. On veut comparer les valeurs optimales $\alpha^*$ et $\beta^*$ de la méthode _empirical bayes_ avec $\mathbb{E}_{q^*}(\alpha)$ et $\mathbb{E}_{q^*}(\beta)$ de la _méthode variationnelle_. 

```{r}
glinear_fit <- function(Alpha, Beta, data, feature_map, target)
    #' ARGUMENTS: 
    #' Alpha: prior precision on theta
    #' Beta: noise precision
    #' data: the input variables (x): a matrix with n rows where n is the sample size
    #'feature_map: the basis function, returning a vector of  size p equal to the dimension of theta
    #' target: the observed values y: a vector of size n
    #' RETURNS: a  list with entries (mean, cov) (osterior mean and variance)
  {
    
    Phi <-  t(apply(X= data, MARGIN=1, FUN = feature_map))
    p = ncol(Phi)
    posterior_variance_inverse <-  diag(x=Alpha, nrow= p) +
      Beta * t(Phi)%*%Phi
    posterior_variance <-  solve(posterior_variance_inverse)
    posterior_mean <-   Beta *
      posterior_variance %*% t(Phi)%*% target
    return(list(mean=posterior_mean, cov=posterior_variance))
  }
```

```{r}
logevidence <- function(Alpha, Beta, data ,feature_map, target)
    ## ARGUMENTS: 
    ## Alpha: prior precision for theta
    ## Beta: noise precision
    ## data: the input points x_{1:n}
    ## feature_map: the vector of basis functions
    ## target: the observed values y: a vector of size n.
    ## RETURNS: the logarithm of the model evidence forthe considered dataset. 
  {
    Phi_transpose <-  apply(X= data, MARGIN=1, FUN = feature_map)
    if(is.vector(Phi_transpose)){
      Phi_transpose = matrix(Phi_transpose,nrow=1)
    }
    Phi <- t(Phi_transpose)
    N <- nrow(Phi)
    p <- ncol(Phi)
    A <- Alpha*diag(p) + Beta * Phi_transpose %*% Phi
    postmean <- Beta * solve(A) %*% Phi_transpose %*% target
    energy <- Beta/2 * sum(( target - Phi%*%postmean)^2) + Alpha/2 * sum((postmean)^2)
    res <- p/2 * log(Alpha) + N/2 * log(Beta) - energy - 1/2 * log(det(A)) - N/2 * log(2*pi)
    return(res)    
  }
```

```{r}
optAB <- optim(par=c(1, 1),
               fn=function(par){-logevidence(Alpha=par[1], Beta=par[2], data_4 ,feature_map=Fmap, target_4)},
               method = "L-BFGS-B",
               lower=c(0.1 , 0.1), upper = c(250,30))

optAB$par
```

```{r}
c(L$varpar$a_n/L$varpar$lambda_n, L$varpar$b_n/L$varpar$tau_n)
```
```{r}
# Les valeurs des paramètres alpha et beta estimés par les deux méthodes sont quasiment les mêmes !
```

```{r}
mfit <- glinear_fit(Alpha = optAB$par[1], Beta = optAB$par[2],
                      data= data_4, feature_map = Fmap, target = target_4)

  plot(L$varpar$m_n, pch=19,col='black',main = paste(" N = ", toString(N), sep=""), ylim = range(Iminus, Iplus, theta0))
  points(mfit$mean, col='red')
```
```{r}
# Les valeurs estimées de theta est la même pour les deux méthodes, nous avons une superposition !  
```

### Choix du modèle : 
On considère maintenant le problème de prédiction de la distance d'arrêt en fonction de la vitesse du véhicule. On utilise pour cela le jeu de données *cars* de _R_. On adope le même modèle de régression linéaire sur une base de fonctions polynomiales comme décrit précédemment. On cherche à determiner maximal de la base polynomiale adoptée, c'est-à-dire qu'on veut adapter la dimension au jeu de données.

```{r}
data(cars)
names(cars)
```

```{r}
F6 <- function(x){c(1, x, x^2, x^3, x^4, x^5, x^6)}
F5 <- function(x){c(1, x, x^2, x^3, x^4, x^5)}
F4 <- function(x){c(1, x, x^2, x^3, x^4)}
F3 <- function(x){c(1, x, x^2, x^3)}
F2 <- function(x){c(1, x, x^2)}
F1 <- function(x){c(1, x)}
F0 <- function(x){1}
listF=list(F0,F1,F2,F3,F4,F5,F6)
```

```{r}
feature_cars = matrix(cars$speed, ncol=1)
target_cars = cars$dist
# Pour des degrés supérieurs à 4, S_n devient singulière
res <- sapply(0:5,FUN=function(i){
  Phi_temp = matrix(sapply(feature_cars, listF[[i+1]]), 50, i+1, byrow=TRUE)
  temp = variational_lm(Phi_temp, target_cars, hyperpar ,maxiter = 100, tol =1e-4)
  temp$lowerbound[temp$niter]
})

plot(0:5, res)

best_degre = which.max(res) - 1
```
```{r}
  ### Le degré du meilleur polynôme est égal à 2
```
### Prédictive approchée dans l'approximation variationnelle :
Dans cette partie, on continue à considérer le jeu de données *cars*. On utilise une nouvelle abscisse $x_{new}$ dans l'intervalle $[0,25]$ à laquelle on associe une nouvelle abscisse $Y_{new} = <h(x), \theta> + \epsilon$ avec $\epsilon \sim \mathcal{N}(0,\beta^{-1})$ et telle que $(\theta, \beta)$ sont distribués selon la loi a posteriori. On utilise ici l'approximation variationnelle de la loi a posteriori et on remplace la composante $q_3^*(\beta)$ par $\mathbb{E}_{q_3^*}(\beta)$. En effet, le calcul de la variance de la loi a posteriori de $\beta$ (ie, $b_n / \tau_n^2$), donne une valeur très faible (de l'ordre de $10^{-2}$. Ce qui rend notre approximation legitime vu que la loi a posteriori de $\beta$ tendrait vers une dirac. 
On se propose à présent de caractériser la loi prédictive a posteriori $\mathbb{E}(Y_{new} | y_{1:n})$ sachant les valeurs de $x_{new}$. 
$$
\begin{align}
p(Y_{new} | x_{new}, y_{1:n}) &=^{(a)}\int_{\theta} \int_{\alpha} \int_{\beta} p(Y_{new} | x_{new}, \theta, \alpha, \beta)p(\theta, \alpha, \beta,  | y_{1:n})d\theta d\alpha d\beta\\
&=^{(b)} \int_{\theta} \int_{\alpha} \int_{\beta} p(Y_{new} | x_{new}, \theta, \alpha, \beta)q_1^*(\theta)q_2^*(\alpha)q_3^*(\beta)d\theta d\alpha d\beta\\
&=^{(c)} \int_{\theta} \int_{\alpha}p(Y_{new} | x_{new}, \theta, \alpha, \mathbb{E}_{q_3^*}(\beta))q_1^*(\theta)q_2^*(\alpha)d\theta d\alpha\\
&=^{(d)}\int_{\theta} \mathcal{N}(Y_{new}; \phi \theta, (\frac{b_n}{\tau_n})^{-1} I_n) \mathcal{N}(\theta; \tilde{m_n}, \tilde{S_n}) d\theta \int_{\alpha} q_2^*(\alpha)d\alpha\\
&=\int_{\theta} \mathcal{N}(Y_{new}; \phi \theta, (\frac{b_n}{\tau_n})^{-1} I_n) \mathcal{N}(\theta; \tilde{m_n}, \tilde{S_n}) d\theta\\
&= \mathcal{N}(Y_{new}; \phi(x_{new}) \tilde{m_n}, \sigma^2(x_{new}))\\
avec\ \ \sigma^2(x) &= \frac{\tau_n}{b_n} + \phi(x)^T \tilde{S_n} \phi(x)
\end{align}
$$
On peut maintenant tracer la loi de $\mathbb{E}(Y_{new} | y_{1:n})$ en fonction des $x_{new}$. Dans le même graph, on fait apparaitre le jeu des données ainsi que les quantiles $0.975$ et $0.025$.
```{r}
Phi_best = matrix(sapply(feature_cars, listF[[best_degre+1]]), 50, best_degre+1, 
                  byrow=TRUE)
temp = variational_lm(Phi_best, target_cars, hyperpar ,maxiter = 100, tol =1e-4)
theta_pred = temp$varpar$m_n

test = 1:25
Phi_test = matrix(sapply(test, listF[[best_degre+1]]), 25, best_degre+1, byrow=TRUE)
Etarget_fit <-  Phi_test%*%theta_pred
# Calcul de la variance
variance_new <- temp$varpar$tau_n/temp$varpar$b_n + Phi_test%*% temp$varpar$S_n %*% t(Phi_test)
# Calcul des quantiles à 0.975 et 0.025
ymax = Etarget_fit + 1.96 * sqrt(diag(variance_new))
ymin = Etarget_fit - 1.96 * sqrt(diag(variance_new))
# Plot graphique des résultats
plot(cars$speed, cars$dist)
points(1:25, Etarget_fit, col='blue')
lines(ymax, col='red')
lines(ymin, col='red')
polygon(c(test, rev(test)), c(ymax, rev(ymin)),
        col=rgb(1, 0, 0,0.5), border = NA)
```
 
On remarque que la plupart des points du jeu de données appartiennent effectivement à l'intervalle de crédibilité. Quoique, cet intervalle de crédibilité peut être encore raffiner surtout pour des valeurs faible de $x_{new}$

## Méthodes MCMC
Dans cette partie, on cherche à approximer la loi a posteriori $p(\theta, \alpha, \beta | y)$ par des méthodes de chaines de Markov. Dans un premier temps, nous allons utilisé l'algorithme de Metropolis-Hastings. Ensuite, nous allons passer à l'échantillonneur de Gibbs. Nous allons à la fin tester les résultats obtenus par chaque algorithme.

### Algorithme Metropolis-Hastings

```{r}
MH_lm <- function(Phi, target, hyperpar, proposalpar = list(sd_theta = 0.05, sd_alpha=0.01, sd_beta = 0.01), startpar = list(theta = rep(0,ncol(Phi)), alpha = 1, beta = 1), Nsim = 10e+2 )
    ## ARGUMENTS:
    ## Phi: design matrix (feature map applied to data): a n*p matrix
    ## target: a  vector of size n: the observed values y.
    ## hyperpar: a list of hyperparameters containing entries a_0, lambda_0, b_0, tau_0:
    ##          hyper-prior gamma parameters respectively for alpha and beta
    
    ##proposalpar : a list containing entries sd_theta, sd_alpha, sd_beta: the standard deviations respectively for the theta proposal (gaussian) and for alpha and beta on the exponential scale (alpha and beta proposal are lognormal)
    ## startpar: the starting value for the cahin: a list with entries theta, alpha, beta.
    ## Nsim: the desired length for the markov chain (number of MCMC interations)
    ##
    ## RETURNS: a list with entries :
  ## statesChain:  a matrix of dimension Nsim* (p+2): row number i  contains the current
  ##               state at time i, which is the concatenation of the vector theta, alpha and beta
  ## naccept: an integer: the number of accepted moves
  ## lastpar: the latest state: a list with entries (theta, alpha, beta)
  {
    n <- nrow(Phi)
    p <- ncol(Phi)
    statesChain <- matrix(nrow=Nsim, ncol = p + 2 )
    currentpar <- startpar
    naccept <- 0
    
    logposterior <- function(par){ ## logarithm of the unnormalizized posterior density
      llkl <- sum( dnorm(target, mean = Phi %*% par$theta , sd = (par$beta)^(-1/2) , log=TRUE ))
      lprior <- sum( dnorm(par$theta, mean = 0, sd = (par$alpha)^(-1/2) ,
                           log = TRUE )) +
        dgamma(par$alpha, shape = hyperpar$a_0, rate = hyperpar$lambda_0, log = TRUE) +
        dgamma(par$beta, shape = hyperpar$b_0, rate = hyperpar$tau_0, log = TRUE)
      return(llkl + lprior)}
    
    gen_proposal <- function(cpar, ppar){
      theta_prop <- t(rmvnorm(1, mean = cpar$theta, sigma = ppar$sd_theta^2*diag(p)))
      alpha_prop <- rlnorm(1,mean = log(cpar$alpha) , sd = ppar$sd_alpha)
      beta_prop <- rlnorm(1,mean = log(cpar$beta) , sd = ppar$sd_beta)
      return(list(theta = theta_prop, alpha = alpha_prop, beta = beta_prop))
    }
    
    logproposal <- function(evalpar, par, ppar){
      q1 = dmvnorm(t(evalpar$theta), mean = par$theta, sigma=ppar$sd_theta^2*diag(p), log = TRUE)
      q2 = dlnorm(evalpar$alpha, mean = log(par$alpha), sd = ppar$sd_alpha ,log = TRUE)
      q3 = dlnorm(evalpar$beta, mean = log(par$beta), sd = ppar$sd_beta, log=TRUE)
      return(q1 + q2 + q3)
    }
    
    for(i in (1:Nsim)){
      candidat_par = gen_proposal(currentpar, proposalpar)
      quotient = exp(logposterior(candidat_par)+logproposal(candidat_par, currentpar, proposalpar) - logposterior(currentpar) - logproposal(currentpar, candidat_par, proposalpar))
      
      u = runif(1)
      if (u <= min (1, quotient))
      {
        currentpar <- candidat_par
        naccept <- naccept+1
      }
      else { }
      statesChain[i, ] <- c(currentpar$theta, currentpar$alpha, currentpar$beta)
    }
    return(list(statesChain = statesChain, naccept = naccept, lastpar = currentpar ))
  }
```
s
```{r}
MHchain <- MH_lm(Phi = Phi_4, target = target_4, hyperpar = hyperpar,
                         proposalpar = list(sd_theta = 0.05, sd_alpha=0.01, sd_beta = 0.01),
                         startpar = list(theta = rep(0, ncol(Phi_4)) , alpha = 1, beta = 1),
                         Nsim = 10e+3)

plot(mcmc(MHchain$statesChain))
```
```{r}
 hh <- heidel.diag(MHchain$statesChain, pvalue = 0.05)
 hh
```

```{r}
 # Some failed to converge in stationary test, we increase Nsim
 MHchain <- MH_lm(Phi = Phi_4, target = target_4, hyperpar = hyperpar,
                  proposalpar = list(sd_theta = 0.1, sd_alpha=0.05, sd_beta = 0.05),
                  startpar = list(theta = rep(0, ncol(Phi_4)) , alpha = 1, beta = 1),
                  Nsim = 30e+3)
 hh <- heidel.diag(MHchain$statesChain, pvalue = 0.05)
 hh
```

```{r}
 ## run chains in parallel: 
 nchains = 3
 chainlist = list(MHchain$statesChain, MHchain$statesChain, MHchain$statesChain)
 for(k in 1:nchains){
   Startpar <-  list(theta = rep(-5 + k * 10/nchains, ncol(Phi_4)),
                     alpha = 0.01 + k * 5/nchains,
                     beta = 0.01 + k * 5/nchains)
   MHchain <- MH_lm(Phi = Phi_4, target = target_4, hyperpar = hyperpar,
                    proposalpar = list(sd_theta = 0.1, sd_alpha=0.05, sd_beta = 0.05),
                    startpar = Startpar,
                    Nsim = 30e+3) 
   chainlist[[k]] <- mcmc(MHchain$statesChain)
 }
 
 chainlist <- mcmc.list(chainlist)
 
 plot(chainlist)
```

```{r} 
 gg <- gelman.diag(chainlist)
 gg
```

```{r}
 gp <- gelman.plot(chainlist)
```

```{r}
 hh <- heidel.diag(chainlist)
 hh
```
On choisit ici le modèle ayant accepté le plus de tests, on définira les valeurs des paramètres retrouvés par ce modèle par la moyenne des derniers paramètres dans la liste (c'est à dire qu'on est sûr d'être au régime stationnaire).
```{r}
chosen_chain = chainlist[[3]]
theta_est1 = mean(tail(chosen_chain[,1], 1000))
theta_est2 = mean(tail(chosen_chain[,2], 1000))
theta_est3 = mean(tail(chosen_chain[,3], 1000))
theta_est4 = mean(tail(chosen_chain[,4], 1000))
theta_est5 = mean(tail(chosen_chain[,5], 1000))
theta_est = c(theta_est1, theta_est2, theta_est3, theta_est4, theta_est5)

alpha_est = mean(tail(chosen_chain[,6], 1000))
beta_est = mean(tail(chosen_chain[,5], 1000))
```

On définit la variance du paramètre $\theta$ par la différence entre la plus grande valeur et la plus petite valeur des valeurs considérées. 

```{r}
variance_est = sapply(1:5, function(i){
  max(tail(chosen_chain[,i], 1000)) - min(tail(chosen_chain[,i], 1000))
})
```

On s'intéresse ici à comparer les valeurs de chaque composante de $\theta$ trouvées par la méthode variationnelle et par la méthode de Metropolis Hastings

```{r}
Iplus <- theta_est + 1.96*sqrt(variance_est)
Iminus <- theta_est - 1.96*sqrt(variance_est)

plot(L$varpar$m_n, pch=19,col='black',main = paste(" N = ", toString(N), sep=""), ylim = 
       range(Iminus, Iplus, theta0))
points(theta_est, col='red')

arrows( x0 = 1:5, y0 = Iminus, 
        y1 = Iplus ,code=3, length=0.1, col="red")
```

```{r}
# Les valeurs des paramètres theta par la méthode variationnelle et la méthode de Metropolis Hasting sont pratiquement les mêmes
```
### Algorithme de Gibbs

Dans cette partie, les valeurs des loi conditionnelles seraient prises comme on l'a fait dans la première partie pour toutes les $q_i$. La seule différence est qu'ici on ne considère pas l'espérence mais plutôt les valeurs réelles des paramètres. 

```{r}
 ##' Gibbs sampler
 gibbs_lm <- function(Phi, target, hyperpar,
                      startpar = list(theta = rep(0,ncol(Phi)), alpha = 1, beta = 1),
                      Nsim = 10e+3 )
   ## ARGUMENTS: 
   ## Phi: design matrix (feature map applied to data): a n*p matrix
   ## target: a  vector of size n: the observed values y.
   ## hyperpar: a list of hyperparameters containing entries a_0, lambda_0, b_0, tau_0:
   ##          hyper-prior gamma parameters respectively for alpha and beta
   ## startpar: the starting value for the chain: a list with entries theta, alpha, beta.
   ## Nsim: the desired length for the markov chain (number of MCMC interations)
   ##
   ## RETURNS: a list with entries :
   ## statesChain:  a matrix of dimension Nsim* (p+2): each row contains the current
   ##               state (concatenation of the vector theta, alpha, beta)
 ## lastpar: the latest state: a list with entries (theta, alpha, beta)
 ##
 {
   n <- nrow(Phi)
   p <- ncol(Phi)
   statesChain <- matrix(nrow=Nsim, ncol = p +2 )
   currentpar <- startpar
   an <-  hyperpar$a_0 + p/2
   bn <-  hyperpar$b_0 + n/2
   
   S_n <- function(alpha, beta){
     S_n_inv <- alpha * diag(p) + beta * t(Phi)%*%Phi
     return(solve(S_n_inv))
   }
   
   m_n <- function(alpha, beta){
     return(beta * S_n(alpha, beta) %*% t(Phi) %*% target)
   }
   
   lambda_n <- function(theta){
     return(hyperpar$lambda_0 + 0.5*t(theta)%*%theta)
   }
   
   tau_n <- function(theta){
     return(hyperpar$tau_0 + 0.5 * t(target - Phi%*%theta)%*%(target-Phi%*%theta))
   }
   
   simulate <-function(cpar){
    theta_sim = t(rmvnorm(1, mean = m_n(cpar$alpha, cpar$beta), sigma = S_n(cpar$alpha, cpar$beta)))
    alpha_sim = rgamma(1, shape = an, rate = lambda_n(theta_sim)) 
    theta_sim1 = t(rmvnorm(1, mean = m_n(alpha_sim, cpar$beta), sigma = S_n(alpha_sim, cpar$beta)))
    beta_sim = rgamma(1, shape = bn, rate = tau_n(theta_sim1))
    return(list(theta = theta_sim, alpha = alpha_sim, beta = beta_sim))
   }
   
   
   for (i in 1:Nsim){
     currentpar = simulate(currentpar)
     
     statesChain[i,] <- c(currentpar$theta, currentpar$alpha, currentpar$beta)
     
   }
   return(list( statesChain = statesChain, lastpar = currentpar ))
 }
```

```{r}
 gibbschain <- gibbs_lm(Phi = Phi_4, target = target_4, hyperpar = hyperpar,
                  startpar = list(theta = rep(0, ncol(Phi_4)) , alpha = 1, beta = 1),
                  Nsim = 30e+3)
 
 plot(mcmc(gibbschain$statesChain))
```
```{r}
 hh <- heidel.diag(gibbschain$statesChain, pvalue = 0.05)
 hh
```
On remarque que le diagnostic de heidelberg pour l'échantillonneur de Gibbs donne des tests positifs pour toutes les valeurs des paramètres. On peut facilement voir que toutes ces valeurs convergent effectivement vers les valeurs réelles. Dans cet exemple, l'échantillonneur de Gibbs est plus performant.


