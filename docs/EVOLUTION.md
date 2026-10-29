# Série d'évolution de Clarni

## Base et intention

Base exacte : archive `clarni-master (1).zip`, révision source indiquée par
l'archive `ccff7243d9638c651e3142a6c51d4c5386bcc201`.

La série préserve l'API dense historique tout en ajoutant un calcul de batch
matriciel réutilisable sur CPU et CUDA. Il ne s'agit pas d'un remplacement par
une bibliothèque ML externe. Seul cuBLAS est une option de performance NVIDIA.

Les patchs sont ordonnés. Chaque étape a été compilée et testée sur CPU. Le
backend Tensor CUDA devient disponible au huitième commit ; activer cette
voie seulement après cette étape. Appliquer toute la série pour retrouver
l'état final documenté. Les identifiants Git exacts et les commandes git am
sont fournis dans le guide de livraison extérieur au dépôt.

## 1 — Dérivées et initialisations

Le tanh ajouté dans la base calcule son activation correctement, mais sa
fonction dérivée attend l'activation a et lui applique encore tanh. La règle
correcte est f'(z)=1-a², avec a=tanh(z). Un test différencie numériquement la
sortie par rapport aux entrées pour identité, sigmoïde, tanh, ReLU et Leaky ReLU.

Les poids hors biais suivent une uniforme de variance choisie : Xavier
2/(n_in+n_out), He 2/n_in pour ReLU et 2/((1+alpha²)n_in) pour Leaky ReLU,
avec alpha=0.01. La borne de l'uniforme est sqrt(3*variance). Les biais restent
nuls. Ce changement accompagne tes activations existantes au lieu de les supprimer.

## 2 — Tensor de rang deux et pertes stables

Tensor est une valeur immuable : copier un Tensor partage un stockage qu'aucune
opération publique ne modifie. Les opérations produisent un nouveau résultat.
Cela permet de garder des instantanés cohérents pour l'autodiff et l'arrêt
anticipé. Le rang est fixé à deux : lignes de caractéristiques, colonnes d'exemples.

Les primitives ajoutées sont produit matriciel, transposition, Hadamard,
division, affine scalaire, fonctions élémentaires, réductions, broadcasts,
ajout de la ligne constante de biais et extraction de lignes.

Les pertes retournent un vecteur 1×B de valeurs et une matrice de gradients
individuels non moyennés. La quadratique conserve 1/2 Σ(prediction-cible)².
La BCE est calculée sur les logits z : max(z,0)-y*z+log(1+exp(-abs(z))).
Sa dérivée est sigmoid(z)-y. L'entropie croisée utilise le log-sum-exp stabilisé
par le maximum et son gradient softmax(z)-y pour une cible normalisée.
Pour les cibles normalisées à une tolérance près, le gradient utilise exactement
softmax(z)*Σy-y, cohérent avec la valeur effectivement calculée.

Softmax est couplé sur les lignes de chaque colonne. Son backward calcule
p ⊙ (g - Σ(p⊙g)), pas un vecteur de dérivées diagonales.
Tests : logits ±1000, sommes des probabilités, différences finies et algèbre.

## 3 — Optimiseurs indépendants

L'optimiseur reçoit un gradient déjà moyenné. SGD applique theta-eta*g.
Momentum conserve v=beta*v+g, puis applique theta-eta*v. Adam conserve
m=beta1*m+(1-beta1)*g et v=beta2*v+(1-beta2)*g², corrige les biais avec
1-beta1^t et 1-beta2^t, puis applique theta-eta*mhat/(sqrt(vhat)+epsilon).

Le compteur avance à chaque mise à jour du paramètre, pas à chaque époque.
Chaque couche possède une clé d'état distincte. L'état de la mise à jour n'est
validé qu'après calcul d'un résultat fini. Les moments sont réinitialisés si
les poids sont changés via les mutateurs contrôlés. Des tests vérifient à la
main les premières mises à jour, dont la correction de biais Adam.

## 4 — Vrais mini-batches et classification

Pour B exemples en colonnes :

- X : N×B ; X augmenté : (N+1)×B ; W : M×(N+1).
- Z=W*X_aug ; A=f(Z).
- Delta contient les gradients individuels dL_s/dZ_s.
- Somme des gradients : G=Delta*X_aug^T.
- Gradient vers les entrées : supprimer la première ligne de W^T*Delta.
- Mise à jour : moyenne G/B une seule fois, puis optimiseur.

Layer conserve le chemin Vector comme adaptateur B=1, mais NeuralNetwork ne
boucle plus sur les exemples pour calculer le réseau. Les caches sont des
matrices de batch et conservent Z et A.

SIMPLE accepte toute activation finale. BINARY_CROSS_ENTROPY exige SIGMOID ;
CROSS_ENTROPY exige SOFTMAX et des cibles distributions dont la somme vaut 1.
La perte de classification fournit déjà le gradient par rapport aux logits :
le dernier backward saute explicitement la dérivée de l'activation. Les autres
couches appliquent leur produit jacobien-vecteur normal.

Les caches de paramètres restent sur le backend choisi. Les accès historiques
get_weights/get_gradients matérialisent une matrice hôte sur demande. Leurs
références const restent des vues d'inspection ; utiliser set_weights pour
modifier les paramètres et invalider proprement les états.

Tests : conservation des tests antérieurs, équivalences des modes, dernier
batch incomplet, régression affine, BCE binaire, CE à trois classes et
contrôle par différences finies de leurs gradients de paramètres.

## 5 — Validation, standardisation et L2

split_dataset distribue les exemples sans recouvrement ; shuffle=false permet
un découpage chronologique. Le test final ne sert ni aux gradients ni au choix
des paramètres. StandardScaler.fit reçoit uniquement le jeu d'entraînement.
Il calcule moyenne et variance population avec Welford, puis transforme tous
les jeux avec ces mêmes statistiques. Une caractéristique constante utilise
une échelle de 1, afin d'éviter une division presque nulle.

L'objectif régularisé vaut moyenne(L_s)+lambda/2*||W_sans_biais||². Le gradient
L2 lambda*W_sans_biais est ajouté après la moyenne du gradient de données :
aucune division supplémentaire par B sur lambda. Il s'agit d'une pénalité L2,
pas d'AdamW ni d'un weight decay découplé.

fit mesure les pertes de données training/validation. La patience utilise
min_delta, mais le meilleur état réel est conservé même pour une amélioration
plus petite. On restaure poids, optimiseur et RNG, y compris l'état initial
s'il était le meilleur. Les statistiques du scaler restent explicites et
séparées du réseau, ce qui évite un ajustement caché sur la validation.

## 6 — Checkpoints versionnés

Le format texte de modèle contient dimensions, activations, perte, mode
d'apprentissage, taux, L2, poids/biais, options et moments de l'optimiseur,
compteurs et état mt19937. Les dimensions, valeurs finies et moments non
négatifs sont contrôlés ; le lecteur refuse les versions inconnues et les
données excédentaires. Des limites de taille empêchent des métadonnées
aberrantes de déclencher des allocations démesurées.

Le scaler a un autre format versionné : conserver les deux fichiers pour
reproduire les prédictions prétraitées. La sauvegarde concerne une frontière
d'époque : un batch partiellement accumulé et les caches de forward ne sont
pas un checkpoint d'entraînement.

Le test recharge le modèle, vérifie les prédictions et poursuit l'entraînement
avec Adam et shuffle. La reprise est comparée au même état continu.

## 7 — Différentiation automatique matricielle

Le moteur ml::ad est autonome. Chaque opération mémorise sa valeur et ses
pullbacks vers les parents. Un tri topologique est parcouru en sens inverse.
Les contributions vers un parent partagé sont additionnées.

Pour C=A*B, les règles sont grad_A=grad_C*B^T et grad_B=A^T*grad_C. La
rétropropagation d'un biais diffusé sur B colonnes somme les gradients sur
ces colonnes. Le softmax utilise son produit jacobien-vecteur, et les pertes
fusionnées exposent leur gradient moyen directement.

Les valeurs étant immuables, un graphe ne voit pas les modifications d'un
paramètre futur. Construire un nouveau graphe après une mise à jour. Chaque
backward efface les gradients des nœuds accessibles avant le nouveau calcul ;
les branches partagées s'accumulent à l'intérieur de ce calcul. Le réseau dense
conserve son backward analytique pour sa simplicité ; il n'utilise pas encore
ce graphe comme backend de toutes ses couches.

Tests : branches x²+x, répétition de backward, constantes sans gradient,
produit matriciel, broadcast de biais, activation et différences finies.

## 8 — Tenseurs CUDA résidents

Le stockage Tensor possède soit des valeurs CPU, soit un buffer CUDA RAII.
Les opérations Tensor restent sur leur backend. Les kernels couvrent GEMM,
transpositions, opérations élémentaires, réductions/broadcasts, biais,
activations, produits jacobiens-vecteurs et pertes sur logits.

Le réseau envoie les batches X/Y mais conserve poids, gradients, états de
l'optimiseur et intermédiaires en VRAM. Les contrôles de finitude et les losses
rapatrient des scalaires. Les inspections/checkpoints/prédictions hôtes font
les copies explicitement demandées. Il n'y a pas de recalcul CPU de secours
après un calcul GPU. Une synchronisation ou lecture hôte remonte aussi les
erreurs d'exécution différées.

Un test de protocole simule le backend pour exercer le chemin de stockage,
les durées de vie et les compteurs sans matériel. Pour le réseau testé, le
second appel d'une époque avec deux mini-batches envoie exactement six matrices
X/Y : deux paires d'entraînement et une paire d'évaluation. Les poids et moments
n'ajoutent aucun upload. Ce test ne valide pas les kernels ni la performance GPU.

Le test matériel compare toutes les primitives à une référence CPU, puis les
poids d'un réseau entraîné avec Adam. Il est ignoré en l'absence de GPU.
Le GEMM de référence reste simple et l'allocateur crée de nombreux buffers.
La résidence des données n'est pas une promesse d'accélération.

## 9 — Correction supplémentaire de reprise exacte

Le test initial de checkpoint comparait deux appels poursuivant le même état.
Il a été renforcé pour comparer 17 époques d'un seul appel avec 10 époques,
sauvegarde/rechargement, puis 7 époques. Pour rendre ces trajectoires identiques,
chaque époque mélange une permutation canonique plutôt que la permutation de
l'époque précédente. Le RNG sauvegardé suffit ainsi à reconstruire l'ordre.

Une modification des poids avant save doit aussi invalider les anciens
moments : save écrit un état effectif dans lequel les clés devenues obsolètes
sont réinitialisées, comme elles le seraient au prochain learn. Un test couvre
ce cas. Le lecteur refuse des seconds moments Adam négatifs. La boucle de fit
évite également un dépassement du compteur pour une borne maximale d'époques.

## 10 — Synchronisation, exemples et cuBLAS optionnel

Tensor.synchronize fournit une barrière explicite sans matérialiser une
matrice sur CPU. Le benchmark mesure GEMM avec allocations et synchronisation,
avec entrées résidentes ou recréées/transférées à chaque itération. Les copies
de contrôle final du checksum et l'échauffement sont hors chronométrage.

cuBLAS est optionnel. Comme les données sont row-major et l'API Dgemm est
column-major, on calcule C^T=B^T*A^T en inversant les arguments : les dimensions
cuBLAS sont (colonnes_sortie, lignes_sortie, dimension_interne). Cela évite une
transposition matérielle. Un handle est conservé par thread et par device
courant ; la bibliothèque impose une initialisation de backend stable avant
les threads. Le noyau de référence reste disponible avec l'option OFF.

Les exemples ajoutés couvrent classification softmax, pipeline complet avec
scaler/validation/checkpoint, et un graphe autodiff. Les exemples historiques
et XOR sont conservés. Le script de vérification sans CMake accepte ASan/UBSan.

## 11 — Documentation et mode d'emploi de livraison

Le dernier commit documente les contrats, les changements de comportement,
les formats et les limites. Le guide extérieur au dépôt contient les commandes
git am/git apply, les identifiants réels et le relevé des validations.

## Validation observée dans cet environnement

- GCC 13.3, C++23, compilation directe avec Wall/Wextra/Wpedantic.
- 56 cas de tests passent, sur CPU et avec les simulations explicitement indiquées.
- Les mêmes 56 cas passent sous AddressSanitizer et UBSan. LeakSanitizer est
  désactivé car l'environnement empêche son accès à /proc : pas de résultat
  de recherche de fuites mémoire revendiqué.
- Deux exécutables matériels CUDA sont ignorés, pas comptés comme réussis.
- Six exemples et un benchmark CPU sont compilés et exécutés.
- Pipeline affine : perte sur le jeu de test tenu à part d'environ 2.04e-13 ;
  après rechargement, prédiction 2.5 sur une entrée dont la cible connue vaut 2.5.
- Classification à trois classes : entropie croisée d'environ 0.00301 sur le
  petit jeu pédagogique. Ces exemples ne mesurent pas une performance générale.

CMake, nvcc et un GPU ne sont pas disponibles ici. La configuration CMake,
la compilation CUDA/cuBLAS et les tests matériels restent à exécuter sous
MSVC/CUDA sur ta machine. Les simulations CPU ne remplacent pas ces validations.

Les mesures CPU fournies sont des diagnostics locaux, pas un classement de
performance. Les kernels, cuBLAS, architectures cibles et bibliothèques standard
peuvent produire de légères différences d'arrondi : comparer avec une tolérance.

## Références

- Glorot et Bengio, initialisation et propagation des gradients :
  https://proceedings.mlr.press/v9/glorot10a.html
- He et al., initialisation pour rectifieurs : https://arxiv.org/abs/1502.01852
- Kingma et Ba, Adam : https://arxiv.org/abs/1412.6980
- cuBLAS, stockage et GEMM : https://docs.nvidia.com/cuda/cublas/index.html
- CMake, targets CUDA : https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
