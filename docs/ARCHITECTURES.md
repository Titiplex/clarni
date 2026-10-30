# Extensions de Clarni : embeddings, CNN, RNN et Transformer

Ces extensions conservent les API `ml::Tensor`, `ml::Layer`, `ml::NeuralNetwork` et
`ml::ad::Variable`. Les nouveaux modèles vivent dans `ml::nn`, utilisent les
optimiseurs existants et construisent leurs graphes sur **le même moteur autograd**.

## Périmètre et exécution

Le frontend `ml::nd` conserve son implémentation CPU en `double` et dispose
maintenant d'un chemin CUDA résident pour les réseaux denses modulaires et les
CNN (Conv1D/Conv2D, pooling, activations et pertes). La liste exacte des opérations
portées, les transferts et les limites sont décrits dans
[GPU_MNIST.md](GPU_MNIST.md). Les séquences récurrentes, embeddings et Transformer
restent CPU dans cette version. Le backend CUDA historique demeure disponible.
Aucune dépendance à une bibliothèque de réseaux de neurones n'est ajoutée.

Les nouveaux modèles et leurs tests ciblent des petits problèmes, l'apprentissage
et la validation mathématique. La convolution directe et l'attention quadratique
ne sont pas conçues pour entraîner de grands modèles. Aucun modèle préentraîné
ni corpus externe n'est inclus.

## Formes et types

| Objet                   | Convention                                                              |
|-------------------------|-------------------------------------------------------------------------|
| `nd::Array`             | Stockage contigu row-major, forme N-D, dimensions strictement positives |
| Scalaire                | Forme `{}`, un élément                                                  |
| `Linear`                | `[..., input_features]` → `[..., output_features]`                      |
| `TokenBatch`            | Identifiants entiers avec une forme explicite                           |
| `Embedding`             | Identifiants `[...]` → `[..., embedding_dim]`                           |
| `Conv1D`                | `[batch, channels, length]`                                             |
| `Conv2D`                | `[batch, channels, height, width]`                                      |
| RNN, GRU, LSTM          | `[batch, time, input_features]` → `[batch, time, hidden_features]`      |
| Attention / Transformer | `[batch, time, features]`                                               |
| Logits de langage       | `[batch, time, vocabulary_size]`                                        |

`nd::Array` ajoute des métadonnées de forme à une `ad::Variable` dont le stockage
est un `Tensor` plat. Il ne s'agit pas d'une réécriture du stockage matriciel.
`reshape` partage le stockage et le nœud ; `permute`, `slice`, `concat` et les
opérations d'indexation créent actuellement leurs sorties contiguës.
Le nombre d'éléments est limité à `INT_MAX` par l'indexation du stockage actuel.
Les tenseurs vides et les vues à strides arbitraires ne sont pas pris en charge.

Les opérations disponibles incluent `reshape`, `permute`, `slice`, `concat`,
`broadcast_to`, `add`, `multiply`, `scale`, `activate`, `sum`, `mean`,
`reduce_sum`, `matmul`, `softmax`, `mse` et `cross_entropy`. Le broadcasting
aligne les dimensions à droite. `matmul` prend deux tableaux de rang 2 ;
les modules gèrent explicitement les dimensions supplémentaires.

## Modules, paramètres et entraînement

```cpp
#include "module.h"
using namespace ml;

nn::Sequential model;
model.add(std::make_shared<nn::Linear>(2, 8));
model.add(std::make_shared<nn::ActivationLayer>(Activation::Tanh));
model.add(std::make_shared<nn::Linear>(8, 2));

OptimizerOptions options;
options.kind = OptimizerKind::Adam;
nn::Trainer trainer(model, options);
auto x = nd::Array::from({4, 2}, {0,0, 0,1, 1,0, 1,1});
std::vector<int> labels{0, 1, 1, 0};
for (int epoch = 0; epoch < 300; ++epoch) {
    trainer.zero_grad();                    // avant forward
    auto logits = model.forward(x);
    auto loss = nd::cross_entropy(logits, labels);
    loss.backward();
    trainer.step(0.03, 1.0);                // norme globale maximale facultative
}
```

`Parameter` contient une feuille entraînable. `assign` conserve sa forme et
remplace sa feuille, sans conserver le graphe précédent. Les valeurs doivent
être finies. Les paramètres sont identifiés par leur objet, et non par une
copie de leur valeur. `nn::parameters(model)` déduplique les objets partagés ;
Adam reçoit donc un seul état par paramètre partagé.

Un `Trainer` est lié à une topologie et à des identités de paramètres. Une
modification de celles-ci impose de reconstruire le trainer. Les modules doivent
former une composition sans cycle. Un module peut être partagé à plusieurs
positions dans cette composition.

`backward()` réinitialise les gradients des nœuds visités à chaque appel,
conformément au comportement historique de `ad::Variable`. Pour agréger plusieurs
pertes, les additionner puis appeler **une seule fois** `backward`.
Ne pas utiliser deux appels successifs comme mécanisme d'accumulation de lots.
`zero_grad()` doit précéder la construction du nouveau graphe. Ne pas appliquer
un optimiseur avant le backward de toutes les branches partageant ses paramètres.

`train()` et `eval()` se propagent aux sous-modules. `eval()` **ne désactive pas**
autograd : il n'existe pas encore de contexte global `no_grad`. Les sorties et
les états dont on n'a plus besoin doivent être libérés ou détachés.

Un module personnel expose `named_parameters()` et définit `forward(Array)` ou
une interface typée telle que `logits(TokenBatch)`. Les noms des sous-modules
sont préfixés pour fournir des noms uniques et stables.

## Embeddings et vocabulaire

```cpp
#include "embedding.h"
nn::Vocabulary vocabulary({"chat", "chien", "chat"});
nn::Embedding embedding(vocabulary.size(), 16, nn::Vocabulary::pad_id);
auto ids = vocabulary.encode({"chat", "chien", "inconnu"});
auto vectors = embedding.lookup({ids, {1, 3}}); // [1,3,16]
```

`Vocabulary` réserve `<pad>` = 0 et `<unk>` = 1, filtre selon `min_count`, puis
ordonne les autres tokens lexicalement pour rendre les identifiants déterministes.
Il peut être sauvegardé/rechargé indépendamment du modèle. `whitespace_tokens`
ne fait qu'un découpage sur les espaces. La normalisation du texte et la
tokenisation Unicode/IPA restent à fournir par l'appelant : un octet UTF-8
n'est pas un phonème.

La table `Embedding` utilise des gradients denses et un scatter-add : les
occurrences répétées d'un identifiant cumulent leurs contributions. La ligne
`padding_idx`, lorsqu'elle est définie, est initialisée à zéro et ne reçoit pas
de gradient. Les tests numériques excluent cette ligne, dont le gradient est
volontairement contraint. Pour une garantie de gel, ne pas lui injecter un ancien
état d'optimiseur ayant des moments non nuls pour cette ligne.

`SkipGram` apprend deux tables par prédiction du contexte avec un **softmax complet**.
Il n'implémente ni negative sampling, ni sous-échantillonnage des mots fréquents.
`skipgram_pairs` prend une phrase : appeler cette fonction séparément sur chaque
phrase réelle pour ne pas créer de contextes artificiels entre phrases.
Le softmax complet convient à une démonstration et à de petits vocabulaires.
La structure sémantique dépend du corpus et de l'objectif, pas de la couche seule.

`examples/embedding_cnn.cpp` combine réellement une table entraînable, une
permutation `[B,T,D]` → `[B,D,T]`, `Conv1D`, un pooling et un classifieur.
Il utilise des séquences de longueur fixe ; ce n'est pas un classifieur IPA prêt
à l'emploi. Pour des longueurs variables, il faut aussi masquer les fenêtres
convolutionnelles invalides avant le pooling. Une ligne de padding nulle dans
l'embedding ne suffit pas, à cause des biais et des filtres.

## Convolutions et pooling

`Conv1D` et `Conv2D` calculent une corrélation croisée avec biais, stride, padding
nul et dilation. Les kernels 2D peuvent être rectangulaires. La taille de sortie
sur chaque axe spatial est :

```
floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
```

Les gradients couvrent les entrées, les filtres et les biais. L'implémentation
CPU parcourt directement les connexions sans réserver une grande matrice im2col.
Les convolutions groupées, transposées, le padding réfléchissant et les kernels
CUDA spécialisés ne sont pas inclus.

`pool` et `Pool1D`/`Pool2D` proposent un maximum ou une moyenne, sans padding.
Le backward cumule les contributions de fenêtres qui se chevauchent. En cas
d'égalité au maximum, la première position reçoit le gradient.
`GlobalMaxPool1D` réduit toute la longueur et renvoie `[B,C]`.

## RNN, GRU, LSTM et BPTT

```cpp
#include "gated_recurrent.h"
nn::LSTM recurrent(16, 32);
auto result = recurrent.run(vectors, {3}); // vectors doit ici avoir 16 features
// result.sequence [1,3,32], result.hidden [1,32], result.cell [1,32]
nn::LSTMState next{result.hidden.detach(), result.cell.detach()};
```

Les états initiaux sont facultatifs et explicitement renvoyés. `lengths[b]`
peut être nul. Après cette longueur, l'état est conservé et les sorties de
séquence valent zéro. Les pertes doivent aussi ignorer les cibles de padding.
`cross_entropy` ignore par défaut les étiquettes `-1`, puis moyenne sur les
positions valides ; un lot entièrement ignoré produit une erreur explicite.

`RNNCell` utilise tanh. `GRUCell` utilise l'ordre de portes reset/update/new
et la convention **reset-after**, précisée dans le header. `LSTMCell` utilise
input/forget/candidate/output, avec biais de la porte d'oubli initialisé à 1.
Ce sont des couches unidirectionnelles simples, sans dropout ni projection.

`run(..., detach_every)` détache les états avant chaque nouvelle tranche de
cette longueur. Il coupe les dépendances temporelles entre tranches, tout en
conservant les pertes et les gradients internes à chaque tranche. Pour un
entraînement réellement borné en mémoire, traiter aussi l'entrée par morceaux,
faire backward/step sur chaque morceau et détacher l'état retourné avant le
suivant. Un seul `run` sur une séquence entière conserve toujours ses sorties.

La BPTT utilise le graphe partagé, et non les caches uniques de l'ancienne
`Layer`. Le clipping de `Trainer::step(rate, max_norm)` porte sur la norme L2
globale des gradients atteints. Il ne résout pas les gradients évanescents.
Le moteur autograd parcourt actuellement le graphe récursivement : les graphes
extrêmement profonds restent à éviter.

## Attention et Transformer

`scaled_dot_product_attention` calcule `softmax(Q*K^T/sqrt(d))*V` sur des
matrices de rang 2. Son masque booléen signifie **true = autorisé**.
Une ligne entièrement masquée renvoie zéro et transmet un gradient nul.

`MultiHeadSelfAttention` ajoute les projections entraînables Q/K/V/sortie,
le découpage en têtes, le masque causal et les longueurs de séquence.
Le nombre de têtes doit diviser la dimension des features. Les requêtes de
padding renvoient zéro, y compris après le biais de projection.

`LayerNorm` normalise le dernier axe avec variance divisée par le nombre de
features, epsilon positif et paramètres affine gamma/beta.
`SinusoidalPosition` ajoute des positions sinusoïdales, avec offset facultatif.
`TransformerBlock` est un bloc pre-norm avec deux connexions résiduelles et
un feedforward ReLU. `TransformerLM` associe embeddings, positions, un bloc,
normalisation finale et projection vers le vocabulaire.

Ce petit modèle peut apprendre une séquence et générer la suite. Il n'inclut
pas de dropout, de KV cache, de tokenizer BPE, d'attention optimisée, de chargement
de poids externes ou d'entraînement distribué. Les logits de padding doivent
être ignorés dans la perte, même si le bloc masque ses sorties.

## Checkpoints

```cpp
nn::save_checkpoint("model.nn", model, &trainer);
// Reconstruire exactement la même topologie et les mêmes options d'optimiseur.
nn::load_checkpoint("model.nn", model, &trainer);
```

Le format textuel versionné conserve les noms, formes, valeurs des paramètres,
et facultativement les moments/compteurs/options de l'optimiseur. Un chargement
valide entièrement le fichier avant de remplacer les paramètres. Les modèles
et trainers distincts sont refusés. Charger un fichier sans état d'optimiseur
avec un trainer remet son état à zéro.

La topologie et ses hyperparamètres ne sont pas reconstruits à partir du fichier :
les noms et formes sont vérifiés, mais ils ne détectent pas toutes les différences
possibles de configuration. Le vocabulaire est sauvegardé séparément. Le format
n'est pas compatible avec les checkpoints de `NeuralNetwork`, qui restent pris
en charge par l'ancienne API. Les sorties/caches/graphes, états récurrents externes,
mode train/eval et l'état du chargeur de données ne sont pas sauvegardés.
Pour une reprise complète, conserver également ces informations côté application.

## Compilation et exemples

```powershell
cmake -S . -B build-cpu -DML_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu --config Release --parallel
ctest --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\examples\Release\clarni_example_modular_mlp.exe
.\build-cpu\examples\Release\clarni_example_word_embeddings.exe
.\build-cpu\examples\Release\clarni_example_embedding_cnn.exe
.\build-cpu\examples\Release\clarni_example_cnn.exe
.\build-cpu\examples\Release\clarni_example_character_rnn.exe
.\build-cpu\examples\Release\clarni_example_recurrent_comparison.exe
.\build-cpu\examples\Release\clarni_example_tiny_transformer.exe
```

Les chemins `Release` correspondent à un générateur multi-configuration comme
Visual Studio. Avec Ninja/Makefiles, ajouter `-DCMAKE_BUILD_TYPE=Release` à la
configuration et lancer les exécutables directement dans `build-cpu/examples`.
Le projet conserve son minimum CMake 4.2 et C++23.

Sans CMake, avec Python et GCC déjà installés :

```
python3 tools/verify_cpu.py --build build-direct --examples
```

Python ne participe pas à l'exécution des modèles : ce script pilote seulement
la compilation et les tests. Les données des exemples sont synthétiques et
petites. Une baisse de perte valide une chaîne d'entraînement, pas la capacité
à généraliser ni l'intérêt linguistique d'un modèle.
