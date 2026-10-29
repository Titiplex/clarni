# Entraîner et évaluer Clarni sur des données réelles

Cette série part de `clarni-master(5).zip`, export du commit amont
`405b5438850195c4eaf9a6bb1fb6cbc13d682f8d`. Elle met en œuvre le prochain lot
prioritaire : un pipeline de données et d'entraînement commun, un mode sans
autograd pour l'évaluation, des améliorations ciblées du moteur CPU et une
application MNIST complète. Les anciennes API restent disponibles.

## Périmètre livré

| Élément | Résultat |
|---|---|
| En-têtes autonomes | Includes mathématiques et de tailles explicites ; outil de vérification |
| Autograd | Parcours et destruction itératifs ; gradients alloués à leur première contribution |
| Inférence | `ad::NoGradGuard` local au thread, imbriquable, indépendant de `eval()` |
| Stockage CPU | Adoption d'un vecteur, accès en lecture sans copie, suppression du détour par Matrix dans le frontend N-D |
| Broadcasting | Cas identité sans copie ni nœud supplémentaire |
| Attention | Produits matriciels par lots avec gradients et broadcasting des axes de batch |
| Données | Dataset, Subset, split reproductible, lots typés, padding et longueurs |
| Chargement | Mélange déterministe dans un environnement donné et reprise exacte du sampler |
| Apprentissage | Boucle entraînement/validation commune, métriques masquées et arrêt anticipé |
| Meilleur état | Paramètres, moments et compteurs d'Adam, ordre et RNG du loader d'entraînement |
| Applications | CNN sur fichiers MNIST réels et GRU sur séquences synthétiques de longueurs différentes |
| Mesure | Benchmark forward avec/sans enregistrement du graphe |

Ce lot ne porte pas les opérations N-D sur CUDA, ne change pas le type `double`
en `float32`, et n'ajoute pas de vues à strides arbitraires. Dropout, AdamW,
negative sampling et cache KV restent des développements distincts. Il ne
prétend pas achever toute la feuille de route exploratoire.

## 1. Inférence et autograd

```cpp
#include "module.h"
model.eval();                 // comportement des modules en évaluation
{
    ml::ad::NoGradGuard guard; // pas d'enregistrement des opérations dans le graphe
    auto prediction = model.forward(input);
}
// L'état d'enregistrement antérieur est restauré, même après une exception.
```

Le garde est local au thread, non copiable et non déplaçable. Cela ne rend pas
le backward ou les mises à jour concurrentes d'un même modèle thread-safe. Les gardes imbriqués
restaurent leur état précédent. Il n'altère pas les paramètres et ne remet pas
leurs gradients à zéro. Les feuilles explicitement créées avec `requires_grad`
conservent ce flag ; c'est l'enregistrement des opérations qui est désactivé.
`eval()` seul continue à autoriser autograd, ce qui peut être utile pour analyser
les gradients d'un modèle en évaluation.

Le backward construit maintenant son ordre topologique avec une pile explicite.
La libération des nœuds utilise elle aussi une file de destruction itérative,
sans allocation dans le deleter. Un test crée, différentie et détruit une chaîne
de 40 000 opérations. Le premier gradient reçu est adopté directement, sans
allocation préalable d'un tenseur de zéros pour chaque nœud.

La sémantique historique est conservée : chaque appel à `backward()` réinitialise
les gradients des nœuds visités. Pour agréger des pertes, les additionner puis
appeler `backward()` une seule fois. Cela n'ajoute pas l'accumulation automatique
de gradients entre mini-batchs.

## 2. Stockage et produits matriciels par lots

```cpp
std::vector<double> values{1,2,3,4};
auto tensor = ml::Tensor::from_values(2, 2, std::move(values));
auto view = tensor.reshape(4, 1);
std::span<const double> read_only = view.cpu_values();
```

Sur CPU, `from_values` adopte le stockage fourni. Le span est en lecture seule,
valide tant que le stockage partagé concerné est vivant. Ne pas conserver un
span de paramètre à travers une mise à jour qui peut libérer ce stockage.
L'accès CPU à un tenseur CUDA est refusé ; un transfert doit rester explicite.
Les unités CUDA existantes utilisent leurs headers de backend et restent en
C++17 ; les API applicatives C++ continuent à demander C++23.

`nd::Array::values()` continue à renvoyer une copie possédée. En revanche,
`pack`/`unpack` n'effectuent plus les conversions intermédiaires via Matrix.
`reshape` reste un partage du stockage ; `permute`, `slice`, les vrais
broadcasts et plusieurs kernels de référence matérialisent encore des sorties.
Il s'agit d'une réduction ciblée des copies, pas d'un moteur entièrement sans copie.

`nd::batched_matmul` accepte des formes `[..., M,K]` et `[..., K,N]`, avec
broadcasting des préfixes alignés à droite. Le résultat a pour forme
`[broadcast(...), M,N]`. Le backward cumule les contributions des batches
partageant le même opérande. L'attention multi-tête utilise ce chemin pour les
produits QKᵀ et probabilités×V. Les masques de causalité et de padding, dont les
lignes entièrement masquées, conservent leur comportement.

## 3. Dataset, lots et collateurs

L'interface `ml::data::Dataset` expose `size()` et `get(index)`. Un exemple
contient une entrée et des cibles entières :

```cpp
using namespace ml;
auto dataset = std::make_shared<data::VectorDataset>(
    std::vector<data::Example>{
        {nd::Array::from({2}, {0.1, 0.2}), {0}},
        {nd::Array::from({2}, {0.8, 0.9}), {1}}
    });
data::DataLoader loader(dataset, {2, true, false, 42, "dataset-v1"});
loader.reset_epoch();
while (auto batch = loader.next()) {
    const auto& features = std::get<nd::Array>(batch->input);
    // features [B,2], batch->targets [B]
}
```

Le constructeur crée un loader initialement épuisé. `reset_epoch()` commence
une nouvelle époque et applique le mélange demandé. `next()` renvoie `nullopt`
en fin d'époque. `drop_last=false` conserve le dernier lot incomplet.

| Collateur | Entrées par exemple | Entrée du batch | Cibles |
|---|---|---|---|
| `Stack` | Array de forme fixe `[...]` | Array `[B,...]` | Une classe par exemple |
| `PadTokens` | `vector<int>` de longueur T | `TokenBatch` `[B,Tmax]` | Classe ou séquence |
| `PadFeatures` | Array `[T,...features]` | Array `[B,Tmax,...features]` | Classe ou séquence |

Les identifiants de tokens restent des entiers. Les séquences d'entrée vides
sont refusées. Les features de padding valent zéro ; les tokens utilisent
`padding_id`, zéro par défaut. Pour les cibles séquentielles, le padding vaut
`ignore_index`, -1 par défaut. `Batch::lengths` donne les longueurs avant padding,
`Batch::indices` les positions dans le dataset du loader.

```cpp
auto collator = data::make_collator({
    data::InputLayout::PadTokens,
    data::TargetLayout::Sequence,
    0, -1
});
```

Le collateur vérifie les types, les formes et la correspondance entre entrées
et cibles. Une exception de lecture ou de collation ne consomme pas le batch :
le curseur n'avance qu'après sa construction réussie.

`Subset` applique une liste d'indices à un dataset partagé. `random_split`
produit trois listes disjointes et reproductibles pour une même implémentation
standard et une même graine. Pour du texte, des séries temporelles ou des données
groupées, fournir un découpage par document, chronologique ou par groupe adapté ;
un split aléatoire de fenêtres similaires n'empêche pas les fuites.

Le loader est synchrone. Il ne fournit ni workers de chargement, ni prefetch,
ni mémoire épinglée CUDA. Le dataset sous-jacent doit rester immuable pendant
son utilisation.

## 4. Boucle d'apprentissage commune

```cpp
#include "training.h"
OptimizerOptions optimizer;
optimizer.kind = OptimizerKind::Adam;
training::SupervisedTrainer engine(model, optimizer);
training::Forward forward = [&](const data::Batch& batch) {
    return model.forward(std::get<nd::Array>(batch.input));
};
training::FitOptions fit;
fit.epochs = 30;
fit.classes = 10;
fit.learning_rate = 0.003;
fit.max_grad_norm = 5;
fit.patience = 4;
fit.min_delta = 0.001;
fit.restore_best = true;
auto history = engine.fit(training_loader, validation_loader, forward, fit);
auto metrics = engine.evaluate(test_loader, forward, 10);
```

Les logits ont pour dernier axe le nombre de classes. Les cibles correspondent
à toutes les positions précédentes, donc `[B]` pour la classification et
`[B*T]` pour les séquences. L'interface de forward reste un callback typé :
elle peut appeler `TransformerLM::logits(TokenBatch)`, un embedding + GRU,
ou un réseau qui renvoie un logit par séquence à partir de son état final.

Les loaders de validation et de test doivent utiliser `shuffle=false` et
`drop_last=false`. L'évaluation désactive l'enregistrement du graphe, parcourt
toutes les données puis restaure l'état du loader, y compris après une exception.
Le mode racine du modèle est restauré et propagé aux sous-modules. Si une
application configure délibérément des modes différents dans ses sous-modules,
elle doit réappliquer cette configuration après ces appels.

### Métriques

- `loss` : moyenne de l'entropie croisée sur les cibles valides.
- `accuracy` : proportion correcte sur ces mêmes cibles ; premier maximum en cas d'égalité.
- `macro_f1` : moyenne des F1 sur **toutes les classes déclarées** ; une classe absente contribue zéro.
- `perplexity` : `exp(loss)`, utile pour la prédiction de tokens ; peut valoir +inf si l'exponentielle déborde.
- `count` : nombre de cibles valides, qui peut différer du nombre d'exemples.

Une moyenne de batch est pondérée par son nombre de cibles valides. Le dernier
petit batch et les quantités différentes de padding ne faussent donc pas la
moyenne. Un batch sans cible valide est sauté ; un dataset entièrement ignoré
produit une erreur explicite. Les logits et les pertes doivent être finis.
Les statistiques de classification utilisent O(nombre de classes) mémoire,
sans réserver une matrice de confusion carrée pour les grands vocabulaires.

La perte d'entraînement est mesurée sur les forwards successifs avant leurs
mises à jour, pas recalculée sur tout le dataset après l'époque. La validation
utilise l'état du modèle à la fin de l'époque.

### Arrêt anticipé et scheduling

`best_epoch` est indexé à partir de 1. La plus petite perte de validation définit
le meilleur état ; `min_delta` contrôle uniquement le compteur de patience, par
rapport à la dernière amélioration significative. Plusieurs petites améliorations
peuvent donc finir par dépasser ce seuil. `patience=0` désactive l'arrêt anticipé.

Avec `restore_best=true`, paramètres, moments et compteurs de l'optimiseur, ainsi
que l'ordre et le RNG du loader d'entraînement sont remis à l'état de la meilleure
époque. L'historique conserve toutes les époques réellement exécutées.

Deux fabriques de schedules sont fournies :

```cpp
fit.schedule = training::step_schedule(0.003, 5, 0.5);
// Ou :
fit.schedule = training::cosine_schedule(0.003, 0.0001, fit.epochs, 2);
```

Le callback reçoit un index d'époque à partir de zéro. Le cosinus atteint son
minimum si la phase de décroissance contient au moins deux époques ; avec une
seule, il conserve le taux de base. `on_epoch` permet de journaliser les résultats.
Une exception pendant `fit` est propagée ; les mises à jour déjà effectuées ne
sont pas annulées. Un nouvel appel à `fit` recommence son historique, sa patience et l'index de son
schedule. Ce lot ne sérialise pas une session `fit` arbitraire ou ses callbacks.

## 5. Reprise au milieu d'une époque

```cpp
nn::save_checkpoint("model.nn", model, &engine.optimizer());
training_loader.save_state("training.loader");
// Reconstruire le même modèle, le même dataset, collateur et optimiseur.
nn::load_checkpoint("model.nn", model, &engine.optimizer());
training_loader.load_state("training.loader");
while (auto batch = training_loader.next()) {
    // Continuer les mises à jour avec le learning rate approprié.
}
// Ne pas appeler reset_epoch() avant d'avoir fini l'époque restaurée.
```

L'état du loader contient la permutation, le curseur, le numéro d'époque, les
options de batching, le RNG et `dataset_key`. Un état mal formé ou incompatible
est refusé avant mutation. `dataset_key` est fourni par l'application : ce n'est
pas un hash calculé automatiquement. Les données et le comportement du collateur
ne sont pas embarqués dans le fichier et doivent correspondre.

Le test de reprise vérifie les mêmes batches et des poids exactement identiques,
mise à jour après mise à jour, sur le reste de l'époque et l'époque suivante.
Cela suppose un même environnement numérique. Les algorithmes de mélange et de
distribution aléatoire de la bibliothèque standard peuvent différer entre GCC et MSVC.

Les fichiers du modèle et du loader sont indépendants : leur sauvegarde n'est
pas une transaction multi-fichiers. Un checkpoint complet d'application doit aussi
conserver vocabulaire, hyperparamètres, splits, progression du schedule et état
externe éventuel. Les fichiers de configuration et de splits de l'exemple MNIST
aident à reconstruire cette configuration ; ils ne constituent pas un chargeur
automatique de topologie.

## 6. Application MNIST

Le lecteur accepte les quatre fichiers **IDX décompressés**. Il valide les magic
numbers, les tailles, les dimensions, la correspondance images/labels et les labels
0..9. Les pixels restent stockés en octets ; seuls les exemples du batch sont
convertis en doubles et normalisés dans `[0,1]`.

Téléchargement facultatif avec Python 3.9+ (bibliothèque standard uniquement) :

```powershell
python tools/download_mnist.py C:\datasets\mnist
```

Le helper utilise le miroir CVDF, vérifie les checksums MD5 conventionnels des
archives et la taille décompressée, puis installe chaque fichier par renommage
atomique. Les checksums sont des contrôles d'intégrité, pas des signatures.
Un fichier existant différent est refusé sauf `--force`. Le programme C++ lui-même
ne télécharge rien et n'utilise pas Python.

Compilation puis petit essai CPU :

```powershell
cmake -S . -B build-cpu -DML_ENABLE_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-cpu --config Release --parallel
ctest --test-dir build-cpu -C Release --output-on-failure
.\build-cpu\examples\Release\clarni_example_mnist.exe `
  --data C:\datasets\mnist --output runs\mnist-42 `
  --train-limit 2000 --validation 200 --test-limit 1000 `
  --epochs 3 --seed 42
```

Avec Ninja/Makefiles, ajouter `-DCMAKE_BUILD_TYPE=Release` à la configuration et
retirer `Release` des chemins d'exécutables. Le minimum reste CMake 4.2 et C++23.

Pour charger les splits officiels entiers :

```powershell
.\build-cpu\examples\Release\clarni_example_mnist.exe `
  --data C:\datasets\mnist --output runs\mnist-full `
  --train-limit 0 --validation 6000 --test-limit 0 --epochs 10
```

L'architecture est `Conv2D(1,4,3) → ReLU → MaxPool2D(2,2) → Flatten → Linear(676,10)`.
La validation est extraite du split d'entraînement ; le split officiel de test
n'est évalué qu'après la sélection du meilleur modèle. Une limite charge les
premiers N exemples du split concerné avant le split train/validation.

Le dossier de sortie contient :

| Fichier | Contenu |
|---|---|
| `config.txt` | Architecture, backend, dtype, graine et options de l'essai |
| `epochs.csv` | Courbes train/validation et nombres de cibles |
| `training_indices.csv`, `validation_indices.csv` | Positions dans le split d'entraînement chargé |
| `best.nn` | Meilleurs paramètres et état d'Adam |
| `training.loader` | État du loader de cette meilleure époque |
| `metrics.txt` | Meilleure époque et métriques finales validation/test |

Les fichiers connus de ce dossier sont remplacés si l'on réutilise le même
`--output`. Utiliser un dossier différent par expérience.

L'exemple `sequence_pipeline` est une démonstration séparée : embedding + GRU +
projection, séquences synthétiques de longueurs différentes, padding ignoré et
perplexité. Il utilise volontairement les mêmes motifs en entraînement et en
validation ; il ne mesure pas une généralisation linguistique.

## 7. Vérification et mesure

```powershell
python tools/check_headers.py --compiler cl --msvc
```

Cette commande se lance depuis un environnement développeur Visual Studio.
Avec GCC : `python3 tools/check_headers.py --compiler g++`.

Sans CMake, le vérificateur GCC historique reste utilisable :

```
python3 tools/verify_cpu.py --build build-direct --examples
python3 tools/verify_cpu.py --build build-direct --examples --mnist-data /chemin/mnist
```

Sans `--mnist-data`, le programme MNIST est compilé mais son entraînement est
explicitement ignoré. Avec cette option, le script lance un petit essai d'une
époque. Les exemples sans données externes sont exécutés normalement.

Le benchmark s'active avec `-DML_BUILD_BENCHMARKS=ON`, puis se lance via
`clarni_benchmark_pipeline`. Il vérifie l'égalité des prédictions avant de mesurer
le forward avec et sans enregistrement du graphe. Il ne compare pas cette version
à une ancienne version de Clarni et ne garantit pas un facteur d'accélération.

## Sources des données et conventions

- MNIST / miroir CVDF : https://github.com/cvdfoundation/mnist
- Catalogue TensorFlow Datasets : https://www.tensorflow.org/datasets/catalog/mnist
- Distinction eval/autograd : https://docs.pytorch.org/docs/stable/notes/autograd.html

Ces liens documentent les données et les conventions ; aucune dépendance à
TensorFlow ou PyTorch n'est introduite dans Clarni.
