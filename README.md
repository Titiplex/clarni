# Clarni

Bibliothèque pédagogique C++23 : réseaux denses, mini-batches matriciels,
régression et classification, SGD/Momentum/Adam, validation et checkpoints,
différentiation automatique matricielle, et tenseurs CPU ou CUDA résidents.
Les kernels CUDA sont compilés en C++17.

## Construire

```sh
cmake -S . -B build-cpu -DML_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DML_BUILD_EXAMPLES=ON
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

Sous Windows/CLion, utiliser MSVC amd64 pour CUDA :

```sh
cmake -S . -B build-cuda -DML_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DML_BUILD_BENCHMARKS=ON
cmake --build build-cuda --config Release
ctest --test-dir build-cuda -C Release --output-on-failure
```

89 reprend le projet initial : adapter aux GPU cibles. Le kernel de réduction
FP64 demande une architecture CUDA au moins 6.0. Le toolkit doit aussi prendre
cette architecture en charge. Les deux tests matériels sont ignorés sans GPU.

Pour comparer avec cuBLAS, configurer un autre dossier avec `-DML_USE_CUBLAS=ON`
et `-DML_ENABLE_CUDA=ON`. Cette option ajoute une dépendance runtime cuBLAS ;
sous Windows les DLL correspondantes doivent être disponibles. Elle est OFF
par défaut. La runtime CUDA de base est liée statiquement.

## API dense conservée

```cpp
ml::initialize_compute(ml::Backend::Auto);
ml::NeuralNetwork net(2, 0.02, ml::NeuralNetwork::MINI_BATCH,
                      ml::NeuralNetwork::CROSS_ENTROPY, 42);
net.add_layer(2, 8, ml::Layer::TANH);
net.add_layer(8, 3, ml::Layer::SOFTMAX);
ml::OptimizerOptions optimizer;
optimizer.kind = ml::OptimizerKind::Adam;
net.set_optimizer(optimizer);
net.learn(dataset, 300, 32, true);
```

Les anciens appels avec `SIMPLE`, `SIGMOID`, `Layer::forward/backward/step` et
`Matrix * Vector` restent disponibles. `SIMPLE` conserve la demi-somme des
carrés par exemple. Les pertes BCE et CE exigent respectivement une sortie
SIGMOID et SOFTMAX et calculent leur gradient directement sur les logits.

## Objets et conventions

| Objet           | Contrat                                                                            |
|-----------------|------------------------------------------------------------------------------------|
| Matrix / Vector | Conteneurs hôtes historiques ; Vector conserve le stockage 1 × dim                 |
| Tensor          | Valeur matricielle immuable CPU ou CUDA, copies partageant un stockage non modifié |
| Layer           | Poids M × (N+1), biais en colonne 0 ; initialisation automatique via add_layer     |
| NeuralNetwork   | Mini-batches en colonnes, backward analytique dense, optimiseur et RNG persistants |
| StandardScaler  | Statistiques ajustées uniquement sur l'entraînement, sauvegarde séparée            |
| ad::Variable    | Graphe reverse-mode distinct du réseau dense, sur des Tensor de rang 2             |

Pour un batch : X a N lignes et B colonnes. Le forward calcule W*[1;X].
Le backward accumule les gradients non moyennés ; la division par B a lieu
une seule fois avant l'optimiseur. La pénalité L2 est ajoutée après cette
moyenne et exclut la colonne des biais.

AUTO choisit min(32,N) si batch_size vaut 0 ; MINI_BATCH demande une taille
explicite positive. ONLINE et BATCH choisissent respectivement 1 et N.
Le shuffle repart d'indices canoniques à chaque époque, pour reprendre un
checkpoint avec le seul état du RNG. La séquence reste dépendante de la
bibliothèque standard utilisée : pas de promesse bit à bit inter-plateformes.

## Généralisation et sauvegarde

Utiliser `split_dataset`, puis `StandardScaler::fit` sur l'entraînement et
`transform` sur les trois ensembles. Un split avec shuffle=false conserve
l'ordre temporel. Les fractions sont arrondies vers le bas ; vérifier qu'un
petit dataset produit une validation non vide avant fit.

`fit(training, validation, options)` mesure les pertes non régularisées et
restaure le meilleur état complet : poids, optimiseur, RNG. `min_delta` pilote
la patience sans empêcher la conservation du véritable minimum observé.
`best_epoch=0` signifie que l'état initial est le meilleur. `objective_loss`
ajoute la pénalité L2 pour inspecter l'objectif régularisé.

`net.save(path)` et `NeuralNetwork::load(path)` conservent architecture, perte,
hyperparamètres, poids/biais, moments et compteurs de l'optimiseur, RNG et L2.
Les checkpoints sont destinés aux frontières d'époques ; caches et gradients
d'un batch partiellement calculé ne sont pas sauvegardés. Utiliser les setters
contrôlés pour modifier des poids ; remplacer arbitrairement un Layer en cours
d'entraînement n'est pas une opération de gestion d'état supportée.

Le scaler se sauvegarde et se recharge séparément. L'exemple pipeline montre
la paire modèle + prétraitement. Les formats textuels sont versionnés et les
dimensions/valeurs sont contrôlées à la lecture.

## Autodiff

`ml::ad` fournit add, affine, matmul, hadamard, transpose, broadcast_columns,
sum, mean, square, activate et loss. Les branches partagées additionnent leurs
contributions. Chaque backward remet à zéro les gradients des nœuds accessibles
avant le nouveau calcul ; ce comportement diffère de Layer::backward qui
accumule jusqu'à zero_grad. Un nouveau graphe est créé après une mise à jour
pour refléter les nouvelles valeurs immuables des paramètres.

Ce moteur est volontairement matriciel : pas de tenseurs ND, de mutation
in-place, de gradients des cibles, ni de remplacement automatique du backward
dense de NeuralNetwork. Il constitue un socle autonome pour étendre le graphe.

## Résidence CUDA

Initialiser le backend avant les objets et les threads ; ne pas le reconfigurer
pendant des calculs. Pour l'API Tensor avancée, choisir explicitement le même
backend pour les entrées et paramètres (Backend::Auto suit le backend choisi).

Sur CUDA, les poids, moments d'Adam et intermédiaires Tensor restent en VRAM.
Les batches X/Y sont envoyés depuis le dataset hôte. Les lectures de matrices,
prédictions hôtes et checkpoints déclenchent les copies nécessaires. Les
réductions de contrôle et de loss rapatrient des scalaires, pas les poids.
`synchronize()` attend explicitement la fin des opérations sans lire une matrice.

Les buffers sont libérés par RAII. L'implémentation alloue encore de nombreux
résultats intermédiaires ; ce n'est pas un allocateur à pool ni un moteur de
fusion. Le GEMM CUDA de référence reste simple ; cuBLAS est optionnel. Aucune
accélération n'est garantie. Le chemin historique Matrix continue à transférer
ses données opération par opération ; le réseau utilise maintenant Tensor.

## Exemples et benchmarks

- `clarni_example_matrix` : produit rectangulaire connu.
- `clarni_example_network` : forward dense.
- `clarni_example_xor` : apprentissage XOR.
- `clarni_example_classification` : softmax à trois classes avec Adam.
- `clarni_example_pipeline [modèle.clarni]` : split, scaler, validation,
  arrêt anticipé, test tenu à part et éventuellement sauvegarde/rechargement.
- `clarni_example_autograd` : gradient d'un graphe à branches partagées.

Avec ML_BUILD_BENCHMARKS=ON : `clarni_benchmark_device`. Le temps mesuré inclut
allocations et synchronisation ; le mode recreate/upload inclut les copies
d'entrées. L'échauffement et la lecture du checksum final sont hors mesure.
Comparer des configurations identiques, et ne pas assimiler ce temps à celui
d'un kernel GEMM isolé. Les compteurs enregistrent les buffers Tensor et les
transferts explicites ; les petits flags de validation ne sont pas comptés
comme allocations Tensor.

## Vérification sans CMake

Sur un environnement GCC, `python3 tools/verify_cpu.py` compile et exécute les
tests sans dépendance tierce. Pour ASan/UBSan :

```sh
python3 tools/verify_cpu.py --sanitize --build build-sanitized
```

Dans les environnements qui bloquent LeakSanitizer via /proc, ajouter
`--no-leak-check`. Cela ne constitue pas un test des fuites mémoire.
Le test residency_mock est une simulation du protocole et des transferts ;
il ne valide pas les kernels. Les tests cuda/residency nécessitent du matériel.

Voir `docs/EVOLUTION.md` pour les formules, les choix et les limites de validation.

## Embeddings, CNN, RNN et petit Transformer

Les nouvelles API `ml::nd` et `ml::nn` ajoutent des formes N-D, des modules et
paramètres partagés, les embeddings/SkipGram, Conv1D/Conv2D et le pooling,
RNN/GRU/LSTM avec BPTT et masques, ainsi que l'attention multi-tête et un petit
Transformer de langage. Elles utilisent le moteur autograd et les optimiseurs
existants. Leur implémentation est **CPU de référence** ; les chemins CUDA
historiques restent disponibles pour les réseaux denses existants.

Voir [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md) pour les conventions, exemples,
checkpoints, limites et commandes Windows. Les exemples `modular_mlp`,
`word_embeddings`, `embedding_cnn`, `cnn`, `character_rnn`,
`recurrent_comparison` et `tiny_transformer` exécutent des entraînements complets.

## Pipeline d'entraînement sur données réelles

`ml::data` fournit des datasets, splits, loaders reprenables et collateurs pour
images et séquences. `ml::training::SupervisedTrainer` réunit entraînement,
validation sans graphe, métriques masquées, scheduling et restauration du meilleur
état, y compris Adam et le sampler. L'exemple `mnist` entraîne un CNN depuis les
fichiers IDX ; `sequence_pipeline` illustre le padding avec un GRU.

Voir [docs/TRAINING_PIPELINE.md](docs/TRAINING_PIPELINE.md) pour l'API et les
commandes Windows, et [docs/VALIDATION_TRAINING.md](docs/VALIDATION_TRAINING.md)
pour les contrôles et les résultats réels sur un sous-ensemble MNIST.
