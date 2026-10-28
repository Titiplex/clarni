# Clarni

Réseau dense pédagogique C++23, sigmoïde et perte quadratique, avec backend
CPU ou CUDA pour les operations Matrix. Les kernels CUDA restent en C++17.

## Construire et tester

```sh
cmake -S . -B build-cpu -DML_ENABLE_CUDA=OFF -DBUILD_TESTING=ON -DML_BUILD_EXAMPLES=ON
cmake --build build-cpu --config Debug
ctest --test-dir build-cpu -C Debug --output-on-failure
```

Les cibles ``clarni_example_matrix``, ``clarni_example_network`` et ``clarni_example_xor``
sont trois executables distincts.

Pour CUDA sous Windows, utiliser la toolchain MSVC amd64 puis :

```sh
cmake -S . -B build-cuda -DML_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --config Debug
ctest --test-dir build-cuda -C Debug --output-on-failure
```

## Utilisation

```cpp
ml::initialize_compute(ml::Backend::Auto); // Avant les threads et les calculs.
ml::NeuralNetwork net(2, 0.8, ml::NeuralNetwork::MINI_BATCH,
                      ml::NeuralNetwork::SIMPLE, 42);
net.add_layer(2, 4, ml::Layer::SIGMOID);
net.add_layer(4, 1, ml::Layer::SIGMOID);
const ml::Dataset data(std::vector<ml::Sample>{
    {{0,0},{0}}, {{0,1},{1}}, {{1,0},{1}}, {{1,1},{0}}
});
const auto losses = net.learn(data, 20000, 2, true);
const auto prediction = net.output({0, 1});
```

NeuralNetwork initialise chaque couche en Xavier uniforme, avec biais nuls.
La graine par defaut est 42 ; la fournir en cinquieme argument pour la changer.
Les tirages sont reproductibles avec la meme bibliotheque standard ; la
distribution uniforme n'a pas une sequence bit a bit garantie entre stdlibs.

## Conventions mathematiques

- Matrix(lignes, colonnes), indices commencant a zero, stockage contigu.
- Une couche de N entrées et M neurones utilise Matrix(M, N+1).
- La colonne 0 contient les biais. Le dataset fournit N valeurs, sans le 1.
- forward memorise [1, input...] et les activations ; output est const et ne
  modifie pas les caches.
- backward recoit dL/da, applique la derivee de l'activation element par element,
  accumule le produit exterieur $delta\times [1,input...]$ et renvoie $W_{sans-biais}^T\times delta$.
- SIMPLE = $\frac{1}{2} \sum_k (prediction_k-target_k)^2$ pour un exemple.
  simple() renvoie son gradient prediction-target ; simple_loss() sa valeur.
- mean_loss fait la moyenne sur les exemples, pas sur les sorties.
- step applique la moyenne des gradients du groupe. Les poids restent fixes
  pendant tous les forwards/backwards de ce groupe, puis sont mis a jour.

Vector conserve son stockage 1 x dim pour compatibilite. M*v utilise la
surcharge libre qui l'interprete comme une colonne algebrique : ecrire M*v,
pas M.operator*(v). Vector*Vector n'est pas un produit Hadamard ; backward
emploie donc des boucles explicites pour cette operation.

## Modes d'apprentissage

| Mode       | Taille de groupe                                                    |
|------------|---------------------------------------------------------------------|
| ONLINE     | 1, argument batch_size ignore                                       |
| BATCH      | N exemples, argument batch_size ignore                              |
| MINI_BATCH | Taille explicite strictement positive, bornee a N                   |
| AUTO       | Si batch_size vaut 0, min(32,N) ; sinon taille explicite bornee a N |

Le defaut de learn est batch_size=0. Avec MINI_BATCH, passer donc explicitement
la taille : net.learn(data, epochs, 32). Le dernier groupe est divise par son
nombre reel d'exemples. shuffle=false conserve l'ordre des donnees.

## Inspection et couches seules

get()/operator[] renvoient des references controlees avec at(). set_weights()
verifie la forme M x (N+1) et les valeurs finies, puis invalide cache et gradients.
get_weights() et get_gradients() exposent des references const pour les tests
et l'inspection, sans permettre des modifications sauvages.

Une Layer construite seule garde des poids nuls jusqu'a init_random() ou
set_weights(), ce qui permet des tests numeriques explicites. L'initialisation
aleatoire est automatique pour les couches ajoutees a NeuralNetwork.

## Tests

Les tests fournis couvrent calculs Matrix/Vector, loss et derivee, biais,
gradients par differences finies, cumul des gradients, gardes de l'API,
equivalence des modes, dernier groupe incomplet, reproductibilite et XOR.
Ils n'utilisent pas assert et restent actifs en Release.

Le test dispatch utilise un faux backend uniquement pour verifier la selection
exclusive CPU/CUDA et la propagation d'erreurs, sans lancer de GPU. Le test
cuda, lui, compare reellement des resultats CPU/GPU si du materiel est disponible.

## Limites

La selection de backend est globale : l'initialiser une fois au demarrage,
avant les threads, et ne pas la modifier pendant des calculs concurrents.
Les caches des couches ne sont pas prevus pour un entrainement concurrent.

Les poids restent en RAM. Le backend CUDA copie les entrees et resultats a
chaque operation ; il ne calcule pas en plus le meme produit sur CPU. Les
activations et le backward elementaire restent CPU. Cette version corrige
les calculs ; elle n'annonce pas un entrainement entier resident en VRAM ni
une acceleration mesuree. Le noyau GEMM est volontairement simple.
