# CUDA résident pour le CNN MNIST

Base : l'archive fournie `clarni-master (1)(1).zip`, export du commit
`33116009228e93fd550663a1a10dd3ee8b6f2d31`.

Cette série porte le chemin du CNN MNIST sur CUDA : stockage, forward,
backward, perte, clipping et optimiseur. Les kernels N-D sont écrits en CUDA C++ ;
aucune dépendance à PyTorch, TensorFlow ou cuDNN n'est ajoutée. CUDA/cuBLAS restent
les dépendances optionnelles déjà prévues par le projet.

## Ce qui change au lancement

`mnist` accepte maintenant :

| Option | Comportement |
|---|---|
| `--backend auto` | Valeur par défaut ; CUDA si le backend est compilé et utilisable, sinon CPU avec diagnostic |
| `--backend cpu` | Force le chemin CPU |
| `--backend cuda` | Exige CUDA ; erreur explicite si compilation, pilote ou GPU indisponible |

`config.txt` enregistre le backend réellement sélectionné, le nom du périphérique,
la présence du backend compilé et l'implémentation de matmul. La valeur CPU n'est
plus codée en dur. Le programme migre les paramètres une fois, puis transfère
chaque batch vers le backend sélectionné. Il vérifie que le backend du modèle
correspond au choix effectué.

## Compilation Windows / CLion

Utiliser un environnement **MSVC** pour CUDA sous Windows : profil MSVC dans
CLion ou Developer PowerShell x64 de Visual Studio. Les commandes suivantes
supposent un toolkit CUDA compatible et `nvcc` accessible à CMake.

```powershell
cmake -S . -B build-gpu -DML_ENABLE_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=ON
cmake --build build-gpu --config Release --parallel
ctest --test-dir build-gpu -C Release --output-on-failure
```

`native` vise le GPU détecté sur la machine de compilation. Pour compiler sans
GPU exposé, fournir explicitement l'architecture cible, par exemple `89` pour
une cible qui la prend en charge. Le binaire doit contenir une architecture
compatible avec le GPU d'exécution. Ne pas réutiliser un répertoire de build
configuré avec un autre compilateur ou une ancienne configuration CUDA.

cuBLAS reste facultatif. Pour utiliser le GEMM cuBLAS déjà présent :

```powershell
cmake -S . -B build-gpu -DML_ENABLE_CUDA=ON `
  -DML_USE_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-gpu --config Release --parallel
```

Sans cuBLAS, le produit matriciel utilise le kernel CUDA de référence existant.
Les nouveaux kernels n'imposent aucune bibliothèque additionnelle.

Les chemins `Release` ci-dessous correspondent aux générateurs Visual Studio.
Avec Ninja/Makefiles, ajouter `-DCMAKE_BUILD_TYPE=Release` à la configuration et
retirer `Release` des chemins.

## Lancer MNIST sur GPU

Réutiliser les quatre fichiers IDX décompressés de MNIST. Téléchargement facultatif
avec le helper existant :

```powershell
python tools/download_mnist.py C:\datasets\mnist
.\build-gpu\examples\Release\clarni_example_mnist.exe `
  --data C:\datasets\mnist --backend cuda --output runs\mnist-cuda `
  --train-limit 2000 --validation 200 --test-limit 1000 `
  --epochs 3 --batch-size 32 --seed 42
```

Le lancement doit annoncer `Selected backend=CUDA` et le fichier de configuration
doit contenir `backend=CUDA` et le nom de la carte. Avec `--backend cuda`, le
programme n'effectue pas de repli silencieux sur CPU.

Pour comparer, lancer la même configuration avec `--backend cpu` et un autre
répertoire de sortie. Le type reste **float64**. Ce portage ne garantit pas une
accélération : ce petit CNN et ses petits batchs peuvent être limités par les
lancements de kernels, les allocations et les synchronisations. Les GPU grand
public n'ont pas tous un débit float64 élevé. Mesurer avant de conclure.

## Stockage et autograd

```cpp
using namespace ml;
auto device = initialize_compute(Backend::Auto);
model.to(device.selected);                  // avant de construire le Trainer
nn::Trainer trainer(model, options);
auto batch_on_device = batch_cpu.to(device.selected);
auto logits = model.forward(batch_on_device);
```

`nd::Array::backend()` expose le backend du stockage. `Array::to` est un transfert
explicite et différentiable : un gradient peut revenir vers le backend de son
entrée. Pour les paramètres, `Module::to` conserve les objets `Parameter` et
remplace leurs feuilles par des valeurs résidentes. Les paramètres partagés sont
migrés une seule fois. Faire la migration avant l'entraînement et ne pas conserver
un ancien graphe à travers une migration du modèle.

Les entrées d'une opération et ses paramètres doivent partager un backend.
Un mélange CPU/CUDA est refusé ; aucun transfert automatique d'une activation
ou d'un poids n'est caché dans un opérateur mathématique.

`Array::values()`, `Array::grad()`, `Tensor::to_matrix()` et les transferts
explicites vers CPU peuvent télécharger leurs données. En revanche,
`Tensor::cpu_values()` refuse toujours une mémoire CUDA. Les exports et checkpoints
sont des frontières explicites où une lecture complète sur CPU est attendue.

## Opérations portées

| Opération | Forward CUDA | Backward CUDA |
|---|---|---|
| Reshape / Flatten | Partage de stockage | Partage / réinterprétation |
| Broadcast, permutation, slice | Kernel d'indexation | Kernel de réduction / indexation inverse |
| Add, multiply, affine | Kernels Tensor existants | Kernels Tensor existants |
| Identity, ReLU, LeakyReLU, tanh, sigmoid | Kernels Tensor existants | Kernels Tensor existants |
| Matmul de rang 2 / Linear | CUDA de référence ou cuBLAS | Produits matriciels CUDA |
| Sum, mean, réduction par axe | Kernels de réduction | Broadcasting CUDA |
| Softmax par axe | Permutation + softmax Tensor CUDA | Softmax backward + permutation inverse |
| MSE | Composition d'opérations résidentes | Autograd résident |
| Cross-entropy avec labels entiers | Log-sum-exp stable CUDA | Gradient CUDA, padding ignoré |
| Conv1D / Conv2D | Corrélation croisée CUDA | Gradients entrée, poids et biais |
| Max / average pooling 1D et 2D | CUDA | CUDA, y compris fenêtres chevauchantes |
| GlobalMaxPool1D | Pooling + reshape | Même chemin |
| Norme et clipping global | Normes par paramètre sur GPU, combinaison de scalaires sur CPU | Facteur appliqué sur GPU |
| SGD, Momentum, Adam | Mises à jour Tensor sur GPU | Moments conservés sur GPU |

L'indexation CUDA couvre les rangs jusqu'à 8 ; les opérations CPU conservent
leur comportement antérieur. Les convolutions prennent stride, padding nul,
dilation et kernels rectangulaires, sans groupes. Le pooling n'a pas de padding.
En cas d'égalité au maximum, la première position est choisie, y compris pour
une fenêtre entièrement à -inf. Les gradients spatiaux sont calculés par sortie
indépendante, sans atomics doubles dans les nouveaux kernels.

Le format des labels reste un vecteur d'entiers sur CPU à la frontière de la
perte. Les identifiants validés sont transférés en un petit tenseur double
(l'intervalle d'indices actuel est représentable exactement). Les lignes ignorées
produisent zéro et la perte est moyennée sur les labels valides. Un batch sans
cible valide est refusé par la perte, comme sur CPU.

## Transferts et synchronisations

Pendant un pas déjà résident, les paramètres, activations, gradients et moments
ne reviennent pas sur CPU. Les shapes sont des descripteurs POD passés aux kernels,
sans construire une table d'indices de la taille des activations à uploader.

Restent sur CPU : lecture du dataset, normalisation des pixels, collation,
labels initiaux, orchestration du graphe, hyperparamètres et compteurs de
l'optimiseur. Des scalaires reviennent du GPU pour la perte, les normes et les
contrôles de finitude. Ces lectures synchronisent et peuvent limiter les petits
workloads. Le portage ne fournit pas encore de pipeline asynchrone ou de pool
d'allocations.

Pour les métriques, l'argmax est calculé sur GPU. Seul un identifiant prédit par
exemple est téléchargé ; les logits complets ne sont pas exportés pour calculer
accuracy/F1. Les cibles et ces identifiants servent aux compteurs sur CPU.

Les compteurs `gpu::resident::statistics()` suivent les allocations Tensor et
les copies explicites de ce backend. Ils n'instrumentent pas toutes les opérations
internes du runtime CUDA, ni le transfert des arguments de lancement. Ils ne
remplacent pas un profiler CUDA.

## Checkpoints et meilleur état

Le format textuel de checkpoint est conservé. Un modèle GPU peut être sauvegardé
puis chargé dans un modèle CPU, ou inversement. Le backend de destination est
celui des paramètres du modèle reconstruit : le fichier ne force pas un backend.
Les moments de l'optimiseur sont transférés vers le backend correspondant lors
de leur restauration. Les paramètres et les moments valides restent résidents
lors de la restauration du meilleur état en fin de `fit`.

Le contrôle de non-négativité du second moment d'Adam utilise maintenant un
prédicat sur GPU, renvoyant un scalaire. Il ne télécharge plus tout le second
moment pour le valider. La sauvegarde sur disque télécharge explicitement les
valeurs nécessaires ; ce coût ne fait pas partie d'un pas d'entraînement résident.

Il faut toujours reconstruire la même topologie, les mêmes options et le même
dataset pour une reprise complète. La progression de `fit`, les callbacks,
vocabulaires et états externes restent soumis aux limites du guide d'entraînement.

## Périmètre encore CPU

Cette série vise un chemin CNN utilisable de bout en bout. `Embedding`, les
wrappers de séquences RNN/GRU/LSTM, LayerNorm, l'attention masquée et Transformer
ne sont pas encore des modèles CUDA pris en charge. `concat` et
`batched_matmul` N-D restent CPU. Ces entrées CUDA sont refusées avant de télécharger
les tenseurs pour un calcul de remplacement sur CPU. Déplacer un module avec
`Module::to` ne rend pas automatiquement tous ses opérateurs compatibles CUDA.

Le dtype reste double. Ni float32/mixed precision, ni cuDNN, ni kernels de
convolution optimisés par im2col/GEMM, ni plusieurs GPU ne sont ajoutés ici.

## Tests à lancer sur la machine NVIDIA

```powershell
.\build-gpu\tests\Release\clarni_test_nd_cuda.exe
```

Le code retour 0 signifie que les tests ont été exécutés et ont réussi. Le code
77 signifie **ignoré, GPU absent** : ce n'est pas une validation matérielle.
Le test couvre valeurs et gradients, cas de convolution/pooling, perte masquée,
clipping, Adam, compteurs de copies, métriques et portabilité des checkpoints.
Les tolérances acceptent les petites différences d'arrondi CPU/CUDA.

Un helper Windows configure le build puis exige le test matériel avant CTest :

```powershell
.\tools\check_gpu.ps1 -Build build-gpu -MnistData C:\datasets\mnist
```

Avec Compute Sanitizer installé, compléter par exemple avec :

```powershell
compute-sanitizer --tool memcheck .\build-gpu\tests\Release\clarni_test_nd_cuda.exe
```

Les tests `nd_cuda_mock` exécutent sur CPU les mêmes expressions par sortie que
le kernel N-D CUDA, avec le protocole résident simulé. Ils comparent ce chemin à
l'implémentation CPU indépendante. Ils vérifient les maths et les transferts du
protocole, mais ne valident ni l'exécution GPU, ni ses accès mémoire, ni sa vitesse.

## Références

- Runtime CUDA : https://docs.nvidia.com/cuda/cuda-runtime-api/
- Installation CUDA : https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- Compute Sanitizer : https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/
