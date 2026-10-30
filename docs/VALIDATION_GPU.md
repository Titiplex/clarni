# Validation du portage CUDA — 30 octobre 2026

## Ce qui a été exécuté

Environnement : Linux x86-64, GCC 13.3.0, CMake 4.4.4 et **NVCC 13.4.92**.
Les outils CUDA ont été installés pour compiler ; aucun périphérique NVIDIA
utilisable n'est exposé dans cet environnement.

| Contrôle                                                   | Résultat                                                                 |
|------------------------------------------------------------|--------------------------------------------------------------------------|
| Configuration et compilation CPU Release                   | Réussies                                                                 |
| Configuration et compilation CUDA Release                  | Réussies avec NVCC 13.4.92, cible `89`, cuBLAS désactivé                 |
| CTest, build CPU                                           | 27 suites exécutées avec succès, trois suites matérielles ignorées       |
| CTest, build CUDA                                          | Même résultat : les suites de calcul matériel détectent l'absence de GPU |
| Cas de test exécutés                                       | **113 réussis**, dont neuf nouveaux cas sur backend CUDA simulé          |
| Headers publics autonomes                                  | 23 headers compilés indépendamment sous GCC                              |
| ASan/UBSan                                                 | Suites CPU et simulées réussies avec détection des fuites désactivée     |
| Vérificateur GCC direct                                    | Tests et exemples réussis, y compris le nouveau test simulé              |
| MNIST réel, mode CPU                                       | Smoke test réussi depuis les builds CPU et CUDA                          |
| Exécution des nouveaux kernels sur carte NVIDIA            | **Non réalisée ici**                                                     |
| Compute Sanitizer GPU, vitesse et consommation VRAM réelle | **Non mesurés ici**                                                      |
| Windows/MSVC et cuBLAS activé                              | Non validés ici                                                          |

LeakSanitizer nécessite des accès aux processus que cet environnement ne permet
pas correctement. `ASAN_OPTIONS=detect_leaks=0` a été utilisé : aucune absence de
fuite mémoire n'est revendiquée. ASan/UBSan n'instrumentent pas la mémoire CUDA.

## Distinction entre simulation et test matériel

`clarni_test_nd_cuda_mock` utilise un protocole CUDA simulé en mémoire CPU. Il
exécute **les mêmes expressions par élément de sortie que `src/gpu/nd.cu`**, puis
compare valeurs, gradients et mises à jour à l'implémentation CPU indépendante.
Ce partage évite de tester seulement une seconde réécriture des kernels, mais
ne valide pas le matériel, les accès mémoire du périphérique ni les performances.

`clarni_test_nd_cuda` se lie au véritable backend. Sans GPU utilisable, il
retourne 77 et CTest le marque ignoré. Ici, le diagnostic est :

```
CUDA detection : CUDA driver version is insufficient for CUDA runtime version
```

Ce code de retour n'est pas un succès GPU. Le helper `tools/check_gpu.ps1` exige
que le test matériel retourne 0 avant d'annoncer une validation matérielle.

## Neuf nouveaux cas

1. Transferts de backend explicites et gradients traversant un transfert.
2. Permutation, slice, broadcasting et réductions sur le backend résident.
3. Cross-entropy stable avec grands logits et labels ignorés.
4. Conv1D : valeurs et gradients, stride, padding et dilation.
5. Conv2D : kernels rectangulaires, valeurs et gradients.
6. Pooling : maximum/moyenne, chevauchement, égalités et fenêtres à -inf.
7. Clipping, grande norme finie, prédicats, résidence des moments et métriques compactes.
8. Parité d'un CNN entraîné avec Adam, no-grad et checkpoints CPU/GPU.
9. Refus des opérateurs non portés sans téléchargement pour un fallback CPU.

Les mêmes cas sont inclus dans l'exécutable matériel. Les comparaisons utilisent
des tolérances pour les arrondis CPU/CUDA. Les répertoires de checkpoint de ces
tests sont isolés pour permettre une exécution parallèle des variantes simulées
et matérielles.

## Compteurs de copies contrôlés

Pour un pas dont entrée et paramètres sont déjà résidents :

- une seule copie suivie CPU→GPU est attendue pour les labels ;
- les poids, activations, gradients et moments ne doivent pas être copiés en bloc sur CPU ;
- les lectures de normes et contrôles de finitude restent des scalaires ;
- les métriques ne téléchargent qu'un identifiant prédit par exemple ;
- la validation des moments restaurés utilise un scalaire, pas un export complet du second moment.

Ces assertions portent sur les copies suivies par l'API résidente. Elles ne
comptent pas tous les mouvements internes du runtime, notamment les arguments de
lancement des kernels. Une mesure sur carte avec un profiler reste nécessaire.

## Smoke test MNIST

Un essai d'une époque a été exécuté avec 240 images d'entraînement, 60 de
validation et 100 de test, batch 32, graine 42 et Adam à 0.003. Le résultat CPU
est identique depuis les builds sans CUDA et avec CUDA compilé, forcé en mode CPU :

```
validation_loss=1.91004
validation_accuracy=0.483333
test_loss=1.90331
test_accuracy=0.47
```

Ce petit essai vérifie le programme et ses fichiers de sortie. Il ne constitue
pas un résultat GPU ni un benchmark de généralisation. Aucun score ou gain de
vitesse GPU n'a été inventé à partir de la simulation.

Les modes `cpu`, `auto` et `cuda` ont aussi été exercés avant le chargement d'un
dataset volontairement absent. Les modes CPU/auto annoncent leur sélection puis
échouent sur le chemin de données absent ; le mode CUDA strict échoue dès la
sélection du périphérique. Les retours non nuls de ce test de contrôle sont attendus.

## Corrections héritées de l'archive

- Rétablissement d'includes explicites `<cmath>` et `<cstddef>` nécessaires à une
  compilation autonome sous GCC et à certaines unités CUDA.
- Dans le test DataLoader, les deuxième et troisième appels à `next()` avaient
  été affectés à des variables inversées, tout en conservant les assertions de
  l'ordre initial. L'ordre des appels/variables a été corrigé ; les assertions
  restent inchangées et vérifient bien les batchs `[0,1]`, `[2,3]`, puis `[4]`.

## Validation à compléter sur la machine NVIDIA

Suivre `GPU_MNIST.md`, compiler pour l'architecture de la carte, puis lancer :

```powershell
.\build-gpu\tests\Release\clarni_test_nd_cuda.exe
compute-sanitizer --tool memcheck .\build-gpu\tests\Release\clarni_test_nd_cuda.exe
.\build-gpu\examples\Release\clarni_example_mnist.exe `
  --data C:\datasets\mnist --backend cuda --output runs\mnist-cuda
```

Vérifier `backend=CUDA`, le nom de carte et les métriques, puis mesurer les temps
avec les mêmes données, graines et batch sizes qu'en CPU. Les kernels sont des
implémentations de référence float64 ; aucune accélération n'est garantie.
