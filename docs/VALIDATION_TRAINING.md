# Validation du pipeline d'entraînement — 29 octobre 2026

## Environnement et périmètre

Linux x86-64, GCC 13.3.0, C++23, CMake 4.4.4. Les données MNIST ont été récupérées
sur le miroir CVDF et les quatre archives ont passé leurs checksums conventionnels.
Aucun entraînement n'a utilisé de bibliothèque ML externe.

Les nouvelles fonctionnalités restent CPU / float64. Windows/MSVC et les kernels
CUDA historiques ne sont pas validés matériellement ici. Les commandes Windows
fournies sont distinctes des mesures effectuées sous Linux.

## Vérifications logicielles

- Compilation Release avec `-Wall -Wextra -Wpedantic`, sans avertissement final.
- **104 cas de test passent** dans **26 suites exécutées** ; deux suites
  matérielles CUDA sont explicitement ignorées par CTest.
- Les **21 headers publics** compilent indépendamment sous GCC.
- Les 13 anciens exemples et le nouvel exemple de séquences s'exécutent avec
  succès. Le programme MNIST est également compilé, son aide est vérifiée et
  son entraînement réel est exécuté sur trois graines.
- ASan + UBSan : les mêmes suites CPU passent avec `detect_leaks=0`.
  La détection des fuites n'est pas validée : LeakSanitizer ne fonctionne pas
  correctement dans cet environnement de processus tracés et d'accès restreint
  à `/proc`. Cette désactivation ne supprime pas les contrôles mémoire d'ASan
  ni les contrôles de comportement indéfini d'UBSan.

Les tests ajoutés couvrent les gardes no-grad imbriqués, les exceptions et
l'isolation entre threads ; une chaîne autograd de 40 000 nœuds ; l'adoption du
stockage ; les gradients numériques de produits matriciels par lots broadcastés ;
les queues de batch ; le padding de tokens/features ; le split disjoint ;
la reprise du sampler et de son RNG ; le format IDX ; les métriques pondérées ;
les schedules ; la patience ; et la restauration des paramètres, moments d'Adam
et sampler. Une reprise au milieu d'une époque est comparée mise à jour après
mise à jour à un entraînement ininterrompu, jusque dans l'époque suivante.

Le test des métriques inclut des pertes finies très grandes : leur moyenne doit
rester finie, tandis que la perplexité peut légitimement déborder à +inf.

## Expérience MNIST réelle

Configuration fixée avant lecture des scores :

- Premiers 2 000 exemples du split officiel d'entraînement, séparés en 1 800
  exemples d'entraînement et 200 de validation par la graine de l'essai.
- Premiers 1 000 exemples du split officiel de test, communs aux trois essais.
- CNN : Conv2D(1,4,3), ReLU, max pooling 2×2, Flatten, Linear(676,10).
- Adam, learning rate 0.003, batch size 32, trois époques, clipping global à 5.
- Choix du meilleur modèle par perte de validation uniquement ; évaluation test
  après sélection. Les trois meilleures époques sont ici les troisièmes.

| Graine | Accuracy validation | Perte test | Accuracy test | Macro F1 test |
|---|---:|---:|---:|---:|
| 7 | 86.5 % | 0.369288 | 88.1 % | 0.878115 |
| 19 | 88.0 % | 0.390928 | 87.5 % | 0.871826 |
| 42 | 90.5 % | 0.339570 | 90.1 % | 0.898447 |

La moyenne d'accuracy test de ces trois essais est d'environ **88.6 %**.
La graine modifie ici le split de validation, l'initialisation et le mélange :
la dispersion ne mesure pas uniquement l'effet de l'initialisation.

Ce résultat valide une chaîne réelle de données et d'apprentissage. Ce n'est
pas un score sur l'ensemble du benchmark MNIST, ni une comparaison à l'état de
l'art. Les limites de taille sont explicites ; les splits officiels entiers
restent à entraîner et évaluer. Aucun réglage n'a été choisi à partir des trois
scores de test rapportés.

Les fichiers `epochs.csv`, `config.txt`, `metrics.txt` et les listes d'indices
permettent d'inspecter les essais. Le dossier de livraison contient leurs copies
légères dans ses preuves de validation ; les images et les poids entraînés ne
sont pas intégrés aux patchs.

## Mesure de forward

Le benchmark fourni compare sur cette version le même MLP en mode eval avec ou
sans enregistrement du graphe, après warmup. Les prédictions doivent être
identiques. Cette mesure sur un hôte partagé ne constitue ni une comparaison
avant/après la série, ni une garantie de vitesse. Le coût des produits matriciels
peut dominer celui de l'enregistrement du graphe.

## Reproduction

```sh
cmake -S . -B build-cpu -DML_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
  -DML_BUILD_BENCHMARKS=ON
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
python3 tools/check_headers.py
python3 tools/download_mnist.py /chemin/mnist
build-cpu/examples/clarni_example_mnist --data /chemin/mnist \
  --output runs/mnist-42 --train-limit 2000 --validation 200 \
  --test-limit 1000 --epochs 3 --seed 42
build-cpu/benchmarks/clarni_benchmark_pipeline
```

Pour ASan/UBSan avec GCC/Clang :

```sh
cmake -S . -B build-sanitize -DML_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g"
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

Ne désactiver LeakSanitizer que si l'environnement l'exige, en conservant cette
limitation dans le compte rendu. Les instructions Visual Studio sont dans
`TRAINING_PIPELINE.md`.
