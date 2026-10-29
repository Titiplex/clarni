# Validation des extensions — 29 octobre 2026

Environnement de validation : Linux x86-64, GCC 13.3.0, C++23, CMake 4.4.4.
Aucune validation MSVC/Windows ni matérielle CUDA n'a été réalisée ici.

## Résultats

- Configuration et compilation CMake Release avec `-Wall -Wextra -Wpedantic` :
  réussies, sans avertissement dans la compilation finale.
- CTest : **21 suites exécutées avec succès**, deux suites matérielles CUDA
  explicitement ignorées. Total : **85 cas de test réussis**, dont 29 nouveaux.
- Les **13 exemples**, anciens et nouveaux, s'exécutent avec un code de retour 0.
- AddressSanitizer + UndefinedBehaviorSanitizer : mêmes 21 suites réussies et deux
  suites matérielles ignorées, avec `ASAN_OPTIONS=detect_leaks=0`.
- La détection des fuites n'est **pas validée** : LeakSanitizer échoue dans cet
  environnement à cause de l'accès restreint à `/proc/.../task` et de son
  incompatibilité avec le traçage du processus. Désactiver cette détection ne
  désactive pas les contrôles d'accès mémoire d'ASan ni ceux d'UBSan.

## Couverture nouvelle

- Formes, permutations, slices, concaténation, réductions et broadcasting.
- Gradients numériques de compositions N-D, de Linear et des pertes.
- Paramètres partagés, détachement, clipping global, identités du trainer.
- Reprise exacte d'un entraînement déterministe avec Adam ; rejet transactionnel
  de checkpoint corrompu ; rejet d'un trainer appartenant à un autre modèle.
- Embeddings avec identifiants répétés, padding gelé, vocabulaire et SkipGram.
- Gradients des entrées/filtres/biais Conv1D et Conv2D, stride, padding, dilation,
  kernels rectangulaires, pooling avec fenêtres qui se chevauchent et égalités.
- BPTT RNN/GRU/LSTM, paramètres partagés dans le temps, états initiaux,
  longueurs variables, séquences entièrement masquées et BPTT tronquée.
- LayerNorm, gradients Q/K/V et multi-têtes, lignes entièrement masquées,
  masques de padding, causalité d'un bloc Transformer.
- Surapprentissage volontaire de minuscules problèmes pour chaque famille.

Les tests numériques évaluent les gradients par différences finies centrées.
Ils ne remplacent pas l'évaluation d'un modèle sur un jeu de données indépendant.
Les exemples utilisent tous des données synthétiques ou un corpus jouet intégré.

## Exemples : résultats observés en Release

| Exemple nouveau | Résultat |
|---|---|
| `modular_mlp` | XOR, entropie croisée ≈ 0.000431 |
| `word_embeddings` | SkipGram, perte 2.36058 → 0.775308 |
| `embedding_cnn` | Classification, perte 0.84013 → 0.000283 |
| `cnn` | Classification Conv1D, perte 0.673292 → 0.002509 |
| `character_rnn` | Perte 1.42508 → 0.001200 ; génération du motif `abc` |
| `recurrent_comparison` | MSE RNN ≈ 1.57e-6, GRU ≈ 4.72e-8, LSTM ≈ 1.39e-8 |
| `tiny_transformer` | Perte 1.45025 → 0.000405 ; génération du motif `abc` |

Ces chiffres dépendent du compilateur et de l'implémentation de la distribution
aléatoire standard ; ce ne sont pas des mesures de généralisation ou de vitesse.

## Reproduction

```sh
cmake -S . -B build-cpu -DML_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel
ctest --test-dir build-cpu --output-on-failure
```

Pour ASan/UBSan avec GCC ou Clang sous Linux :

```sh
cmake -S . -B build-sanitize -DML_ENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g"
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

Conserver la détection des fuites activée sur une machine qui la prend en charge.
Dans un environnement où LeakSanitizer ne peut pas fonctionner, utiliser
`ASAN_OPTIONS=detect_leaks=0` en documentant cette limitation.
