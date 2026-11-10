# v7_verified results (verified pyrepro backend)

21/24 arms complete. Pending: full_nn_lib_s1, full_nonn_lib_s2, full_nonn_nolib_s0.

Backend: dreamcoder-pyrepro fdf68ad (pyfast enumerator, pyvs compressor). This backend predates the list-domain parity fixes 0f7f479 and e473c9b; the cube results were not rerun on them.

Each cell is mean ± sd over the completed seeds (seed values in parentheses). E2E = solved out of the same 100 unseen random cubes (seed 9999); train = training tasks solved after 20 rounds; lib = learned library size at round 20. 20 rounds, -t 5, 4 CPUs; NN arms use contextual recognition + biasOptimal.

| Training | Arm | seeds | E2E solved / 100 | Train solved | Library size | Mean moves (solved cubes) |
|---|---|---|---|---|---|---|
| stages | NN + library | 3/3 | 100 ± 0 (100 / 100 / 100) | 119 ± 0 / 119 (119 / 119 / 119) | 35 ± 4.36 | 82.08 ± 14.56 |
| stages | NN, no library | 3/3 | 100 ± 0 (100 / 100 / 100) | 119 ± 0 / 119 (119 / 119 / 119) | 0 ± 0 | 51.72 ± 4.71 |
| stages | No NN + library | 3/3 | 94.67 ± 5.51 (95 / 89 / 100) | 108 ± 7.81 / 119 (112 / 113 / 99) | 27.67 ± 4.93 | 74.51 ± 15.19 |
| stages | No NN, no library | 3/3 | 6 ± 0 (6 / 6 / 6) | 55.33 ± 6.66 / 119 (61 / 57 / 48) | 0 ± 0 | 20.67 ± 0 |
| full | NN + library | 2/3 | 12.5 ± 2.12 (11 / 14) | 11 ± 5.66 / 120 (15 / 7) | 2.5 ± 2.12 | 21.81 ± 13.77 |
| full | NN, no library | 3/3 | 8.33 ± 0.58 (8 / 8 / 9) | 7.33 ± 2.08 / 120 (8 / 9 / 5) | 0 ± 0 | 12.44 ± 1.47 |
| full | No NN + library | 2/3 | 0 ± 0 (0 / 0) | 0 ± 0 / 120 (0 / 0) | 0 ± 0 | — |
| full | No NN, no library | 2/3 | 0 ± 0 (0 / 0) | 0 ± 0 / 120 (0 / 0) | 0 ± 0 | — |
