# `code` folder


### Word2Vec models

Found in `models` subfolder. Computing data for __8 threads__ in parallel. Downsampling set at .001. Context window set at 10.

| Features | Min. words | Voc. size | Comp. time | Comp. speed |
| --- | --- | --- | --- | --- |
| 100 | 60 | 12 963 | 39.0s | 1 665 601 words/s |
| 300 | 60 | 12 963 | 58.7s | 1 107 160 words/s |
| 1000 | 60 | 12 96 | 150.8s | 430 721 words/s |
| 100 | 40 | 16 493 | 45.9s | 1 436 928 words/s |
| 300 | 40 | 16 493 | 61.8s | 1 66 090 words/s |
| 1000 | 40 | 16 493 | 157.2s | 419 240 words/s |
| 100 | 20 | 24 159 | 46.1s | 1 456 663 words/s |
| 300 | 20 | 24 15 | 67.3s | 997 401 words/s |
| 1000 | 20 | 24 159 | 160.3s | 418 596 words/s |