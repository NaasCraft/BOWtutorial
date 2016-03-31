# `code` folder


### Word2Vec models

Found in `models` subfolder. Computing data for __8 threads__ in parallel. Downsampling set at .001. Context window set at 10.

Model presented as follows :

> _\[x\]features\_[x]minword_ : \[vocabulary size\]\[computing time\] \[computing speed\] 

+ _100features\_60minwords_ : **12 963** **39.0**s **1 665 601**words/s
+ _300features\_60minwords_ : **12 963** **58.7**s **1 107 160**words/s
+ _1000features\_60minwords_ : **12 963** **150.8**s **430 721**words/s
+ _100features\_40minwords_ : **16 493** **45.9**s **1 436 928**words/s
+ _300features\_40minwords_ : **16 493** **61.8**s **1 66 090**words/s
+ _1000features\_40minwords_ : **16 493** **157.2**s **419 240**words/s
+ _100features\_20minwords_ : **24 159** **46.1**s **1 456 663**words/s
+ _300features\_20minwords_ : **24 159** **67.3**s **997 401**words/s
+ _1000features\_20minwords_ : **24 159** **160.3**s **418 596**words/s