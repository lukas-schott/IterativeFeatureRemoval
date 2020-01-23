# key experiments

# done, EnsembleCW0
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0. --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW0 --train_greedily True

# done, EnsembleCW0005 --> old best 1
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.005 --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW0005 --train_greedily True

# EnsembleCW005
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.05 --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW005 --train_greedily True

# EnsembleCW05
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.5 --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW05 --train_greedily True

# done, EnsembleCW005, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.05 --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW005 --train_greedily True --logits_for_similarity target_vs_best_other

# done, ProjExpEnsembleCW0005WD --> best
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.005 --training_mode redundancy --exp_series shift_mnist --ename ProjExpEnsembleCW0005WD --train_greedily True --projection_exponent 2. --weight_decay 0.0001

# done, EnsembleCW0005WD
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.005 --training_mode redundancy --exp_series shift_mnist --ename ProjExpEnsembleCW0005WD --train_greedily True --weight_decay 0.0001

# done, TvsOLogitEnsembleCW0005WD
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.005 --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW0005WD --train_greedily True --weight_decay 0.0001 --logits_for_similarity target_vs_best_other

# done, EnsembleCW0WD
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0. --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW0WD --train_greedily True --weight_decay 0.0001

# done, random vectore RandomVecTvsOLogitEnsembleCW0005WD
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.005 --training_mode redundancy --exp_series shift_mnist --ename RandomVecTvsOLogitEnsembleCW0005WD --train_greedily True --weight_decay 0.0001 --logits_for_similarity target_vs_best_other --similarity_measure random_vec

# done, GNnBest
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.005 --training_mode redundancy --exp_series shift_mnist --ename GNnBest --train_greedily True --weight_decay 0.0001 --logits_for_similarity target_vs_best_other --add_gaussian_noise_during_training 0.01

# done, NNEnsembleCW005, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.05 --training_mode redundancy --exp_series shift_mnist --ename NNEnsembleCW005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# done, NNEnsembleCW05, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.5 --training_mode redundancy --exp_series shift_mnist --ename NNEnsembleCW05TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# sys1_2, NNEnsembleCW0005, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.005 --training_mode redundancy --exp_series shift_mnist --ename NNEnsembleCW0005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# gpu8_3, NNEnsembleCW0005, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.0005 --training_mode redundancy --exp_series shift_mnist --ename NNEnsembleCW00005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other


---------------
# done, NNEnsembleCW005, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.05 --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# done, NNEnsembleCW0005, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.005 --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW0005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# done, NNEnsembleCW05, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.5 --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW05TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# done, VanillaNNEnsembleCW0
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.0 --training_mode redundancy --exp_series shift_mnist --ename VanillaEnsembleCW0 --train_greedily True

# sys1_0, NNEnsembleCW5, target_vs_best_other -->
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 5. --training_mode redundancy --exp_series shift_mnist --ename EnsembleCW5TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# gpu19_1, StdHLR
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.05 --training_mode redundancy --exp_series shift_mnist --ename EnsembleHLr --train_greedily True --logits_for_similarity target_vs_best_other --lr 0.005

# sys1_1f, StdWD
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.05 --training_mode redundancy --exp_series shift_mnist --ename EnsembleHLr --train_greedily True --logits_for_similarity target_vs_best_other --weight_decay 0.0001

# sys1_2, StdHHLR
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dataset_modification shift_mnist --dissimilarity_weight 0.05 --training_mode redundancy --exp_series shift_mnist --ename EnsembleHLr --train_greedily True --logits_for_similarity target_vs_best_other --lr 0.01

