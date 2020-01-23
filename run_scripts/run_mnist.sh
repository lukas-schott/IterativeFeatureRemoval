# done, EnsembleCW0
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0. --training_mode redundancy --exp_series mnist --ename EnsembleCW0 --train_greedily True

# done, EnsembleCW0005
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename EnsembleCW0005 --train_greedily True

# sys1_0, EnsembleCW005
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.05 --training_mode redundancy --exp_series mnist --ename EnsembleCW005 --train_greedily True

# gpu19_0, EnsembleCW0005WD
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename EnsembleCW0005WD --train_greedily True --weight_decay 0.0001

# fbgpu1_5, EnsembleCW0005LogitDiff
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename EnsembleCW0005LogitDiff --train_greedily True --logits_for_similarity target_vs_best_other

# gpu8_1, EnsembleCW0005LogitDiffWD
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename EnsembleCW0005LogitDiffWD --train_greedily True --weight_decay 0.0001 --logits_for_similarity target_vs_best_other

# sys3_3, EnsembleCW0005LogitDiffWDProj
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename EnsembleCW0005LogitDiffWDProj --train_greedily True --weight_decay 0.0001 --logits_for_similarity target_vs_best_other --projection_exponent 2.

# sys1_1, RandomVecTvsOLogitEnsembleCW0005WD
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename RandomVecTvsOLogitEnsembleCW0005WD --train_greedily True --weight_decay 0.0001 --logits_for_similarity target_vs_best_other --similarity_measure random_vec

# GN
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename GNnBest --train_greedily True --weight_decay 0.0001 --logits_for_similarity target_vs_best_other --add_gaussian_noise_during_training 0.01

# done, NN best
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.05 --training_mode redundancy --exp_series mnist --ename NNEnsembleCW005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# done, NN best
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.5 --training_mode redundancy --exp_series mnist --ename NNEnsembleCW05TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# sys1_0, NN best
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename NNEnsembleCW0005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# gpu19_1, NN best
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.0005 --training_mode redundancy --exp_series mnist --ename NNEnsembleCW00005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other


------------------
# sys1_2, EnsembleCW005TvsBo
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.05 --training_mode redundancy --exp_series mnist --ename EnsembleCW005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# gpu8_3, EnsembleCW0005TvsBo
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.005 --training_mode redundancy --exp_series mnist --ename EnsembleCW0005TvsBo --train_greedily True --logits_for_similarity target_vs_best_other

# gpu19_1, EnsembleCW05TvsBo
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.5 --training_mode redundancy --exp_series mnist --ename EnsembleCW05TvsBo --train_greedily True --logits_for_similarity target_vs_best_other


# gpu19_1, VanillaEnsembleCW0
gpumonitor run -- ./agmb-docker run --rm -it lukasschott/ifr:v2 python3 ~/src/IterativeFeatureRemoval/main.py --dissimilarity_weight 0.0 --training_mode redundancy --exp_series mnist --ename VanillaEnsembleCW0 --train_greedily True
