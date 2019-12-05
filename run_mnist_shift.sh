

#########################
# shift_mnist_2


# gpu 8_1, GreedyCW5
python3 main.py --cosine_dissimilarity_weight 5. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW5 --train_greedily True --dataset_modification shift_mnist

# gpu15_1, JointCW5
python3 main.py --cosine_dissimilarity_weight 5. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name JointCW5 --train_greedily False --dataset_modification shift_mnist

# gpu8_0, GreedyCW05
python3 main.py --cosine_dissimilarity_weight 0.5 --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW05 --train_greedily True --dataset_modification shift_mnist

# gpu8_3, JointVanilla, CW0
python3 main.py --cosine_dissimilarity_weight 0. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name JointVanilla --train_greedily False --dataset_modification shift_mnist

# gpu 8_1, GreedyCW50
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW50 --train_greedily True --dataset_modification shift_mnist

# gpu 8_2, GreedyCW50AllLogits
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW50AllLogits --train_greedily True --dataset_modification shift_mnist

# gpu 8_1, GreedyCW50AllLogits
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW50AllLogits --train_greedily True --dataset_modification shift_mnist --all_logits True


# gpu16_0 , GreedyCW200
python3 main.py --cosine_dissimilarity_weight 200. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW200 --train_greedily True --dataset_modification shift_mnist


python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name trash --train_greedily True --dataset_modification shift_mnist

# gpu19_0, joint5 gut greedy
python3 main.py --cosine_dissimilarity_weight 5. --training_mode redundancy --lr 0.0005 --loss_fct soft_ce --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name JointAndGreedyCW5 --dataset_modification shift_mnist




############ reimplementation
# gpu8_0, JoinCW0
python3 main.py --cosine_dissimilarity_weight 0. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name VanillaJointCW0 --dataset_modification shift_mnist

# gpu8_3, GreedyCW0
python3 main.py --cosine_dissimilarity_weight 0. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name VanillaGreedyCW0 --dataset_modification shift_mnist --train_greedily True

# gpu14_0, greedyCW50
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW50 --dataset_modification shift_mnist --train_greedily True
# rerun
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name RerunJointAndGreedyCW50 --dataset_modification shift_mnist
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ReRerunJointAndGreedyCW50 --dataset_modification shift_mnist
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.001 --weight_decay 1e-4 --n_redundant 5 --exp_series shift_mnist --name SGDReRerunJointAndGreedyCW50 --dataset_modification shift_mnist

# fbgpu1_4, joint50 gut greedy
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name JointAndGreedyCW50 --dataset_modification shift_mnist

# sys3_1, joint5 gut greedy
python3 main.py --cosine_dissimilarity_weight 5. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name JointAndGreedyCW5 --dataset_modification shift_mnist

# sys3_0, greedyCW5
python3 main.py --cosine_dissimilarity_weight 5. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW5 --dataset_modification shift_mnist --train_greedily True


# gpu19_1, all_logitsgreedyCW50,
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name AllLogitsGreedyCW50 --dataset_modification shift_mnist --train_greedily True --all_logits True
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 5 --exp_series shift_mnist --optimizer adam --name AbsReRerunJointAndGreedyCW50 --dataset_modification shift_mnist



##  shift_mnist v3

# JoinCW0, VanillaJointCW0
python3 main.py --cosine_dissimilarity_weight 0. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name VanillaJointCW0 --dataset_modification shift_mnist

# gpu8_3, VannillaGreedyCW0
python3 main.py --cosine_dissimilarity_weight 0. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name VanillaGreedyCW0 --dataset_modification shift_mnist --train_greedily True

# gpu8_3, greedyCW50
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW50 --dataset_modification shift_mnist --train_greedily True

# fbgpu1_4, joint50 gut greedy
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name JointAndGreedyCW50 --dataset_modification shift_mnist

# gpu19_1, AllLogitsGreedyCW50,
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name AllLogitsGreedyCW50 --dataset_modification shift_mnist --train_greedily True --all_logits True

# gpu18 greedyCW200
python3 main.py --cosine_dissimilarity_weight 200. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW200 --dataset_modification shift_mnist --train_greedily True

#  greedyCW200
python3 main.py --cosine_dissimilarity_weight 200. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GNGreedyCW200 --dataset_modification shift_mnist --train_greedily True --add_gaussian_noise_during_training True

#  GNGreedyCW50
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GNGreedyCW50 --dataset_modification shift_mnist --train_greedily True --add_gaussian_noise_during_training True

#  gpu16_0, VanillaGNGreedyCW0
python3 main.py --cosine_dissimilarity_weight 0. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name VanillaGNGreedyCW0 --dataset_modification shift_mnist --train_greedily True --add_gaussian_noise_during_training True

#  LongGreedyCW50
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 100 --exp_series shift_mnist --optimizer adam --name LongGNGreedyCW50 --dataset_modification shift_mnist --train_greedily True --add_gaussian_noise_during_training True --n_epochs 1000


#  ModifiedCosineSimilarity
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ModifiedCosineSimilarityGreedyCW50 --dataset_modification shift_mnist --train_greedily True


# fbgpu1_6, gradient_regularization_weight
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name RegularizationGreedyCW50 --dataset_modification shift_mnist --gradient_regularization_weight 1. --train_greedily True


# sys2_1, scalar_prod_as_similarity
python3 main.py --cosine_dissimilarity_weight 5. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ScalarProdreedyCW5 --dataset_modification shift_mnist --train_greedily True --scalar_prod_as_similarity True

# gpu9_0, scalar_prod_as_similarity
python3 main.py --cosine_dissimilarity_weight 0.5 --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ScalarProdreedyCW05 --dataset_modification shift_mnist --train_greedily True --scalar_prod_as_similarity True

# gpu9_2, scalar_prod_as_similarity
python3 main.py --cosine_dissimilarity_weight 0.05 --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ScalarProdreedyCW005 --dataset_modification shift_mnist --train_greedily True --scalar_prod_as_similarity True




# key experiments
# fbgpu1_6, VanillaJointCW0
python3 main.py --cosine_dissimilarity_weight 0. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name VanillaJointCW0 --dataset_modification shift_mnist

# fbgpu1_4, VanillaGreedyCW0
python3 main.py --cosine_dissimilarity_weight 0. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name VanillaGreedyCW0 --dataset_modification shift_mnist --train_greedily True

# sys2_1, AllLogitsGreedyCW50,
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name AllLogitsGreedyCW50 --dataset_modification shift_mnist --train_greedily True --all_logits True

# sys1_1, greedyCW50
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name GreedyCW50 --dataset_modification shift_mnist --train_greedily True

# sys1_2, JointCW50
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name JointCW50 --dataset_modification shift_mnist

# sys1_3, JointGenerousCW50
python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name JointGenerousCW50 --dataset_modification shift_mnist

# fbgpu1_5, ScalarProdGreedyCW005
python3 main.py --cosine_dissimilarity_weight 0.05 --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ScalarProdGreedyCW005 --dataset_modification shift_mnist --train_greedily True --scalar_prod_as_similarity True

# fbgpu1_0, ScalarProdJoint
python3 main.py --cosine_dissimilarity_weight 0.05 --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ScalarProdJointCW005 --dataset_modification shift_mnist --scalar_prod_as_similarity True

# fbgpu1_2, ScalarProdJointGenerous
python3 main.py --cosine_dissimilarity_weight 0.05 --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ScalarProdJointGenerousCW005 --dataset_modification shift_mnist --scalar_prod_as_similarity True

# GNScalarProdGreedyCW005
python3 main.py --cosine_dissimilarity_weight 0.05 --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ScalarProdGreedyCW005 --dataset_modification shift_mnist --train_greedily True --scalar_prod_as_similarity True --add_gaussian_noise_during_training 0.02

# sys1_0, ScalarProdJointCW0005
python3 main.py --cosine_dissimilarity_weight 0.005 --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name ScalarProdJointCW0005 --dataset_modification shift_mnist --scalar_prod_as_similarity True



python3 main.py --cosine_dissimilarity_weight 50. --training_mode redundancy --lr 0.0005 --weight_decay 1e-4 --n_redundant 10 --exp_series shift_mnist --optimizer adam --name trash --dataset_modification shift_mnist --train_greedily True --all_logits True --n_epochs_per_net 2
