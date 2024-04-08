for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    python train_PPO_Unsuper.py --env metaworld_button-press-v2 --seed $seed --lr 0.0003 --batch-size 128 --n-envs 8 --ent-coef 0.0 --n-steps 250 --total-timesteps 4000000 --num-layer 3 --hidden-dim 256 --clip-init 0.4 --gae-lambda 0.92 --unsuper-step 32000 --unsuper-n-epochs 50
done
