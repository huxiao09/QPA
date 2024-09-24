# if not in debug mode, need to specify wandb config manually in 'config/train_QPA.yaml' and set 'wandb=true'

# walker_walk:
for seed in 1 2 3 4 5; do
python train_QPA.py env=walker_walk agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
done

# cheetah_run:
for seed in 1 2 3 4 5; do
python train_QPA.py env=cheetah_run agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
done

# walker_run:
for seed in 1 2 3 4 5; do
python train_QPA.py env=walker_run agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=250 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
done

# quadruped_walk:
for seed in 1 2 3 4 5; do
python train_QPA.py env=quadruped_walk agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=30000 max_feedback=1000 reward_batch=100 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
done

# quadruped_run:
for seed in 1 2 3 4 5; do
python train_QPA.py env=quadruped_run agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=30000 max_feedback=1000 reward_batch=100 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
done

# humanoid_stand:
for seed in 1 2 3 4 5; do
python train_QPA.py env=humanoid_stand agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=2000000 num_interact=5000 max_feedback=10000 reward_batch=50 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
done

# metaworld_drawer-open:
for seed in 1 2 3 4 5; do
python train_QPA.py env=metaworld_drawer-open-v2 agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=5000 max_feedback=3000 reward_batch=30 reward_update=200 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=30 ensemble_size=1 explore=false her_ratio=0.5 num_eval_episodes=100 wandb=true device='cuda' seed=$seed
done

# metaworld_door-open:
for seed in 1 2 3 4 5; do
python train_QPA.py env=metaworld_door-open-v2 agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=5000 max_feedback=3000 reward_batch=30 reward_update=200 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=60 ensemble_size=1 explore=false her_ratio=0.5 num_eval_episodes=100 wandb=true device='cuda' seed=$seed
done

# metaworld_door-unlock:
for seed in 1 2 3 4 5; do
python train_QPA.py env=metaworld_door-unlock-v2 agent.params.actor_lr=0.0001 agent.params.critic_lr=0.0001 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=2000 reward_batch=100 reward_update=200 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=30 ensemble_size=1 explore=false her_ratio=0.5 num_eval_episodes=100 wandb=true device='cuda' seed=$seed
done
