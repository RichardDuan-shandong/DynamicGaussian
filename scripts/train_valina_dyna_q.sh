# example to run dyna-q strategy:
#       bash scripts/train_valina_dyna_q.sh
# this file does not support other examples.
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1

# set the method name
method=${1}

# set the seed number
seed="0"
# set the gpu id for training. we use two gpus for training. you could also use one gpu.
train_gpu=${2:-"0,1"}
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
port=${3:-"12345"}
# you could enable/disable wandb by this.
use_wandb=True

cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/online_data/train_data"
test_demo_path="${cur_dir}/data/online_data/test_data"

# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${4:-"${method}_${addition_info}"}
replay_dir="${cur_dir}/replay/online_replay/${exp_name}"

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (5s)"
sleep 5s
tmux kill-session -t ${exp_name}
sleep 3s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}

tmux select-pane -t 0
tmux send-keys "conda activate manigaussian; CUDA_VISIBLE_DEVICES=${train_gpu} python test_agent.py method=$method \
rlbench.task_name=${exp_name} \
rlbench.demo_path=${train_demo_path} \
replay.path=${replay_dir} \
framework.start_seed=${seed} \
framework.use_wandb=${use_wandb} \
method.use_wandb=${use_wandb} \
framework.wandb_group=${exp_name} \
framework.wandb_name=${exp_name} \
ddp.num_devices=${#train_gpu_list[@]} \
replay.batch_size=${batch_size} \
ddp.master_port=${port} \
rlbench.tasks=${tasks} \
rlbench.demos=${demo} \
method.neural_renderer.render_freq=${render_freq} \
method.neural_renderer.lambda_embed=${lambda_embed} \
method.neural_renderer.lambda_dyna=${lambda_dyna} \
method.neural_renderer.lambda_reg=${lambda_reg} \
method.neural_renderer.foundation_model_name=diffusion \
method.neural_renderer.use_dynamic_field=True" C-m

# remove 0.ckpt
rm -rf logs/${exp_name}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}