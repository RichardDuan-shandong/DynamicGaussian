# this script launch the process of training the gaussian_rendering module
# just run in terminal:
#       bash scripts/perception_trainer/gaussian_pretrain.sh

task=${1}
# set the seed number
seed="0"
# set the gpu id for training. we use two gpus for training. you could also use one gpu.
train_gpu=${2:-"4,5,6,7"}   
train_gpu_list=(${train_gpu//,/ })

# set the port for ddp training.
# you could enable/disable wandb by this.
use_wandb=True

cur_dir=$(pwd)
train_demo_path="${cur_dir}/data/train_data"
test_demo_path="${cur_dir}/data/test_data"
seg_save="${cur_dir}/data/data_temp"

# we set experiment name as method+date. you could specify it as you like.
addition_info="$(date +%Y%m%d)"
exp_name=${3:-"gaussian_rendering_train_${addition_info}"}
tasks=[close_jar,open_drawer,sweep_to_dustpan_of_size,meat_off_grill,turn_tap,slide_block_to_color_target,put_item_in_drawer,reach_and_drag,push_buttons,stack_blocks]

# create a tmux window for training
echo "I am going to kill the session ${exp_name}, are you sure? (1s)"
sleep 1s
tmux kill-session -t ${exp_name}
sleep 1s
echo "start new tmux session: ${exp_name}, running main.py"
tmux new-session -d -s ${exp_name}

tmux select-pane -t 0 
tmux send-keys "conda activate manigaussian; CUDA_VISIBLE_DEVICES=${train_gpu} python train_gaussian_rendering.py \
rlbench.task_name=${exp_name} \
rlbench.demo_path=${train_demo_path} \
framework.start_seed=${seed} \
framework.use_wandb=${use_wandb} \
method.use_wandb=${use_wandb} \
framework.wandb_group=${exp_name} \
framework.wandb_name=${exp_name} \
ddp.num_devices=${#train_gpu_list[@]} \
rlbench.tasks=${tasks} \
train.seg_save=${seg_save} \
rlbench.demos=${demo} " C-m

# remove 0.ckpt
rm -rf logs/${exp_name}/seed${seed}/weights/0

tmux -2 attach-session -t ${exp_name}            