# Test different states 3 times. 
# First group: - Individual states [backlog, drops, bw_rx, bw_tx]
# Second group: - Difference for backlog [d_backlog, D_backlog]
# Third group: - Combined states [backlog+bw_rx, backlog+drops, backlog+bw_tx]
# Fourth group: - Combined states [backlog+drops+bw_rx, bw_tx]

iterations=3
timesteps=1000000

for i in 1 .. $iterations; do
    echo "SB3 testing state backlog, iteration $i/$iterations using $timesteps timesteps"
    # Save the output to a file
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Testing state [d_backlog], iteration $i/$iterations with $timesteps timesteps" >> states_test_sb3.log
    
    ~/Github/iroko/ir_env/bin/python ~/Github/iroko/dc_gym/sb3Solve.py --reward_model joint_queue --state_model d_backlog --total_timesteps $timesteps
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Testing state [D_backlog], iteration $i/$iterations with $timesteps timesteps" >> states_test_sb3.log
    
    ~/Github/iroko/ir_env/bin/python ~/Github/iroko/dc_gym/sb3Solve.py --reward_model joint_queue --state_model D_backlog --total_timesteps $timesteps

    
done


# Expected outcome: 3 sets of results. backlog drops bw_rx bw_tx should the best reward than all other tests. 
#  Episodic Length of backlog drops bw_rx bw_tx is expected to be shorter due to slower processing.
# Status of the current run will be saved in states_test.log
# A comparison of results can be done by comparing the wandb dashboard for each run.