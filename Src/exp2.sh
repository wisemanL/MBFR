echo Dlgusdls96$ | sudo -S sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches';
python run_NS.py --max_episodes_syntheticTrajectory 30 --max_step_syntheticTrajectory 7 --gradient_step 10 --grid_size 15 --entropy_alpha -0.1 --sars_batchSize_for_policyUpdate 100;

echo Dlgusdls96$ | sudo -S sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches';
python run_NS.py --max_episodes_syntheticTrajectory 15 --max_step_syntheticTrajectory 5 --gradient_step 10 --grid_size 10 --entropy_alpha -0.1 --sars_batchSize_for_policyUpdate 100;

echo Dlgusdls96$ | sudo -S sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches';
python run_NS.py --max_episodes_syntheticTrajectory 10 --max_step_syntheticTrajectory 5 --gradient_step 10 --grid_size 10 --entropy_alpha -0.1 --sars_batchSize_for_policyUpdate 100;

