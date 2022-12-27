for syn_ep in 30 50 70
do
  for syn_st in 5 7 9
  do
    for g_st in 1 10
    do
      for gs in 10 15
      do
        for ea in 0 0.1 0.2
        do
          echo $syn_ep
          echo $syn_st
          echo $g_st
          echo $gs
          echo $ea
          echo Dlgusdls96$ | sudo -S sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo -S sh -c 'echo 3 >/proc/sys/vm/drop_caches';
          python run_NS.py --max_episodes_syntheticTrajectory $syn_ep --max_step_syntheticTrajectory $syn_st --gradient_step $g_st --grid_size $gs --entropy_alpha $ea;
        done
      done
    done
  done
done