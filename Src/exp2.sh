for syn_ep in 5 10 15 20 30
do
  for syn_st in 3 5 7
  do
    for g_st in 1 10
    do
      for gs in 10 15
      do
        for ea in 0 0.1
        do
          echo $syn_ep
          echo $syn_st
          echo $g_st
          echo $gs
          echo $ea
          python run_NS.py --max_episodes_syntheticTrajectory $syn_ep --max_step_syntheticTrajectory $syn_st --gradient_step $g_st --grid_size $gs --entropy_alpha $ea;
          sudo sh -c 'echo 1 >/proc/sys/vm/drop_caches' && sudo sh -c 'echo 2 >/proc/sys/vm/drop_caches' && sudo sh -c 'echo 3 >/proc/sys/vm/drop_caches';
        done
      done
    done
  done
done