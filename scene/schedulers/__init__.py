from scene.schedulers.piecewise_schedulers import CGIterScheduler, BatchSizeScheduler, SampleSizeScheduler, LambdaScheduler
piecewise_schedulers =  {"cgiter_sched": CGIterScheduler, "batchsize_sched": BatchSizeScheduler, 
                         "samplesize_sched": SampleSizeScheduler, "lambda_sched":LambdaScheduler}