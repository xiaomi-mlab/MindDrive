from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook,
                         build_runner, )
from mmcv.datasets.pipelines import Compose


def custom_collect_dataset(model,
                   cfg,
                   args=None,
                   ):
    data_loaders = None # we collect the data in carla
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    runner = build_runner(
            cfg.runner,
            default_args=dict(model=model,logger=logger))
    # Load Checkpoints   
    if cfg.load_from is not None:
        runner.load_checkpoint(cfg.load_from)

    if cfg.runner['type'] == 'RLIterBasedRunner':
        if hasattr(cfg, 'PENALTY_CONFIG'):
            args.PENALTY_CONFIG = cfg.PENALTY_CONFIG
        inference_only_pipelines = []
        for inference_only_pipeline in cfg.inference_only_pipeline:
                inference_only_pipelines.append(inference_only_pipeline)
        inference_only_pipelines = Compose(inference_only_pipelines)
        runner.run(data_loaders, cfg.workflow, inference_only_pipelines, args)
    else:
        runner.run(data_loaders, cfg.workflow)