_base_ = 'runtime_160k.py'
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW',
                   lr=1e-4,
                   betas=(0.9, 0.999),
                   weight_decay=0.05),
    constructor='ViTLearningRateDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, decay_rate=0.9)
)
param_scheduler = [dict(type='LinearLR',
                        start_factor=1e-6,
                        by_epoch=False, begin=0, end=1500),
                   dict(type='PolyLR',
                        eta_min=0.0,
                        power=1.0,
                        begin=1500,
                        end=160000,
                        by_epoch=False)]
