batch_size = 1
crop_size = (
    1024,
    1024,
)
custom_hooks = [
    dict(type='SimpleTimeLoggerHook'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'models',
        'mmdet.models',
    ])
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=0,
    size=(
        1024,
        1024,
    ),
    size_divisor=None,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = 'data/sam-hq'
dataset = dict(
    data_root='data/sam-hq',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadExtHQSeg44kTrainAnnotations'),
        dict(keep_ratio=True, scale=(
            1024,
            1024,
        ), type='Resize'),
        dict(prob=0.5, type='RandomFlip'),
        dict(type='PhotoMetricDistortion'),
        dict(type='InterSegPackSegInputs'),
    ],
    type='ExtHQSeg44kTrainDataset')
dataset_type = 'ExtHQSeg44kTrainDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, type='CheckpointHook'),
    logger=dict(interval=200, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, window_size=1000)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        depth=12,
        downsample_sizes=[
            512,
        ],
        drop_path_rate=0.0,
        drop_rate=0.0,
        embed_dim=768,
        extend_ratio=3,
        final_embed_dim=256,
        img_size=224,
        in_dim=3,
        mlp_ratio=4.0,
        num_heads=12,
        out_indices=(
            2,
            5,
            8,
            11,
        ),
        patch_size=16,
        state_dim=32,
        type='HRSAMPlusPlusViT',
        use_checkpoint=False,
        window_size=16),
    decode_head=dict(
        align_corners=False,
        attn_cfg=dict(depth=2, mlp_dim=2048, num_heads=8),
        in_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        loss_decode=[
            dict(loss_weight=1.0, type='NormalizedFocalLoss'),
            dict(type='BinaryIoU'),
        ],
        num_multimask_outputs=3,
        type='SAMDecoder'),
    freeze_backbone=False,
    freeze_decode_head=False,
    freeze_neck=False,
    image_embed_loader=None,
    init_cfg=dict(
        checkpoint=
        'work_dirs/hrsam/coco_lvis/simdist_hrsam_plusplus_colaug_1024x1024_bs1_160k/iter_160000.pth',
        type='Pretrained'),
    neck=dict(
        embed_dim=256,
        image_embed_size=(
            64,
            64,
        ),
        input_image_size=(
            1024,
            1024,
        ),
        mask_in_dim=16,
        type='SAMPromptEncoder'),
    remove_backbone=False,
    test_cfg=dict(target_size=1024),
    train_cfg=dict(
        gamma=0.6,
        interact_params=dict(
            coco=dict(gamma=0.6, refine_gamma=0.6),
            lvis=dict(gamma=0.9, refine_gamma=0.35)),
        max_num_clicks=20,
        sfc_inner_k=1.7,
        target_image_size=1024),
    type='PromptSegmentor')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        ignore_keys=[
            'mamba',
            'pos_embed',
            'patch_embed',
        ], num_layers=12),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=40000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(seed=42)
resume = False
size = (
    1024,
    1024,
)
target_size = 1024
test_cfg = None
train_cfg = dict(
    max_iters=40000, type='IterBasedTrainLoop', val_interval=10000)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root='data/sam-hq',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadExtHQSeg44kTrainAnnotations'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='InterSegPackSegInputs'),
        ],
        type='ExtHQSeg44kTrainDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadExtHQSeg44kTrainAnnotations'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='InterSegPackSegInputs'),
]
val_cfg = None
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = None
work_dir = './work_dirs/hrsam/hqseg44k/hrsam_plusplus_simaug_1024x1024_bs1_40k'
