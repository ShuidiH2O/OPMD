from modelscope import snapshot_download

model_dir = snapshot_download(
    'Qwen/Qwen3-14B-Base',
    local_dir='/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/Qwen3-14B-Base',
    ignore_patterns=[
        'original/*',
        '*.pth',
        '*.bin'
    ]
)

print(f"Model downloaded to: {model_dir}")