# Hero MP4 生成（Open3D）

该目录提供一个默认使用 Open3D 的四幕式 Hero Video 生成脚本，不依赖 Matplotlib。

## 脚本

- `make_hero_mp4_open3d.py`

## 输入要求

脚本需要 1 个原视频 + 3 个缓存文件：

- `*_sam2.npz`：使用字段 `masks`
- `*_cotracker.npz`：使用字段 `tracks`、`visibility`
- `*_mega_sam.npz`：使用字段 `pointmaps`

以上字段与 `src/pdi_eval/pipeline.py` 当前缓存格式一致。

## 示例命令

```bash
python show/make_hero_mp4_open3d.py --video data/your_video.mp4 --sam2-cache output/cache/sora/your_video_sam2.npz --track-cache output/cache/sora/your_video_cotracker.npz --megasam-cache output/cache/sora/your_video_mega_sam.npz --output show/your_video_hero.mp4 --fps 30
```

## 输出效果（四幕）

1. `0s-2s` 原始 AI 视频（2D 视觉）
2. `2s-4s` SAM2 语义锁定（暗场 + mask glow + 扫描线）
3. `4s-7s` 3D 点云升维（Open3D 轨道运镜）
4. `7s-12s` 锚点+骨架刚性审计（异常连线红色报警）

## 可调参数

- `--duration-act1/2/3/4`：四幕时长（秒）
- `--max-fg-points`：前景点云采样上限
- `--max-bg-points`：背景点云采样上限
- `--anchor-count`：刚性审计锚点数量（默认 30）
