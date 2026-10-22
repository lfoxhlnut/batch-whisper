# AGENTS.md

永远使用中文回答

## 项目概述

这是一个小型的 Python 命令行工具包（由 uv 管理，`requires-python >= 3.12`），通过 OpenAI Whisper 将视频/音频文件转换为中文字幕，然后将 SRT 文件合并为段落格式的 TXT 文件。

- `run_whisper.py` — 单文件流水线：ffmpeg（16 kHz 单声道 WAV）→ `whisper` 命令行工具 → SRT + TXT → 清理。
- `merge_srt.py` — SRT 转换为 `<name>_merged.txt`，按 30 秒窗口分组条目。
- `batch_process_videos.py` — 批量处理目录中的文件（通过子进程调用上述两个脚本）。
- `batch_process_all.py` — 递归批量处理根目录下的所有文件（调用批处理脚本）。

## 命令

始终通过 uv 环境运行 Python：

```bash
uv run python run_whisper.py <file> [model]   # 默认模型：turbo
uv run python merge_srt.py <srt_path>
uv run python batch_process_videos.py <dir> [model]
uv run python batch_process_all.py <root> [model]
```
```

- 使用 `uv add <pkg>` / `uv remove <pkg>` 添加/移除依赖（由 uv 管理；参见 `uv.lock`）。
- 运行时需要将 `ffmpeg` 和 `whisper` 命令行工具添加到 PATH 中；若缺少任一工具，脚本将提前退出。
- 依赖较重：`torch`、`openai-whisper`；首次运行时会下载 Whisper 模型。

## 注意事项

- `run_whisper copy.py` 是脚本的过时副本——请始终使用 `run_whisper.py` 编辑和运行。
- 处理过程中，包含空格的输入文件名会被重命名（去除空格），处理完成后恢复原名。
- 生成的 SRT/TXT 文件在生成后会重新重命名为原始文件名。
- 如果输入文件旁边已存在 SRT 文件，则跳过处理（按原始文件名键值匹配）。
- 暂存的 WAV 文件会与输入文件同级创建，并在 `finally` 块中自动清理；
生成的 `vtt`、`tsv`、`json` 文件也会被删除。

## 测试

- 目前没有自动化测试、pytest、代码检查、类型检查、CI 或 pre-commit 配置。
- `test/` 目录包含媒体测试用例（如 `input.mp4`、`ff gg.mp4`）；`test-original/` 目录存放备份副本。

## 规范

- 遵循 中文回复；英文代码注释；
遵循 PEP 8 标准，使用类型提示和文档字符串；提交信息使用中文；保持 `README.md` 更新。
- `README.md` 目前为空；