# GCP Detection and ODM Export

本仓库用于无人机影像中的 GCP 红色靶标检测、精定位、全图坐标回投和 ODM GCP 文本生成。

当前文档入口是 `docs/CURRENT_STATUS.md`。新会话接手时先读它，再读 `AGENTS.md`、`docs/ARCHITECTURE.md`、`docs/RUNBOOK.md`、`docs/DECISIONS.md` 和 `docs/SESSION_LOG.md`。

## 项目简介

已确认：项目包含两条可见主链。

- 小图 / ROI fine locator：`detect_gcp_hourglass.py` 从红色 GCP 小图中提取主靶标、数字框和靶标中心点。
- 大图管线：`run_gcp_pipeline.py` 调用 YOLO OBB/SAHI 切片检测，在大图中找 GCP ROI，透视裁剪后调用 fine locator，再把局部点回投到原图全局像素坐标。

另有 `build_odm_gcp_txt.py` 根据控制点实测坐标和大图检测结果生成 ODM 可用的 `images/odm_gcp_list.txt`。

## 项目目标

- 在 `test_images/` 的小图基线上稳定输出 `summary.csv`、`final_result.png`、`debug_overview.png`。
- 在 `images/` 或 `slice/raw_val/` 等大图输入上输出每个 GCP 的 `global_x/global_y`。
- 将 `images/0209GCP.txt` 中的控制点坐标与检测像素坐标合并，生成 `images/odm_gcp_list.txt`。
- 建立可审计的文档和验证记录，避免后续会话遗忘当前状态。

## 核心功能

- 红色掩膜提取、主靶标和数字候选分支分离。
- GCP 中心点定位，当前主方法为 `core_inner_midpoint`，并保留多级 fallback。
- YOLO OBB 切片检测，优先 SAHI，缺少 SAHI 时 fallback 到手工滑窗 Ultralytics 推理。
- OBB 检测框当前只保留置信度 `>= 0.85` 的结果；低于 0.85 的检测不会进入新管线输出、GUI 高可信显示或 GCP TXT 导出。
- 透视 ROI 裁剪和 homography 坐标回投。
- 结果可视化和 CSV/JSON 输出。
- ODM GCP 文本生成。

## 目录结构概览

- `detect_gcp_hourglass.py`：fine locator 主脚本。
- `run_gcp_pipeline.py`：大图管线 CLI 入口。
- `gcp_pipeline.py`：大图管线编排。
- `gcp_sahi_obb_detector.py`：SAHI/Ultralytics 切片 OBB 检测。
- `gcp_crop_mapper.py`：ROI 透视裁剪和坐标映射。
- `gcp_geometry.py`：几何工具函数。
- `gcp_fine_locator.py`：fine locator 子进程适配器。
- `build_odm_gcp_txt.py`：生成 ODM GCP 文本。
- `gui/`：当前主线 PyQt / PySide 桌面 GUI，用于配置路径、运行检测、查看结果和生成 GCP TXT；入口是 `gui/gcp_gui.py`。界面当前为浅色系中文操作台。
- `frontend/`：上一轮本地 Web GUI，可作为历史方案或可选方案保留；当前不再作为主 GUI 主线。
- `tests/`：单元和 smoke 测试。
- `images/`：当前可见大图样例、控制点文件和 ODM 输出。
- `test_images/`：fine locator 小图基线样本。
- `test_results/`：fine locator 历史批处理输出。
- `images_pipeline_output_warped_20260413_164006/`：当前可见的大图管线输出证据。
- `yolov8 obb/`：YOLO OBB 数据集、配置、训练脚本和权重。
- `docx/`：旧项目状态文档，保留为历史依据。
- `docs/`：本轮建立的可续接文档体系。

## 快速开始

项目当前在 Windows 路径 `D:\gcp` 下工作。已确认项目 Conda 环境：

```powershell
D:\Anaconda_envs\envs\gcp\python.exe --version
```

本轮验证结果为 Python `3.10.19`。系统默认 `python` 是 `3.13.9`，不要默认用它跑项目验证。

运行测试：

```powershell
cd D:\gcp
D:\Anaconda_envs\envs\gcp\python.exe -m unittest tests.test_gcp_geometry
D:\Anaconda_envs\envs\gcp\python.exe -m unittest tests.test_gcp_pipeline_smoke
```

运行 fine locator 批处理：

```powershell
cd D:\gcp
D:\Anaconda_envs\envs\gcp\python.exe D:\gcp\detect_gcp_hourglass.py
```

运行大图管线示例：

```powershell
cd D:\gcp
D:\Anaconda_envs\envs\gcp\python.exe D:\gcp\run_gcp_pipeline.py `
  --input D:\gcp\images `
  --weights "D:\gcp\yolov8 obb\runs\train_8cls_obb_s_local_kmpfix\weights\best.pt" `
  --output-dir D:\gcp\images_pipeline_output_warped_YYYYMMDD_HHMMSS `
  --conf 0.85
```

生成 ODM GCP 文本：

```powershell
cd D:\gcp
D:\Anaconda_envs\envs\gcp\python.exe D:\gcp\build_odm_gcp_txt.py `
  --images-dir D:\gcp\images `
  --results-root D:\gcp\images_pipeline_output_warped_20260413_164006 `
  --min-conf 0.85 `
  --review-rejections D:\gcp\images_pipeline_output_warped_20260413_164006\manual_review_rejections.json `
  --output D:\gcp\images\odm_gcp_list.txt
```

`--min-conf 0.85` 是当前高可信检测框保留阈值；低于 0.85 的历史检测不会写入 GCP TXT。`--review-rejections` 是人工复检排除清单；GUI 在 ROI 结果页标记中心错误后会生成 `manual_review_rejections.json`，被标记的 `image_id + det_id` 不会写入 GCP TXT。

启动当前主线 PyQt 桌面操作工作台：

```powershell
cd D:\gcp
D:\Anaconda_envs\envs\gcp\python.exe D:\gcp\gui\gcp_gui.py
```

当前环境已安装并验证 `PyQt5 5.15.11` / Qt `5.15.2`，可直接启动桌面 GUI。无界面自检也可用：

```powershell
cd D:\gcp
D:\Anaconda_envs\envs\gcp\python.exe D:\gcp\gui\gcp_gui.py --check
```

自检已确认默认路径可用、历史样例结果可读、`images/odm_gcp_list.txt` 存在。

启动上一轮 Web GUI（历史 / 可选方案）：

```powershell
cd D:\gcp
D:\Anaconda_envs\envs\gcp\python.exe D:\gcp\frontend\server.py
```

然后在浏览器打开：

```text
http://127.0.0.1:8765/
```

Web GUI 使用 Python 标准库后端，不依赖 Flask/FastAPI；可校验路径、预览命令、启动 `run_gcp_pipeline.py`、流式查看日志、读取结果图和调用 `build_odm_gcp_txt.py`。但当前用户明确要求不要浏览器，因此后续 GUI 开发以 `gui/gcp_gui.py` 为主线。

## 关键依赖

本仓库没有 `requirements.txt`。本轮在 `D:\Anaconda_envs\envs\gcp\python.exe` 中确认：

- Python `3.10.19`
- OpenCV `4.11.0`
- NumPy `2.2.5`
- Ultralytics `8.4.36`
- SAHI `0.11.36`
- PyTorch `2.11.0+cu128`
- PyQt / PySide：已安装并验证 `PyQt5 5.15.11`、Qt `5.15.2`；`PySide6` 和 `PyQt6` 未安装。

还使用 `matplotlib`、`tqdm` 和 Python 标准库。

## 运行方式概览

- `detect_gcp_hourglass.py` 默认读取 `test_images`。目录输入会触发批处理，输出到 `test_results/test_results_vN`。
- `gcp_fine_locator.py` 通过环境变量 `GCP_SINGLE_IMAGE`、`GCP_DISABLE_WINDOWS=1` 调用 `detect_gcp_hourglass.py`，并解析 `SUMMARY|...`。
- `run_gcp_pipeline.py` 默认输入是 `D:\gcp\slice\raw_val`，默认输出是 `D:\gcp\gcp_pipeline_outputs`。
- `run_gcp_pipeline.py` / `gcp_sahi_obb_detector.py` 当前最低保留置信度为 `0.85`；传入更低 `--conf` 会被管线按 `0.85` 处理。
- 注意：`build_odm_gcp_txt.py` 自动发现的是 `images_pipeline_output*` 目录，不会自动发现 `gcp_pipeline_outputs`。运行 ODM 生成时建议显式传 `--results-root`。
- `gui/gcp_gui.py` 是当前主线桌面 GUI 入口；它通过 PyQt/PySide 的 `QProcess` 调用现有 Python 脚本，不走浏览器或 Web 服务。
- GUI 默认置信度阈值为 `0.85`，大图表格、叠加框、统计和 GCP TXT 命令均按该阈值过滤。
- GUI 的 ROI 结果页当前采用批量复检图墙：进入页面即按顺序铺开所有高可信 ROI 的 `final_result.png` 蓝色十字图，不需要先从大图表格选择小图；人工可直接在图墙中将中心错误样本标记为排除，写入 `manual_review_rejections.json`，ODM 导出会跳过这些检测。
- `frontend/server.py` 是上一轮 Web GUI 后端入口，保留为历史 / 可选方案。

## 当前状态入口

- 当前状态快照：`docs/CURRENT_STATUS.md`
- 正式进展汇报：`docs/GCP检测项目进展汇报.md` / `docs/GCP检测项目进展汇报.docx`
- 运行手册：`docs/RUNBOOK.md`
- 架构说明：`docs/ARCHITECTURE.md`
- 关键决策：`docs/DECISIONS.md`
- 会话日志：`docs/SESSION_LOG.md`

## 文档索引

- `AGENTS.md`：后续 Codex 工作规则。
- `docs/PROJECT_OVERVIEW.md`：项目背景、目标、输入输出和总体路线。
- `docs/GCP检测项目进展汇报.md`：本轮自动审计生成的正式项目进展汇报，与 `.docx` 同内容。
- `docs/GCP检测项目进展汇报.docx`：正式 Word 汇报文件。
- `docs/generate_gcp_progress_report.py`：从同一份正文导出 Markdown 和 Word 的报告生成脚本。
- `gui/README.md`：当前 PyQt / PySide 桌面 GUI 的依赖、启动、自检和使用约束。
- `docs/CURRENT_STATUS.md`：当前最可信状态和下一步。
- `docs/ARCHITECTURE.md`：模块职责、调用链、数据流和风险点。
- `docs/RUNBOOK.md`：环境、运行、测试、调试和排障。
- `docs/DECISIONS.md`：关键技术决策和证据。
- `docs/SESSION_LOG.md`：连续工作记录。
- `docs/KNOWN_ISSUES.md`：已知问题和失败历史。
- `docs/VALIDATION.md`：验证命令和结果。
- `docs/GLOSSARY.md`：术语表。
