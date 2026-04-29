# AGENTS.md

本文件是后续 Codex 会话进入 `D:\gcp` 后必须优先读取的项目规则。目标读者优先是下一次接手项目的 Codex，其次是人类维护者。

## 项目目标

已确认：本项目围绕无人机影像中的 GCP（Ground Control Point）红色靶标，建立从检测、精定位到 ODM GCP 文本输出的处理链。

最终要解决的问题：

- 在小图或 ROI 中分离红色主靶标和编号数字，输出数字框与靶标中心点。
- 在大图中通过 YOLO OBB/SAHI 切片检测找到 GCP ROI，调用 fine locator，并把 ROI 内坐标回投到原图全局像素坐标。
- 将全图像素坐标与控制点实测坐标合并，生成 ODM 可用的 `odm_gcp_list.txt`。

当前成功标准 / 验收标准：

- Fine locator 批处理在 `test_images` 上生成 `test_results/test_results_vN/summary.csv` 和 `overview_contact_sheet.png`，且可对照历史基线 `test_results_v48`。
- 大图管线为每张输入图输出 `annotated.png`、`detections.csv`、`detections.json`、`rois/<det_id>/...`，成功行必须包含 `global_x/global_y`。
- `build_odm_gcp_txt.py` 能基于控制点文件和大图检测结果生成 `images/odm_gcp_list.txt`。
- 每轮修改必须有静态检查、单元测试或运行结果作为证据；不能仅凭主观判断声明成功。

## 项目范围

当前核心范围：

- Fine locator：`detect_gcp_hourglass.py`。
- 大图检测与回投：`run_gcp_pipeline.py`、`gcp_pipeline.py`、`gcp_sahi_obb_detector.py`、`gcp_crop_mapper.py`、`gcp_geometry.py`、`gcp_fine_locator.py`、`gcp_visualization.py`、`gcp_detection_models.py`。
- ODM 输出：`build_odm_gcp_txt.py`、`images/0209GCP.txt`、`images/odm_gcp_list.txt`。
- 当前主线 GUI：`gui/gcp_gui.py`、`gui/README.md`。这是 PyQt / PySide 桌面应用，不使用浏览器或 Web 服务。
- 历史 / 可选 Web GUI：`frontend/server.py`、`frontend/index.html`、`frontend/styles.css`、`frontend/app.js`、`frontend/README.md`。仅在用户明确接受浏览器方案时继续维护。
- 训练/数据准备辅助：`create_empty_yolo_labels.py`、`slice/slice_images.py`、`slice/split_raw_images.py`、`yolov8 obb/gcp.yaml`、`yolov8 obb/launch_local_train_8cls_obb_s.ps1`。
- 测试：`tests/test_gcp_geometry.py`、`tests/test_gcp_pipeline_smoke.py`。
- 结果证据：`test_results/`、`images_pipeline_output_warped_20260413_164006/`、`yolov8 obb/runs/train_8cls_obb_s_local_kmpfix/`。

目前不在核心范围内：

- 重新训练 YOLO 模型，除非任务明确要求。
- 清理历史输出目录、缓存、旧实验结果。
- 重写 fine locator 算法主链，除非先确认当前失败样本和对照基线。
- 将 `useless/` 下历史脚本作为当前主链依据；它只能作为历史参考。

## 不可违反的约束

- 不允许跳过验证就宣称成功。
- 不允许删除 `test_results/`、`images_pipeline_output*`、`yolov8 obb/runs/`、`images/odm_gcp_list.txt` 等历史证据，除非用户明确要求。
- 不允许编造未验证结论；必须标注“已确认”“待确认”或“推测 / 尚未验证”。
- 不允许只改代码不更新文档；每轮结束前至少更新 `docs/CURRENT_STATUS.md` 和 `docs/SESSION_LOG.md`。
- 不允许忽略已有文档直接重做；旧文档位于 `docx/PROJECT_STATUS.md` 和 `docx/README_GCP_PIPELINE.md`，需要作为历史材料核对。
- 不允许随意改变以下接口/格式，除非同步更新调用方和文档：
  - `detect_gcp_hourglass.py` 的机器摘要行：`SUMMARY|bbox=...|point=...|method=...|used_fallback=...`
  - fine locator 输出文件名：`number_crop.jpg`、`final_result.png`、`debug_overview.png`、`hourglass_union_mask.png` 等。
  - 批处理输出：`test_results/test_results_vN/summary.csv`、`overview_contact_sheet.png`。
  - 大图管线输出：`detections.csv`、`detections.json`、`annotated.png`、`rois/<det_id>/fine_locator/run.log`。
  - 人工复检排除清单：`manual_review_rejections.json`，用于记录不应进入 GCP TXT 的 `image_id + det_id`；不能用物理删除历史图像替代该清单。
  - OBB 检测框最低保留置信度：当前为 `0.85`；新管线、GUI 高可信显示和 GCP TXT 导出都必须按 `conf >= 0.85` 过滤。不得只改 GUI 默认值而放任低置信度行进入 CSV/TXT。
  - YOLO OBB 数据集格式：`yolov8 obb/dataset_gcp/images/<train|val>` 和 `labels/<train|val>`，标签行为 `class x1 y1 x2 y2 x3 y3 x4 y4` 归一化坐标。

## 每次开始任务前必须先读

- `docs/CURRENT_STATUS.md`
- `docs/DECISIONS.md`
- `docs/SESSION_LOG.md`
- `docs/ARCHITECTURE.md`
- `docs/RUNBOOK.md`
- 与当前任务相关的核心代码文件。
- 如涉及 fine locator，必须读取 `detect_gcp_hourglass.py` 中中心定位函数链和最近结果目录。
- 如涉及大图管线，必须读取 `gcp_pipeline.py`、`gcp_sahi_obb_detector.py`、`gcp_crop_mapper.py`、`gcp_fine_locator.py`。
- 如涉及 ODM 输出，必须读取 `build_odm_gcp_txt.py`、`images/0209GCP.txt`、最近 `images_pipeline_output*/*/detections.csv`。
- 如涉及桌面 GUI，必须读取 `gui/README.md`、`gui/gcp_gui.py`、最近 `images_pipeline_output_warped_20260413_164006/*/detections.csv`、`run_gcp_pipeline.py` 和 `build_odm_gcp_txt.py`。
- 如涉及历史 Web GUI，必须读取 `frontend/README.md`、`frontend/server.py`、`frontend/app.js`、最近 `images_pipeline_output_warped_20260413_164006/*/detections.csv` 和 `build_odm_gcp_txt.py`。

## 每次结束任务前必须更新

- `docs/CURRENT_STATUS.md`：更新阶段、当前问题、风险、下一步。
- `docs/SESSION_LOG.md`：追加本轮目标、阅读文件、修改文件、命令、观察、结论、未解决项。
- `docs/DECISIONS.md`：如有新技术判断、放弃方案、约束变化或经验确认，追加/修订决策。
- `docs/ARCHITECTURE.md`：如模块结构、调用链、输出格式或耦合点变化。
- `docs/RUNBOOK.md`：如运行方式、环境、命令、验证方法变化。
- `README.md`：如入口命令、依赖、状态入口变化。

## 输出要求

每次最终回复必须明确说明：

- 本轮实际修改了什么。
- 哪些结论已验证，验证命令和结果是什么。
- 哪些事项仍待验证。
- 当前阻塞点或风险。
- 下一步建议。

文档中必须明确区分：

- 已确认：来自代码、配置、测试、结果文件或实际命令输出。
- 待确认：缺少运行或样本证据。
- 推测 / 尚未验证：基于阅读或局部现象的判断。

## 禁止事项

- 禁止重复尝试已被明确否定的方案，除非出现新证据。
- 禁止遗漏失败记录；失败样本、异常、回退、跳过测试都必须写入 `docs/SESSION_LOG.md` 或 `docs/KNOWN_ISSUES.md`。
- 禁止模糊表达“应该可以”“大概没问题”而不标注状态。
- 禁止忽略已有文档、历史结果或对照基线。
- 禁止为了让结果变好看而删除或隐藏失败行。
- 禁止把方法名一致误判为几何语义完全正确；必须结合坐标、图像和基线误差判断。
- 禁止在用户明确要求 PyQt 桌面 GUI 时改回浏览器、Web 前端、React/Vite、FastAPI 或 Flask 方案。
- 禁止在当前 PyQt GUI 主线中随意改回深色英文界面；除非用户明确要求，新增 GUI 文案应使用中文，视觉风格应保持浅色工程软件风格。
- 禁止为了人工复检而删除原始检测图片、CSV、JSON 或 ROI artifacts；中心错误样本应写入 `manual_review_rejections.json`，并由 ODM 导出阶段跳过。
- 禁止把低于 `0.85` 的 OBB 检测框当作当前高可信结果写入 GCP TXT；如果为了召回率临时降低阈值，必须明确标注实验性质并更新决策和验证记录。
- 禁止把 ROI 复检界面退回到只能先选单张小图再查看的模式；当前 PyQt GUI 主线要求 ROI 页直接展示批量 `final_result.png` 图墙，方便人工一次性复检多张蓝色十字图。
