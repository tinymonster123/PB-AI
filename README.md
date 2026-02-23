# PB-AI Monorepo (Nx)

本仓库采用 `Nx + pnpm + Rust workspace`，用于实现浏览器端渐进式模型加载与运行时。

## 目录结构

-- `apps/web`：浏览器最小演示（React + Vite）
- `tools/onnx-sharder`：ONNX 模型按层分片工具（Python）
- `tools/quantization`：模型量化工具（Python）
- `specs`：manifest 协议草案（v0.2）

## 安装与启动

1. 安装 JS 依赖
   - `pnpm install`
2. 启动 Web
   - `pnpm nx run web:dev`

## 常用命令

-- `pnpm dev`：等价 `nx run web:dev`
- `pnpm build`：构建全部项目
- `pnpm typecheck`：类型检查
