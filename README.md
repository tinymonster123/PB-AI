# PB-AI Monorepo (Nx)

本仓库采用 `Nx + pnpm + Rust workspace`，用于实现浏览器端渐进式模型加载与运行时。

## 目录结构

-- `apps/web`：浏览器最小演示（React + Vite）
- `packages/sdk-ts`：TypeScript Runtime SDK
- `crates/manifest-core`：Rust manifest 结构与校验
- `crates/model-splitter`：Rust 模型分块 CLI
- `specs`：manifest 协议草案

## 安装与启动

1. 安装 JS 依赖
   - `pnpm install`
2. 启动 Web
   - `pnpm nx run web:dev`
3. 运行 Rust splitter 帮助
   - `pnpm nx run model-splitter:split-help`
4. 生成示例 manifest
   - `pnpm nx run model-splitter:split-sample`

## 常用命令

-- `pnpm dev`：等价 `nx run web:dev`
- `pnpm build`：构建全部项目
- `pnpm typecheck`：类型检查
- `pnpm lint`：运行 lint 目标（当前为占位或 clippy）
