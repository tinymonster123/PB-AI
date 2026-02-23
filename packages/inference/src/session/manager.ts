import * as ort from 'onnxruntime-web';
import type { ModelManifest } from '../manifest/types';

export interface SessionOptions {
	/** 优先使用 WebGPU，不支持时 fallback 到 WASM */
	preferWebGPU?: boolean;
}

/**
 * 从下载好的分片 buffers 拼装 ONNX model 并创建 InferenceSession。
 *
 * 流程:
 * 1. fetch model.onnx (图定义，~1-2MB)
 * 2. 通过 externalData 选项将分片 buffer 传给 ORT
 * 3. 创建 InferenceSession
 *
 * @param baseUrl 模型文件所在的 base URL
 * @param manifest 模型 manifest
 * @param shardBuffers 分片 id → ArrayBuffer 的映射
 * @param options session 选项
 */
export const createSession = async (
	baseUrl: string,
	manifest: ModelManifest,
	shardBuffers: Map<string, ArrayBuffer>,
	options: SessionOptions = { preferWebGPU: true },
): Promise<ort.InferenceSession> => {
	const { preferWebGPU = true } = options;

	// 1. 下载 model.onnx 图定义
	const modelUrl = `${baseUrl.replace(/\/$/, '')}/model.onnx`;
	const modelRes = await fetch(modelUrl);
	if (!modelRes.ok) {
		throw new Error(`Failed to fetch model.onnx: ${modelRes.status}`);
	}
	const modelBuffer = await modelRes.arrayBuffer();

	// 2. 选择执行后端
	const executionProviders: ort.InferenceSession.ExecutionProviderConfig[] =
		preferWebGPU ? ['webgpu', 'wasm'] : ['wasm'];

	// 3. 创建 session，通过 externalData 传入分片
	const session = await ort.InferenceSession.create(modelBuffer, {
		executionProviders,
		graphOptimizationLevel: 'all',
		externalData: manifest.shards.map((s) => {
			const data = shardBuffers.get(s.id);
			if (!data) {
				throw new Error(`Missing shard buffer: ${s.id}`);
			}
			return { path: s.filename, data };
		}),
	});

	return session;
};

/** 销毁 session 并释放资源。 */
export const destroySession = async (
	session: ort.InferenceSession,
): Promise<void> => {
	await session.release();
};
