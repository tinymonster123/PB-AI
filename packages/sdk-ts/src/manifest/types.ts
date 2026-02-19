/**
 * `crates/manifest-core/src/lib.rs` 类型的 TypeScript 镜像。
 * 这些类型描述了由 pb-sharder 生成的模型分块清单。
 */

/** 单个模型块的元数据（例如 "base", "layers_0-3"） */
export interface ManifestChunk {
	/** 唯一的块标识符（例如 "base", "layers_0-3"） */
	id: string;
	/** 输出文件名（例如 "base.safetensors"）。可能为空。 */
	filename?: string;
	/** 起始层索引（基块使用 0） */
	layer_start: number;
	/** 结束层索引（基块使用 0） */
	layer_end: number;
	/** 文件大小（字节） */
	bytes: number;
	/** SHA-256 校验和的十六进制字符串 */
	sha256: string;
	/** 远程下载 URL（由上传工具填写）。可能为空。 */
	url?: string;
}

/** 描述完整分块结构的模型清单 */
export interface ModelManifest {
	/** 模型标识符（例如 "Qwen/Qwen2.5-3B"） */
	model_id: string;
	/** 清单版本 */
	version: string;
	/** 权重数据类型（例如 "bf16", "fp16", "auto"） */
	dtype: string;
	/** 推理所需的最少层数 */
	min_runnable_depth: number;
	/** 有序的块列表 */
	chunks: ManifestChunk[];
}

/**
 * 验证清单对象。
 * @throws 如果清单无效
 */
export const validateManifest = (manifest: ModelManifest): void => {
	if (!manifest.chunks || manifest.chunks.length === 0) {
		throw new Error('Manifest chunks cannot be empty');
	}
	if (!manifest.min_runnable_depth || manifest.min_runnable_depth <= 0) {
		throw new Error('min_runnable_depth must be > 0');
	}
};

/**
 * 从 JSON 解析并验证清单。
 */
export const parseManifest = (json: string): ModelManifest => {
	const manifest = JSON.parse(json) as ModelManifest;
	validateManifest(manifest);
	return manifest;
};
