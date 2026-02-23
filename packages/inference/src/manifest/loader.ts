import type { ModelManifest } from './types';

/**
 * 从 URL 加载并解析 manifest.json。
 *
 * @param baseUrl 模型文件所在的 base URL（不含 manifest.json）
 */
export const loadManifest = async (baseUrl: string): Promise<ModelManifest> => {
	const url = `${baseUrl.replace(/\/$/, '')}/manifest.json`;
	const res = await fetch(url);
	if (!res.ok) {
		throw new Error(`Failed to load manifest: ${res.status} ${res.statusText}`);
	}
	const manifest: ModelManifest = await res.json();

	if (manifest.version !== '0.2') {
		throw new Error(`Unsupported manifest version: ${manifest.version}`);
	}

	return manifest;
};
