import type { SafetensorsHeader, TensorData } from './types';
import { parseHeader, headerPrefixSize } from './safetensors-header';

/**
 * 从完整的 safetensors ArrayBuffer 中提取单个张量的数据。
 *
 * @param buffer - 完整的 safetensors 文件内容
 * @param header - 预先解析的头（来自 parseHeader）
 * @param name - 要提取的张量名称
 * @returns 张量数据，包括名称、dtype、形状和原始字节
 * @throws 如果在头中找不到张量名称
 */
export const extractTensor = (
	buffer: ArrayBuffer,
	header: SafetensorsHeader,
	name: string,
): TensorData => {
	const info = header.tensors[name];
	if (!info) {
		throw new Error(
			`Tensor "${name}" not found in header. Available: ${Object.keys(header.tensors).join(', ')}`,
		);
	}

	const [start, end] = info.data_offsets;
	const absoluteStart = header.dataOffset + start;
	const absoluteEnd = header.dataOffset + end;

	if (absoluteEnd > buffer.byteLength) {
		throw new Error(
			`Tensor "${name}" data range [${absoluteStart}, ${absoluteEnd}) exceeds buffer size ${buffer.byteLength}`,
		);
	}

	return {
		name,
		dtype: info.dtype,
		shape: info.shape,
		data: buffer.slice(absoluteStart, absoluteEnd),
	};
};

/**
 * 使用 HTTP Range 请求仅从远程 safetensors 文件中获取头。
 *
 * 会发出两个请求：
 * 1. 获取前 8 个字节以获得头大小
 * 2. 获取完整的头（8 + headerSize 字节）
 *
 * @param url - safetensors 文件的 URL
 * @returns 解析后的头
 */
export const fetchHeader = async (url: string): Promise<SafetensorsHeader> => {
	// 步骤 1：获取 8 字节的长度前缀
	const prefixSize = headerPrefixSize();
	const prefixResp = await fetch(url, {
		headers: { Range: `bytes=0-${prefixSize - 1}` },
	});

	if (!prefixResp.ok && prefixResp.status !== 206) {
		throw new Error(
			`Failed to fetch header prefix: ${prefixResp.status} ${prefixResp.statusText}`,
		);
	}

	const prefixBuf = await prefixResp.arrayBuffer();
	const view = new DataView(prefixBuf);
	const headerSize = view.getUint32(0, true);
	// 检查高 32 位
	if (view.getUint32(4, true) > 0) {
		throw new Error('Header size exceeds 4GB');
	}

	// 步骤 2：获取完整的头（前缀 + JSON）
	const totalHeaderBytes = prefixSize + headerSize;
	const headerResp = await fetch(url, {
		headers: { Range: `bytes=0-${totalHeaderBytes - 1}` },
	});

	if (!headerResp.ok && headerResp.status !== 206) {
		throw new Error(
			`Failed to fetch full header: ${headerResp.status} ${headerResp.statusText}`,
		);
	}

	const headerBuf = await headerResp.arrayBuffer();
	return parseHeader(headerBuf);
};

/**
 * 使用 HTTP Range 请求从远程 safetensors 文件中获取单个张量的数据。
 *
 * @param url - safetensors 文件的 URL
 * @param header - 预先解析的头（来自 fetchHeader）
 * @param name - 要获取的张量名称
 * @returns 张量数据
 */
export const fetchTensor = async (
	url: string,
	header: SafetensorsHeader,
	name: string,
): Promise<TensorData> => {
	const info = header.tensors[name];
	if (!info) {
		throw new Error(
			`Tensor "${name}" not found in header. Available: ${Object.keys(header.tensors).join(', ')}`,
		);
	}

	const [start, end] = info.data_offsets;
	const absoluteStart = header.dataOffset + start;
	const absoluteEnd = header.dataOffset + end;

	const resp = await fetch(url, {
		headers: { Range: `bytes=${absoluteStart}-${absoluteEnd - 1}` },
	});

	if (!resp.ok && resp.status !== 206) {
		throw new Error(
			`Failed to fetch tensor "${name}": ${resp.status} ${resp.statusText}`,
		);
	}

	const data = await resp.arrayBuffer();

	return {
		name,
		dtype: info.dtype,
		shape: info.shape,
		data,
	};
};
