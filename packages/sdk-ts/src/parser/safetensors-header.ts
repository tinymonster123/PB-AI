import type { SafetensorsHeader, TensorInfo, DType } from './types';

/**
 * 读取头长度前缀所需的最小字节数。
 * Safetensors 格式：[8 字节 LE uint64 = header_size] [header_size 字节 JSON] [张量数据...]
 */
const HEADER_LENGTH_PREFIX = 8;

/**
 * 我们接受的最大头大小（256 MB）。Safetensors 的头是 JSON 格式，
 * 即使对于非常大的模型，也应该远低于此限制。
 */
const MAX_HEADER_SIZE = 256 * 1024 * 1024;

/**
 * 从至少包含 8 字节长度前缀和完整 JSON 头的缓冲区中解析 safetensors 头。
 *
 * @param buffer - 至少包含头部分的 ArrayBuffer
 * @returns 解析后的头，包含张量元数据和数据偏移量
 * @throws 如果缓冲区太小或头 JSON 格式错误
 */
export const parseHeader = (buffer: ArrayBuffer): SafetensorsHeader => {
	if (buffer.byteLength < HEADER_LENGTH_PREFIX) {
		throw new Error(
			`Buffer too small: need at least ${HEADER_LENGTH_PREFIX} bytes, got ${buffer.byteLength}`,
		);
	}

	const view = new DataView(buffer);
	// 读取 64 位小端序（LE）的头大小。出于实际目的，我们只使用低 32 位，
	// 因为大于 4GB 的头是不可能的。
	const headerSizeLow = view.getUint32(0, true);
	const headerSizeHigh = view.getUint32(4, true);

	if (headerSizeHigh > 0) {
		throw new Error('Header size exceeds 4GB, which is not supported');
	}

	const headerSize = headerSizeLow;

	if (headerSize > MAX_HEADER_SIZE) {
		throw new Error(
			`Header size ${headerSize} exceeds maximum ${MAX_HEADER_SIZE}`,
		);
	}

	const totalNeeded = HEADER_LENGTH_PREFIX + headerSize;
	if (buffer.byteLength < totalNeeded) {
		throw new Error(
			`Buffer too small for header: need ${totalNeeded} bytes, got ${buffer.byteLength}`,
		);
	}

	const headerBytes = new Uint8Array(buffer, HEADER_LENGTH_PREFIX, headerSize);
	const headerJson = new TextDecoder().decode(headerBytes);

	let parsed: Record<string, unknown>;
	try {
		parsed = JSON.parse(headerJson) as Record<string, unknown>;
	} catch {
		throw new Error('Failed to parse safetensors header JSON');
	}

	const tensors: Record<string, TensorInfo> = {};
	let metadata: Record<string, string> | undefined;

	for (const [key, value] of Object.entries(parsed)) {
		if (key === '__metadata__') {
			metadata = value as Record<string, string>;
			continue;
		}

		const info = value as {
			dtype: string;
			shape: number[];
			data_offsets: [number, number];
		};

		if (!info.dtype || !info.shape || !info.data_offsets) {
			throw new Error(`Invalid tensor info for "${key}"`);
		}

		tensors[key] = {
			dtype: info.dtype as DType,
			shape: info.shape,
			data_offsets: info.data_offsets,
		};
	}

	return {
		headerSize,
		dataOffset: HEADER_LENGTH_PREFIX + headerSize,
		tensors,
		metadata,
	};
};

/**
 * 计算为了读取头而需要获取的最小字节数。
 * 返回 HEADER_LENGTH_PREFIX (8) 用于初始获取以确定头大小。
 */
export const headerPrefixSize = (): number => {
	return HEADER_LENGTH_PREFIX;
};
