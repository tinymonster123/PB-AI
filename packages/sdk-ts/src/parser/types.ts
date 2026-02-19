/** Safetensors 数据类型 */
export type DType = 'F32' | 'F16' | 'BF16' | 'I32' | 'I64' | 'U8';

/** 每种 dtype 每个元素的字节大小 */
export const DTYPE_BYTE_SIZE: Record<DType, number> = {
	F32: 4,
	F16: 2,
	BF16: 2,
	I32: 4,
	I64: 8,
	U8: 1,
};

/** safetensors 文件中单个张量的元数据 */
export interface TensorInfo {
	dtype: DType;
	shape: number[];
	data_offsets: [number, number];
}

/** 解析后的 safetensors 文件头 */
export interface SafetensorsHeader {
	/** JSON 头的字节大小（不包括 8 字节的长度前缀） */
	headerSize: number;
	/** 张量数据开始的字节偏移量 (8 + headerSize) */
	dataOffset: number;
	/** 张量名称到张量元数据的映射 */
	tensors: Record<string, TensorInfo>;
	/** 可选的文件级元数据（例如格式版本） */
	metadata?: Record<string, string>;
}

/** 从 safetensors 文件中提取的张量数据 */
export interface TensorData {
	name: string;
	dtype: DType;
	shape: number[];
	data: ArrayBuffer;
}
