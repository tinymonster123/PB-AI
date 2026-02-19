/** WebGPU 上下文，包含设备和队列引用 */
export interface GpuContext {
	adapter: GPUAdapter;
	device: GPUDevice;
	queue: GPUQueue;
	/** 此设备支持的最大缓冲区大小 */
	maxBufferSize: number;
	/** 最大工作组维度 */
	maxWorkgroupSize: [number, number, number];
}

/** 带有相关张量元数据的 GPU 缓冲区 */
export interface TensorBuffer {
	name: string;
	buffer: GPUBuffer;
	/** 元素数量（非字节数） */
	elementCount: number;
	/** 缓冲区的字节大小 */
	byteSize: number;
	/** 是否持有存储为 u16 的 BF16 数据 */
	isBF16: boolean;
}
