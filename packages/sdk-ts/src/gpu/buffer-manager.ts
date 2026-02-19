import type { GpuContext, TensorBuffer } from './types';
import type { DType } from '../parser/types';

/**
 * 管理 GPU 缓冲区分配、上传和生命周期。
 *
 * BF16 策略：BF16 数据作为原始 u16 值存储在 GPU 缓冲区中。
 * 着色器在计算时使用 `bitcast<f32>(u32(val) << 16u)` 转换为 F32。
 * 这避免了 CPU 端的转换，并使每个权重的内存使用量保持在 2 字节。
 */
export class BufferManager {
	private readonly ctx: GpuContext;
	private readonly buffers = new Map<string, TensorBuffer>();

	constructor(ctx: GpuContext) {
		this.ctx = ctx;
	}

	/**
	 * 将张量数据上传到 GPU 并注册缓冲区。
	 *
	 * @param name - 唯一的张量名称（用作键）
	 * @param data - 原始张量字节
	 * @param dtype - 张量的数据类型
	 * @param elementCount - 元素数量
	 * @returns 创建的 `TensorBuffer`
	 */
	upload = (
		name: string,
		data: ArrayBuffer,
		dtype: DType,
		elementCount: number,
	): TensorBuffer => {
		// 如果存在同名缓冲区，则先释放
		if (this.buffers.has(name)) {
			this.release(name);
		}

		const byteSize = data.byteLength;
		// 按 4 字节对齐（WebGPU 要求）
		const alignedSize = Math.ceil(byteSize / 4) * 4;

		const buffer = this.ctx.device.createBuffer({
			label: name,
			size: alignedSize,
			usage:
				GPUBufferUsage.STORAGE |
				GPUBufferUsage.COPY_SRC |
				GPUBufferUsage.COPY_DST,
			mappedAtCreation: true,
		});

		const mapped = buffer.getMappedRange();
		new Uint8Array(mapped).set(new Uint8Array(data));
		buffer.unmap();

		const tensorBuffer: TensorBuffer = {
			name,
			buffer,
			elementCount,
			byteSize: alignedSize,
			isBF16: dtype === 'BF16',
		};

		this.buffers.set(name, tensorBuffer);
		return tensorBuffer;
	};

	/**
	 * 创建一个空的 GPU 缓冲区（用于中间结果、KV 缓存等）。
	 *
	 * @param name - 唯一的缓冲区名称
	 * @param byteSize - 字节大小（将按 4 字节对齐）
	 * @param usage - 可选的额外使用标志
	 * @returns 创建的 `TensorBuffer`
	 */
	createEmpty = (
		name: string,
		byteSize: number,
		usage?: GPUBufferUsageFlags,
	): TensorBuffer => {
		if (this.buffers.has(name)) {
			this.release(name);
		}

		const alignedSize = Math.ceil(byteSize / 4) * 4;

		const buffer = this.ctx.device.createBuffer({
			label: name,
			size: alignedSize,
			usage:
				usage ??
				GPUBufferUsage.STORAGE |
					GPUBufferUsage.COPY_SRC |
					GPUBufferUsage.COPY_DST,
		});

		const tensorBuffer: TensorBuffer = {
			name,
			buffer,
			elementCount: alignedSize / 4,
			byteSize: alignedSize,
			isBF16: false,
		};

		this.buffers.set(name, tensorBuffer);
		return tensorBuffer;
	};

	/**
	 * 按名称获取已注册的缓冲区。
	 * @throws 如果未找到缓冲区
	 */
	get = (name: string): TensorBuffer => {
		const buf = this.buffers.get(name);
		if (!buf) {
			throw new Error(
				`Buffer "${name}" not found. Available: ${[...this.buffers.keys()].join(', ')}`,
			);
		}
		return buf;
	};

	/** 检查是否存在具有指定名称的缓冲区。 */
	has = (name: string): boolean => {
		return this.buffers.has(name);
	};

	/** 释放单个缓冲区并将其从注册表中删除。 */
	release = (name: string): void => {
		const buf = this.buffers.get(name);
		if (buf) {
			buf.buffer.destroy();
			this.buffers.delete(name);
		}
	};

	/** 释放所有缓冲区。 */
	releaseAll = (): void => {
		for (const buf of this.buffers.values()) {
			buf.buffer.destroy();
		}
		this.buffers.clear();
	};

	/** 当前分配的缓冲区数量。 */
	get size(): number {
		return this.buffers.size;
	}

	/** 当前使用的 GPU 内存总量（近似值）。 */
	get totalBytes(): number {
		let total = 0;
		for (const buf of this.buffers.values()) {
			total += buf.byteSize;
		}
		return total;
	}

	/**
	 * 将缓冲区数据读回 CPU（用于调试/测试）。
	 * 此操作很慢，不应在生产推理路径中使用。
	 */
	readBack = async (name: string): Promise<ArrayBuffer> => {
		const tensorBuf = this.get(name);
		const staging = this.ctx.device.createBuffer({
			label: `${name}_staging`,
			size: tensorBuf.byteSize,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});

		const encoder = this.ctx.device.createCommandEncoder();
		encoder.copyBufferToBuffer(
			tensorBuf.buffer,
			0,
			staging,
			0,
			tensorBuf.byteSize,
		);
		this.ctx.queue.submit([encoder.finish()]);

		await staging.mapAsync(GPUMapMode.READ);
		const result = staging.getMappedRange().slice(0);
		staging.unmap();
		staging.destroy();

		return result;
	};
}
