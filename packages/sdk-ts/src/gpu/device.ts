import type { GpuContext } from './types';

/**
 * 初始化 WebGPU：请求适配器和设备。
 *
 * @param options - 可选的功耗偏好和所需功能
 * @returns 包含适配器、设备和队列的 GpuContext
 * @throws 如果 WebGPU 不可用或设备创建失败
 */
export const initGpu = async (options?: {
	powerPreference?: GPUPowerPreference;
}): Promise<GpuContext> => {
	if (typeof navigator === 'undefined' || !navigator.gpu) {
		throw new Error(
			'WebGPU is not available. Ensure you are using a compatible browser.',
		);
	}

	const adapter = await navigator.gpu.requestAdapter({
		powerPreference: options?.powerPreference ?? 'high-performance',
	});

	if (!adapter) {
		throw new Error(
			'Failed to get GPUAdapter. WebGPU may not be supported on this device.',
		);
	}

	const requiredLimits: Record<string, number> = {};

	// 从适配器请求最大缓冲区大小
	const adapterLimits = adapter.limits;
	if (adapterLimits.maxBufferSize) {
		requiredLimits['maxBufferSize'] = adapterLimits.maxBufferSize;
	}
	if (adapterLimits.maxStorageBufferBindingSize) {
		requiredLimits['maxStorageBufferBindingSize'] =
			adapterLimits.maxStorageBufferBindingSize;
	}

	const device = await adapter.requestDevice({
		requiredLimits,
	});

	device.lost.then((info) => {
		console.error(`WebGPU device lost: ${info.reason}`, info.message);
	});

	return {
		adapter,
		device,
		queue: device.queue,
		maxBufferSize: device.limits.maxBufferSize,
		maxWorkgroupSize: [
			device.limits.maxComputeWorkgroupSizeX,
			device.limits.maxComputeWorkgroupSizeY,
			device.limits.maxComputeWorkgroupSizeZ,
		],
	};
};
