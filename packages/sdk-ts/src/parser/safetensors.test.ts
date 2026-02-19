import { describe, it, expect } from 'vitest';
import { parseHeader, headerPrefixSize } from '../parser/safetensors-header';
import { extractTensor } from '../parser/safetensors-reader';

/**
 * Build a minimal safetensors file in memory.
 *
 * Format: [8 bytes LE uint64 headerSize] [JSON header] [tensor data]
 */
const buildSafetensors = (
	tensors: Record<string, { dtype: string; shape: number[]; data: Uint8Array }>,
): ArrayBuffer => {
	// Build header JSON and compute data offsets
	let currentOffset = 0;
	const headerObj: Record<
		string,
		{ dtype: string; shape: number[]; data_offsets: [number, number] }
	> = {};
	const dataChunks: Uint8Array[] = [];

	for (const [name, info] of Object.entries(tensors)) {
		const start = currentOffset;
		const end = start + info.data.byteLength;
		headerObj[name] = {
			dtype: info.dtype,
			shape: info.shape,
			data_offsets: [start, end],
		};
		dataChunks.push(info.data);
		currentOffset = end;
	}

	const headerJson = JSON.stringify(headerObj);
	const headerBytes = new TextEncoder().encode(headerJson);
	const headerSize = headerBytes.length;

	// Build full buffer: 8 bytes prefix + header + data
	const totalSize =
		8 + headerSize + dataChunks.reduce((s, c) => s + c.byteLength, 0);
	const buffer = new ArrayBuffer(totalSize);
	const view = new DataView(buffer);
	const uint8 = new Uint8Array(buffer);

	// Write header size as 64-bit LE
	view.setUint32(0, headerSize, true);
	view.setUint32(4, 0, true);

	// Write header JSON
	uint8.set(headerBytes, 8);

	// Write tensor data
	let dataOffset = 8 + headerSize;
	for (const chunk of dataChunks) {
		uint8.set(chunk, dataOffset);
		dataOffset += chunk.byteLength;
	}

	return buffer;
};

describe('safetensors-header', () => {
	it('headerPrefixSize returns 8', () => {
		expect(headerPrefixSize()).toBe(8);
	});

	it('parses a single F32 tensor header', () => {
		// 3 floats = 12 bytes
		const data = new Uint8Array(12);
		const buf = buildSafetensors({
			'test.weight': { dtype: 'F32', shape: [3], data },
		});

		const header = parseHeader(buf);

		expect(header.dataOffset).toBe(8 + header.headerSize);
		expect(Object.keys(header.tensors)).toEqual(['test.weight']);
		expect(header.tensors['test.weight'].dtype).toBe('F32');
		expect(header.tensors['test.weight'].shape).toEqual([3]);
		expect(header.tensors['test.weight'].data_offsets).toEqual([0, 12]);
	});

	it('parses multiple tensors', () => {
		const buf = buildSafetensors({
			'layer.0.weight': {
				dtype: 'F32',
				shape: [2, 3],
				data: new Uint8Array(24),
			},
			'layer.0.bias': {
				dtype: 'F32',
				shape: [3],
				data: new Uint8Array(12),
			},
		});

		const header = parseHeader(buf);

		expect(Object.keys(header.tensors).sort()).toEqual([
			'layer.0.bias',
			'layer.0.weight',
		]);
		expect(header.tensors['layer.0.weight'].shape).toEqual([2, 3]);
		expect(header.tensors['layer.0.bias'].shape).toEqual([3]);
	});

	it('parses BF16 dtype', () => {
		// 4 BF16 elements = 8 bytes
		const buf = buildSafetensors({
			weight: { dtype: 'BF16', shape: [4], data: new Uint8Array(8) },
		});

		const header = parseHeader(buf);
		expect(header.tensors['weight'].dtype).toBe('BF16');
	});

	it('handles __metadata__ field', () => {
		// Build manually with metadata
		const headerObj = {
			__metadata__: { format: 'pt' },
			weight: { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
		};
		const headerJson = JSON.stringify(headerObj);
		const headerBytes = new TextEncoder().encode(headerJson);

		const buf = new ArrayBuffer(8 + headerBytes.length + 8);
		const view = new DataView(buf);
		view.setUint32(0, headerBytes.length, true);
		view.setUint32(4, 0, true);
		new Uint8Array(buf).set(headerBytes, 8);

		const header = parseHeader(buf);
		expect(header.metadata).toEqual({ format: 'pt' });
		expect(header.tensors['weight']).toBeDefined();
		expect(header.tensors['__metadata__']).toBeUndefined();
	});

	it('throws on buffer too small', () => {
		expect(() => parseHeader(new ArrayBuffer(4))).toThrow('Buffer too small');
	});

	it('throws on header larger than buffer', () => {
		const buf = new ArrayBuffer(8);
		const view = new DataView(buf);
		view.setUint32(0, 1000, true); // header says 1000 bytes but buf is only 8
		view.setUint32(4, 0, true);

		expect(() => parseHeader(buf)).toThrow('Buffer too small for header');
	});
});

describe('extractTensor', () => {
	it('extracts tensor data correctly', () => {
		// Create a tensor with known float values
		const floats = new Float32Array([1.0, 2.0, 3.0]);
		const data = new Uint8Array(floats.buffer);

		const buf = buildSafetensors({
			weight: { dtype: 'F32', shape: [3], data },
		});

		const header = parseHeader(buf);
		const tensor = extractTensor(buf, header, 'weight');

		expect(tensor.name).toBe('weight');
		expect(tensor.dtype).toBe('F32');
		expect(tensor.shape).toEqual([3]);

		const result = new Float32Array(tensor.data);
		expect(result[0]).toBeCloseTo(1.0);
		expect(result[1]).toBeCloseTo(2.0);
		expect(result[2]).toBeCloseTo(3.0);
	});

	it('extracts second tensor from multi-tensor file', () => {
		const w1 = new Float32Array([1.0, 2.0]);
		const w2 = new Float32Array([3.0, 4.0, 5.0]);

		const buf = buildSafetensors({
			first: { dtype: 'F32', shape: [2], data: new Uint8Array(w1.buffer) },
			second: { dtype: 'F32', shape: [3], data: new Uint8Array(w2.buffer) },
		});

		const header = parseHeader(buf);
		const tensor = extractTensor(buf, header, 'second');

		expect(tensor.shape).toEqual([3]);
		const result = new Float32Array(tensor.data);
		expect(result[0]).toBeCloseTo(3.0);
		expect(result[1]).toBeCloseTo(4.0);
		expect(result[2]).toBeCloseTo(5.0);
	});

	it('throws on unknown tensor name', () => {
		const buf = buildSafetensors({
			weight: { dtype: 'F32', shape: [1], data: new Uint8Array(4) },
		});
		const header = parseHeader(buf);

		expect(() => extractTensor(buf, header, 'nonexistent')).toThrow(
			'not found',
		);
	});
});
