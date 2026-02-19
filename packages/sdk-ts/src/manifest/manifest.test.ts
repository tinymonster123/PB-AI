import { describe, it, expect } from 'vitest';
import { validateManifest, parseManifest } from '../manifest/types';
import type { ModelManifest } from '../manifest/types';

const sampleManifest = (): ModelManifest => {
	return {
		model_id: 'Qwen/Qwen2.5-3B',
		version: '1.0.0',
		dtype: 'bf16',
		min_runnable_depth: 4,
		chunks: [
			{
				id: 'base',
				filename: 'base.safetensors',
				layer_start: 0,
				layer_end: 0,
				bytes: 1024,
				sha256: 'abc123',
			},
			{
				id: 'layers_0-3',
				filename: 'layers_0-3.safetensors',
				layer_start: 0,
				layer_end: 3,
				bytes: 4096,
				sha256: 'def456',
			},
		],
	};
};

describe('manifest types', () => {
	it('validates a correct manifest', () => {
		expect(() => validateManifest(sampleManifest())).not.toThrow();
	});

	it('rejects empty chunks', () => {
		const m = sampleManifest();
		m.chunks = [];
		expect(() => validateManifest(m)).toThrow('chunks cannot be empty');
	});

	it('rejects zero min_runnable_depth', () => {
		const m = sampleManifest();
		m.min_runnable_depth = 0;
		expect(() => validateManifest(m)).toThrow('min_runnable_depth must be > 0');
	});

	it('parseManifest round-trips through JSON', () => {
		const original = sampleManifest();
		const json = JSON.stringify(original);
		const parsed = parseManifest(json);

		expect(parsed.model_id).toBe('Qwen/Qwen2.5-3B');
		expect(parsed.chunks.length).toBe(2);
		expect(parsed.chunks[0].id).toBe('base');
		expect(parsed.chunks[1].layer_end).toBe(3);
	});

	it('parseManifest throws on invalid JSON', () => {
		expect(() => parseManifest('{')).toThrow();
	});

	it('optional fields can be omitted', () => {
		const manifest: ModelManifest = {
			model_id: 'test',
			version: '1.0',
			dtype: 'fp16',
			min_runnable_depth: 1,
			chunks: [
				{
					id: 'base',
					layer_start: 0,
					layer_end: 0,
					bytes: 100,
					sha256: 'abc',
				},
			],
		};
		expect(() => validateManifest(manifest)).not.toThrow();
		expect(manifest.chunks[0].filename).toBeUndefined();
		expect(manifest.chunks[0].url).toBeUndefined();
	});
});
