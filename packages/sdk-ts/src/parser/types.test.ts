import { describe, it, expect } from 'vitest';
import { DTYPE_BYTE_SIZE } from '../parser/types';

describe('parser/types', () => {
	it('DTYPE_BYTE_SIZE has correct values', () => {
		expect(DTYPE_BYTE_SIZE.F32).toBe(4);
		expect(DTYPE_BYTE_SIZE.F16).toBe(2);
		expect(DTYPE_BYTE_SIZE.BF16).toBe(2);
		expect(DTYPE_BYTE_SIZE.I32).toBe(4);
		expect(DTYPE_BYTE_SIZE.I64).toBe(8);
		expect(DTYPE_BYTE_SIZE.U8).toBe(1);
	});
});
