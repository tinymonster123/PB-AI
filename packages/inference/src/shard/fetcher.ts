import { del, get, keys, set } from 'idb-keyval';
import type { Shard } from '../manifest/types';

/** 默认并发下载数，对齐浏览器 HTTP/2 实际吞吐量 */
const DEFAULT_CONCURRENCY = 6;

/** 从 hash 字符串中提取纯 hex 值 (去掉 "blake3:" 前缀) */
const hashKey = (hash: string): string => hash.replace(/^blake3:/, '');

/**
 * 从 IndexedDB 获取已缓存的分片 buffer。
 * 命中返回 ArrayBuffer，未命中返回 undefined。
 */
export const getCachedShard = async (
	shard: Shard,
): Promise<ArrayBuffer | undefined> => {
	return get<ArrayBuffer>(hashKey(shard.hash));
};

/** 将分片 buffer 写入 IndexedDB 缓存 (key = hash)。 */
export const cacheShard = async (
	shard: Shard,
	buffer: ArrayBuffer,
): Promise<void> => {
	await set(hashKey(shard.hash), buffer);
};

/** 列出当前缓存中所有的 hash key。 */
export const listCachedKeys = async (): Promise<string[]> => {
	return keys<string>();
};

/** 清除指定 hash 的缓存。 */
export const evictCached = async (hash: string): Promise<void> => {
	await del(hashKey(hash));
};

/**
 * 下载单个分片，优先从 IndexedDB 缓存读取。
 *
 * @param baseUrl 模型文件所在的 base URL
 * @param shard 分片描述
 * @param onProgress 下载进度回调 (0-1)
 */
export const fetchShard = async (
	baseUrl: string,
	shard: Shard,
	onProgress?: (ratio: number) => void,
): Promise<ArrayBuffer> => {
	// 1. 尝试 IndexedDB 缓存
	const cached = await getCachedShard(shard);
	if (cached) {
		onProgress?.(1);
		return cached;
	}

	// 2. HTTP 下载
	const url = `${baseUrl.replace(/\/$/, '')}/${shard.filename}`;
	const res = await fetch(url);
	if (!res.ok) {
		throw new Error(`Failed to fetch shard ${shard.id}: ${res.status}`);
	}

	let buffer: ArrayBuffer;

	if (onProgress && res.body) {
		// 流式读取以报告进度
		const reader = res.body.getReader();
		const chunks: Uint8Array[] = [];
		let received = 0;
		const totalBytes = shard.bytes;

		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			chunks.push(value);
			received += value.byteLength;
			onProgress(totalBytes > 0 ? received / totalBytes : 0);
		}

		// 合并 chunks
		const merged = new Uint8Array(received);
		let offset = 0;
		for (const chunk of chunks) {
			merged.set(chunk, offset);
			offset += chunk.byteLength;
		}
		buffer = merged.buffer;
	} else {
		buffer = await res.arrayBuffer();
		onProgress?.(1);
	}

	// 3. 写入缓存
	await cacheShard(shard, buffer);

	return buffer;
};

/**
 * 并发受限的 Promise 池。
 * 维护一个滑动窗口，最多同时执行 concurrency 个任务，
 * 先完成的立即让出槽位给队列中下一个。
 */
const pooledMap = async <T, R>(
	items: T[],
	fn: (item: T) => Promise<R>,
	concurrency: number,
): Promise<R[]> => {
	const results: R[] = new Array(items.length);
	let nextIdx = 0;

	const worker = async () => {
		while (nextIdx < items.length) {
			const idx = nextIdx++;
			results[idx] = await fn(items[idx]);
		}
	};

	const workers = Array.from(
		{ length: Math.min(concurrency, items.length) },
		() => worker(),
	);
	await Promise.all(workers);
	return results;
};

export interface FetchAllOptions {
	/** 最大并发下载数，默认 6 */
	concurrency?: number;
	/** 单个分片进度回调 */
	onShardProgress?: (shardId: string, ratio: number) => void;
	/** 整体进度回调 (已完成分片数 / 总分片数) */
	onTotalProgress?: (completed: number, total: number) => void;
}

/**
 * 并发受限地下载所有分片，返回按 shard.id 索引的 Map。
 *
 * 浏览器对同一 origin 的并发连接有限 
 *
 * 使用滑动窗口并发池：先完成的分片立即写入 IndexedDB 并释放引用，
 * 再启动队列中下一个，控制内存和网络压力。
 */
export const fetchAllShards = async (
	baseUrl: string,
	shards: Shard[],
	options: FetchAllOptions = {},
): Promise<Map<string, ArrayBuffer>> => {
	const {
		concurrency = DEFAULT_CONCURRENCY,
		onShardProgress,
		onTotalProgress,
	} = options;

	const results = new Map<string, ArrayBuffer>();
	let completed = 0;
	const total = shards.length;

	await pooledMap(
		shards,
		async (shard) => {
			const buffer = await fetchShard(baseUrl, shard, (ratio) => {
				onShardProgress?.(shard.id, ratio);
			});
			results.set(shard.id, buffer);
			completed++;
			onTotalProgress?.(completed, total);
		},
		concurrency,
	);

	return results;
};
