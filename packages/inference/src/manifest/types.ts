/** Manifest 类型定义 */

export type ShardKind = 'embed' | 'layer' | 'lm_head';

export interface Shard {
	id: string;
	kind: ShardKind;
	filename: string;
	bytes: number;
	/** 格式: "blake3:<hex>" */
	hash: string;
	/** 仅 kind=layer 时存在，[start, end] 含两端 */
	layer_range?: [number, number];
}

export interface ModelManifest {
	version: string;
	model_id: string;
	variant: string;
	framework: string;
	dtype: string;
	total_layers: number;
	shards: Shard[];
}
