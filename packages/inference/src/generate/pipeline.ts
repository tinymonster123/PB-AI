import {
	AutoTokenizer,
	type PreTrainedTokenizer,
} from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';

export interface GenerateOptions {
	maxNewTokens?: number;
	temperature?: number;
	topK?: number;
	/** 每生成一个 token 的回调，用于流式输出 */
	onToken?: (token: string) => void;
}

export interface TextGenerator {
	generate: (prompt: string, options?: GenerateOptions) => Promise<string>;
	tokenizer: PreTrainedTokenizer;
	dispose: () => Promise<void>;
}

/**
 * 从 HuggingFace tokenizer 文件和 ORT session 创建文本生成器。
 *
 * @param baseUrl 模型文件所在的 base URL（含 tokenizer.json）
 * @param session 已创建的 ORT InferenceSession
 * @param modelId HuggingFace 模型 ID（用于加载 tokenizer）
 */
export const createGenerator = async (
	_baseUrl: string,
	session: ort.InferenceSession,
	modelId: string,
): Promise<TextGenerator> => {
	const tokenizer = await AutoTokenizer.from_pretrained(modelId, {
		local_files_only: false,
	});

	const generate = async (
		prompt: string,
		options: GenerateOptions = {},
	): Promise<string> => {
		const {
			maxNewTokens = 128,
			temperature = 0.7,
			topK = 40,
			onToken,
		} = options;

		const encoded = tokenizer(prompt, { return_tensors: 'ort' });
		let inputIds = encoded.input_ids.data as BigInt64Array;
		let attentionMask = encoded.attention_mask.data as BigInt64Array;

		const generatedTokens: bigint[] = [];
		const eosTokenId = BigInt(tokenizer.eos_token_id ?? 2);

		// js-cache-property-access: vocabSize 在循环中不变，首次推理后缓存
		let vocabSize = 0;

		for (let step = 0; step < maxNewTokens; step++) {
			const seqLen = inputIds.length;

			const feeds: Record<string, ort.Tensor> = {
				input_ids: new ort.Tensor('int64', inputIds, [1, seqLen]),
				attention_mask: new ort.Tensor('int64', attentionMask, [1, seqLen]),
			};

			const output = await session.run(feeds);
			const logits = output.logits;
			if (!logits) {
				throw new Error("Model output missing 'logits'");
			}

			if (step === 0) {
				vocabSize = logits.dims[2];
			}

			const logitsData = logits.data as Float32Array;
			const lastTokenLogits = logitsData.subarray(
				(seqLen - 1) * vocabSize,
				seqLen * vocabSize,
			);

			const nextTokenId = sampleToken(lastTokenLogits, temperature, topK);

			if (nextTokenId === eosTokenId) break;

			generatedTokens.push(nextTokenId);

			if (onToken) {
				const tokenText = tokenizer.decode([Number(nextTokenId)], {
					skip_special_tokens: true,
				});
				onToken(tokenText);
			}

			// 拼接到 input_ids 用于下一步
			const newInputIds = new BigInt64Array(seqLen + 1);
			newInputIds.set(inputIds);
			newInputIds[seqLen] = nextTokenId;
			inputIds = newInputIds;

			const newMask = new BigInt64Array(seqLen + 1);
			newMask.set(attentionMask);
			newMask[seqLen] = 1n;
			attentionMask = newMask;
		}

		return tokenizer.decode(generatedTokens.map(Number), {
			skip_special_tokens: true,
		});
	};

	const dispose = async () => {
		await session.release();
	};

	return { generate, tokenizer, dispose };
};

/**
 * Top-K + Temperature 采样。
 *
 * 优化:
 * - js-min-max-loop: 用 O(n) partial select 替代 O(n log n) 全量排序
 * - js-combine-iterations: softmax exp + normalize 合并为单次循环
 */
const sampleToken = (
	logits: Float32Array,
	temperature: number,
	topK: number,
): bigint => {
	const len = logits.length;

	// js-early-exit: temperature ≈ 0 直接 argmax
	if (temperature < 1e-6) {
		let maxIdx = 0;
		let maxVal = logits[0];
		for (let i = 1; i < len; i++) {
			if (logits[i] > maxVal) {
				maxVal = logits[i];
				maxIdx = i;
			}
		}
		return BigInt(maxIdx);
	}

	// js-min-max-loop: O(n·k) partial selection 替代 O(n log n) sort
	// 对 vocab 32000+，k=40 时快 ~10x
	const k = Math.min(topK, len);
	const topIndices = new Int32Array(k);
	const topValues = new Float32Array(k);
	topIndices.fill(-1);
	topValues.fill(-Infinity);

	const invTemp = 1 / temperature;
	for (let i = 0; i < len; i++) {
		const val = logits[i] * invTemp;
		// 跳过比当前最小 top 还小的值
		if (val <= topValues[k - 1]) continue;
		// 插入排序维护 top-K（k 很小，通常 40）
		let pos = k - 1;
		while (pos > 0 && val > topValues[pos - 1]) {
			topValues[pos] = topValues[pos - 1];
			topIndices[pos] = topIndices[pos - 1];
			pos--;
		}
		topValues[pos] = val;
		topIndices[pos] = i;
	}

	// js-combine-iterations: softmax + 采样合并
	// 先找 max 用于数值稳定
	const maxLogit = topValues[0];

	// 单次循环: 计算 exp 并累加 sum
	let sumExp = 0;
	const exps = new Float32Array(k);
	for (let i = 0; i < k; i++) {
		const e = Math.exp(topValues[i] - maxLogit);
		exps[i] = e;
		sumExp += e;
	}

	// 按概率采样（边归一化边累加，无需单独的 probs 数组）
	const rand = Math.random() * sumExp;
	let cumulative = 0;
	for (let i = 0; i < k; i++) {
		cumulative += exps[i];
		if (rand < cumulative) {
			return BigInt(topIndices[i]);
		}
	}

	return BigInt(topIndices[k - 1]);
};
