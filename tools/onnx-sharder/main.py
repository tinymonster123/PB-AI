"""ONNX 按层分片工具入口。

用法:
    python main.py \
      --input ../../models/tinyllama-1.1b-chat-int8/model_quantized.onnx \
      --output ../../dist/tinyllama-int8/ \
      --model-id "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
      --variant base \
      --dtype int8 \
      --layers-per-chunk 1 \
      --copy-tokenizer ../../models/tinyllama-1.1b-chat-int8/

    # 旧模式 (不拆分 base):
    python main.py \
      --input model.onnx --output dist/ --model-id "model" \
      --no-split-base --layers-per-chunk 4
"""

from src.cli import parse_args
from src.parser import load_onnx_model, classify_initializers, print_summary
from src.writer import write_shards, ModelManifest, generate_config, copy_tokenizer


def main():
    args = parse_args()

    # Step 1: 加载 ONNX 模型
    print(f"\n[1/5] 加载模型: {args.input}")
    model = load_onnx_model(args.input)

    # Step 2: 分类 initializer
    print("\n[2/5] 分类 initializers...")
    result = classify_initializers(list(model.graph.initializer), model.graph)
    print_summary(result)

    total_layers = result.max_layer + 1 if result.max_layer >= 0 else 0

    # Step 3: 写入分片
    print("[3/5] 写入分片 external data 文件...")
    shards = write_shards(
        model, result, args.output, args.layers_per_chunk,
        split_base=args.split_base,
    )

    # Step 4: 生成 manifest
    print("\n[4/5] 生成 manifest.json...")
    manifest = ModelManifest(
        model_id=args.model_id,
        variant=args.variant,
        framework="onnxruntime-web",
        dtype=args.dtype,
        total_layers=total_layers,
        shards=shards,
    )
    manifest.write(args.output)

    # Step 5: 生成 config 并复制 tokenizer
    print("\n[5/5] 生成 config 和 tokenizer...")
    num_data_files = len(shards)
    generate_config(args.output, num_data_files, model_type=args.model_type)

    if args.copy_tokenizer:
        copy_tokenizer(args.copy_tokenizer, args.output)

    # 汇总
    print(f"\n{'='*60}")
    print(f"完成! 输出目录: {args.output}")
    print(f"  变体: {args.variant}")
    print(f"  分片数: {num_data_files}")
    print(f"  总层数: {total_layers}")
    print(f"  每片层数: {args.layers_per_chunk}")
    print(f"  拆分 base: {'是' if args.split_base else '否'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
