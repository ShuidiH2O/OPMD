"""
OPMD/NTP Evaluation — Grading
=============================
对 vLLM 生成的推理结果进行评分（rule-based matching）。

用法:
    python grade.py --input_dir results/ --output_dir results/
"""
import argparse
import json
import sys
from pathlib import Path

# 添加当前目录到 path 以导入 utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import grade_answer_verl, grade_answer_mathd, grade_answer_sympy


def process_jsonl_file(filepath):
    """读取 JSONL 文件，按 example_id 聚合 responses。"""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            eid = int(data["example_id"])
            while len(results) <= eid:
                results.append({"gt": None, "responses": []})
            results[eid]["gt"] = data["answer"]
            results[eid]["responses"].append(data["response"])
    return results


def grade_file(filepath, use_judge=False):
    """对单个 JSONL 文件进行评分。"""
    data = process_jsonl_file(filepath)
    num_pred = len(data[0]["responses"]) if data else 0

    # 如果使用 judge，预提取所有答案
    if use_judge:
        from judge_extract import JudgeExtractor
        print("  Loading judge model (DeepSeek-R1-Distill-Qwen-1.5B)...")
        judge = JudgeExtractor()
        print("  Extracting answers with LLM judge...")
        for item_idx, item in enumerate(data):
            for resp_idx, resp in enumerate(item["responses"]):
                try:
                    extracted = judge.extract(str(resp))
                    item["responses"][resp_idx] = {"text": str(resp), "extracted": extracted}
                except Exception as e:
                    item["responses"][resp_idx] = {"text": str(resp), "extracted": None}
            if (item_idx + 1) % 10 == 0:
                print(f"    {item_idx + 1}/{len(data)} questions processed")
        judge.cleanup()
        print("  Judge extraction complete.")

    all_scores = []
    avg_scores = []
    best_scores = []

    for item in data:
        gt = item["gt"]
        scores = []
        for resp in item["responses"]:
            if use_judge:
                # 使用 judge 提取的答案
                extracted = resp["extracted"] if isinstance(resp, dict) else None
                if extracted and extracted.upper() != "UNKNOWN":
                    score = 1.0 if grade_answer_mathd(extracted, gt) or grade_answer_sympy(extracted, gt) else 0.0
                else:
                    # fallback 到正则提取
                    text = resp["text"] if isinstance(resp, dict) else str(resp)
                    score = 1.0 if grade_answer_verl(text, gt) else 0.0
            else:
                score = 1.0 if grade_answer_verl(str(resp), gt) else 0.0
            scores.append(score)
        all_scores.extend(scores)
        avg_scores.append(sum(scores) / len(scores) if scores else 0)
        best_scores.append(max(scores) if scores else 0)

    total = len(avg_scores)
    return {
        "file": str(filepath),
        "num_questions": total,
        "num_responses_per_question": num_pred,
        "pass@1": sum(best_scores) / total if total > 0 else 0,
        "avg@k": sum(avg_scores) / total if total > 0 else 0,
        "solve_none": sum(1 for s in avg_scores if s == 0),
        "solve_all": sum(1 for s in avg_scores if s == 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing JSONL files from gen_vllm.py")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save grading results (default: same as input_dir)")
    parser.add_argument("--use_judge", action="store_true", default=False,
                        help="Use LLM-as-a-Judge (DeepSeek-R1-1.5B) to extract answers")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        print(f"Grading: {jsonl_file.name}")
        result = grade_file(jsonl_file, use_judge=args.use_judge)
        all_results.append(result)

        # 打印单任务结果
        print(f"  Questions: {result['num_questions']}")
        print(f"  pass@1:    {result['pass@1']:.4f} ({result['pass@1']*100:.1f}%)")
        print(f"  avg@k:     {result['avg@k']:.4f} ({result['avg@k']*100:.1f}%)")
        print()

    # 保存汇总结果
    summary_path = output_dir / "grading_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")

    # 打印汇总表
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Task':<15} {'pass@1':>10} {'avg@k':>10} {'Questions':>10}")
    print("-" * 45)
    for r in all_results:
        task_name = Path(r["file"]).stem.split("_")[0].upper()
        print(f"{task_name:<15} {r['pass@1']:>10.4f} {r['avg@k']:>10.4f} {r['num_questions']:>10}")


if __name__ == "__main__":
    main()