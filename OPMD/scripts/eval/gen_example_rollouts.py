import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

PARQ = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_opmd_rand_0.05_train.parquet"
JSONL = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/datasets/megamath_web_pro_1B.jsonl"
STUDENT = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/eval_results/opmd_qwen3_1.7b_megamath_14b_teacher_rand/step_1000_math500_minerva/merged_model_step1000"
TEACHER = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD_models/Qwen3-14B-Base"
OUT = "/mnt/cpfs/epic-user/zhengruize-20260612/OPMD/eval_results/example_rollouts_raw.json"

ROWS = [99, 212, 33, 52, 22]
MAX_NEW = 128

print("loading tokenizer/models...")
tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
student = AutoModelForCausalLM.from_pretrained(STUDENT, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()
teacher = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:1").eval()
print("loaded")

df = pd.read_parquet(PARQ)

# 读取所需 jsonl 文档
doc_indices = set()
for r in ROWS:
    ei = json.loads(df.at[r, "extra_info"])
    doc_indices.add(int(ei["doc_index"]))

docs = {}
need = set(doc_indices)
with open(JSONL, "r", encoding="utf-8") as f:
    for ln, line in enumerate(f):
        if ln in need:
            docs[ln] = json.loads(line)["text"]
            need.discard(ln)
            if not need:
                break

results = []
for r in ROWS:
    ei = json.loads(df.at[r, "extra_info"])
    rm = json.loads(df.at[r, "reward_model"])
    prompt_raw = json.loads(df.at[r, "prompt"])
    prefix = prompt_raw[0]["content"]
    gt = (rm.get("ground_truth") or "").strip()
    doc_index = int(ei["doc_index"])
    anchor_pos = int(ei["anchor_position"])
    ent = float(ei["anchor_entropy"])
    doc = docs[doc_index]

    ids = tok(doc, add_special_tokens=False)["input_ids"]
    cont_ids = ids[anchor_pos:anchor_pos + MAX_NEW]
    cont_text = tok.decode(cont_ids, skip_special_tokens=True)

    chat = [{"role": "user", "content": prefix}]
    try:
        rendered = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except Exception:
        rendered = tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    inputs = tok(rendered, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    prefix_len = inputs["input_ids"].shape[1]

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        gen = student.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                               temperature=1.0, top_p=0.95,
                               eos_token_id=tok.eos_token_id, pad_token_id=pad)
    full = gen[0]
    gen_ids = full[prefix_len:]
    rollout_text = tok.decode(gen_ids, skip_special_tokens=True)

    with torch.no_grad():
        ls_stu = torch.log_softmax(student(full.unsqueeze(0).to("cuda:0"), use_cache=False).logits.float(), dim=-1)
        ls_tch = torch.log_softmax(teacher(full.unsqueeze(0).to("cuda:1"), use_cache=False).logits.float(), dim=-1)

    steps = []
    for i, gid in enumerate(gen_ids.tolist()):
        pos = prefix_len + i - 1
        sp = ls_stu[0, pos, gid].item()
        tp = ls_tch[0, pos, gid].item()
        steps.append({
            "i": i,
            "tok": tok.decode([gid], skip_special_tokens=False),
            "student_logp": round(sp, 4),
            "teacher_logp": round(tp, 4),
            "reward": round(tp - sp, 4),
        })

    rec = {
        "row": r,
        "doc_index": doc_index,
        "anchor_pos": anchor_pos,
        "anchor_entropy": ent,
        "anchor_token": gt,
        "prefix_tail": prefix[-400:],
        "continuation": cont_text[:500],
        "rollout": rollout_text,
        "first_token_match_anchor_clue": rollout_text.strip()[:40],
        "steps": steps,
    }
    results.append(rec)
    torch.cuda.empty_cache()
    print("=" * 80)
    print(f"ROW {r}  doc={doc_index} ent={ent:.3f} anchor={gt!r}")
    print("ROLLOUT:", rollout_text[:300])
    print("N_TOKENS:", len(steps))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("SAVED", OUT)