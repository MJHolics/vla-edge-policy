"""smolvla_libero 저장소의 파일과 설정을 전부 열어, 어떤 데이터셋으로 학습했는지 찾는다."""
import json

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
REPO = "lerobot/smolvla_libero"

print("=== {} 파일 목록 ===".format(REPO))
files = api.list_repo_files(REPO)
for f in files:
    print("   ", f)

print()
info = api.model_info(REPO)
print("=== 메타 ===")
print("   tags:", getattr(info, "tags", None))
print("   cardData:", str(getattr(info, "cardData", None))[:600])

for fn in ["config.json", "train_config.json", "README.md"]:
    if fn not in files:
        continue
    print()
    print("=== {} ===".format(fn))
    p = hf_hub_download(REPO, fn)
    txt = open(p, encoding="utf-8").read()
    if fn.endswith(".json"):
        d = json.loads(txt)
        if fn == "train_config.json":
            print("   dataset:", json.dumps(d.get("dataset", {}), ensure_ascii=False)[:500])
            print("   env:", json.dumps(d.get("env", {}), ensure_ascii=False)[:400])
            pol = d.get("policy", {})
            print("   policy.num_steps:", pol.get("num_steps"))
            print("   policy.empty_cameras:", pol.get("empty_cameras"))
            print("   policy.input_features:", json.dumps(pol.get("input_features", {}), ensure_ascii=False)[:500])
        else:
            print("   input_features:", json.dumps(d.get("input_features", {}), ensure_ascii=False)[:500])
            print("   empty_cameras:", d.get("empty_cameras"))
    else:
        print(txt[:1500])
