"""smolvla_libero가 학습에 쓴 데이터셋의 카메라 구성을 확인한다.

성공률이 공개 수치(90%대)보다 훨씬 낮은 60%로 나왔다. 1순위 가설은
"모델은 카메라 3대로 학습됐는데 평가에서 2대 + 빈 이미지를 먹이고 있다"이다.
학습 데이터셋의 실제 카메라 키를 보면 세 번째가 무엇인지 알 수 있다.
"""
import json

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

print("=== LIBERO 데이터셋 후보 ===")
for q in ["libero"]:
    for d in api.list_datasets(search=q, limit=25, sort="downloads"):
        print("   {:60s} dl={}".format(d.id, getattr(d, "downloads", 0) or 0))

print()
for repo in ["HuggingFaceVLA/libero", "lerobot/libero"]:
    print("=== {} info.json ===".format(repo))
    try:
        p = hf_hub_download(repo_id=repo, filename="meta/info.json", repo_type="dataset")
        info = json.load(open(p, encoding="utf-8"))
        feats = info.get("features", {})
        cams = [k for k in feats if "image" in k or "camera" in k]
        print("   카메라 키: {}".format(cams))
        for k in cams:
            print("      {} -> shape={}".format(k, feats[k].get("shape")))
        st = {k: v.get("shape") for k, v in feats.items() if "state" in k or k == "action"}
        print("   state/action: {}".format(st))
        print("   fps={} total_episodes={}".format(info.get("fps"), info.get("total_episodes")))
    except Exception as e:
        print("   실패:", type(e).__name__, str(e)[:200])
    print()
