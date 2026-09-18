"""정책이 선언한 camera3 / state[6]이 진짜 학습된 것인지 정규화 통계로 확인한다.

발견: 학습 데이터셋 lerobot/libero 는 카메라가 2대(image, image2)이고 state 는 8차원인데,
정책 config 는 카메라 3대(camera1/2/3)와 state 6차원을 선언한다. 둘 중 하나가 거짓이다.
정규화 통계(mean/std)는 학습 중 실제로 본 데이터에서 계산되므로, camera3 통계가
없거나 퇴화(std=0, mean=0/-1)면 camera3 는 한 번도 실데이터를 받은 적이 없다는 뜻이다.
"""
import json

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

REPO = "lerobot/smolvla_libero"

for fn in ["policy_preprocessor.json", "policy_postprocessor.json"]:
    p = hf_hub_download(REPO, fn)
    d = json.load(open(p, encoding="utf-8"))
    print("=== {} ===".format(fn))
    print(json.dumps(d, ensure_ascii=False)[:1200])
    print()

for fn in [
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
]:
    p = hf_hub_download(REPO, fn)
    t = load_file(p)
    print("=== {} ===".format(fn))
    for k in sorted(t):
        v = t[k].float()
        flat = v.flatten()
        head = ", ".join("{:.4f}".format(x) for x in flat[:8].tolist())
        print("   {:60s} shape={!s:16s} [{}]{}".format(
            k, tuple(v.shape), head, " ..." if flat.numel() > 8 else ""))
        if "std" in k and float(v.abs().sum()) == 0.0:
            print("      ^^ std 가 전부 0 — 실데이터를 본 적이 없다")
    print()
