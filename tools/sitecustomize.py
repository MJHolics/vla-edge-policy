"""도메인 랜덤화(마찰·질량·시각) 몽키패치 — sitecustomize로 주입해 lerobot-eval을 안 건드린다.

PYTHONPATH에 이 파일이 있는 디렉터리를 넣으면 파이썬이 기동할 때마다 자동 import된다
(`sitecustomize`는 파이썬이 시작 시 찾는 특수 모듈명). `VLA_DR_ENABLE=1` 또는
`VLA_DR_VISUAL_ENABLE=1`일 때만 동작하므로 평소 lerobot-eval 호출(sweep_denoise.py 등)은
이 파일이 PYTHONPATH에 없으면 완전히 무관하다.

패치 대상은 `ControlEnv.reset`(LIBERO의 robosuite 래퍼) 이다. `hard_reset=True`가 기본값이라
매 reset()마다 MuJoCo 모델이 XML에서 새로 로드된다 — 그래서 원본 reset() 직후에 스케일을
곱해도 이전 에피소드의 랜덤화가 누적되지 않는다(항상 기본값에서 다시 스케일).

두 축을 독립적으로 켤 수 있다:
- `VLA_DR_ENABLE=1` — 물리(geom_friction 3열 전부·body_mass). 2026-09-03 구현분.
- `VLA_DR_VISUAL_ENABLE=1` — 시각(geom_rgba 색상 스케일·light_diffuse 밝기 스케일). 2026-09-09 추가.
  물리 DR과 마찬가지로 매 리셋마다 XML 기본값에서 다시 스케일(누적 없음). 형상·카메라 위치·질감(텍스처)은
  건드리지 않는다 — 이 실험의 주장 범위는 "geom 색상·조명 밝기 강건성"까지다.

검증 로그(VLA_DR_LOG)에 에피소드별 scale·전후 평균을 남긴다 — "패치가 실제로 걸렸는지"를
print 한 줄로 믿지 않고 파일로 확인하기 위함(이 레포군 규칙: 손잡이 하나 바꾸면 되돌려서 확인한다).

주의(2026-09-03 실제로 걸렸던 함정): 이 파일을 처음에 `dr_sitecustomize.py`로 저장했더니
1차 캐너리에서 DR=off와 DR=on 성공률이 똑같이 70%가 나왔다 — 로그 파일 자체가 안 생겼다.
파이썬은 정확히 `sitecustomize`라는 이름의 모듈만 자동 import한다. 파일명을 이렇게
바꾸고 나서야 로그가 찍혔다. **print 없이 조용히 아무 효과가 없는 실패는 에러보다 위험하다.**
"""
import os

_dr_physics = os.environ.get("VLA_DR_ENABLE") == "1"
_dr_visual = os.environ.get("VLA_DR_VISUAL_ENABLE") == "1"

if _dr_physics or _dr_visual:
    import numpy as np

    _f_lo, _f_hi = (float(x) for x in os.environ.get("VLA_DR_FRICTION_RANGE", "0.5,1.5").split(","))
    _m_lo, _m_hi = (float(x) for x in os.environ.get("VLA_DR_MASS_RANGE", "0.7,1.3").split(","))
    _rgba_lo, _rgba_hi = (float(x) for x in os.environ.get("VLA_DR_RGBA_RANGE", "0.6,1.4").split(","))
    _light_lo, _light_hi = (float(x) for x in os.environ.get("VLA_DR_LIGHT_RANGE", "0.5,1.5").split(","))
    _seed_base = int(os.environ.get("VLA_DR_SEED_BASE", "0"))
    _log_path = os.environ.get("VLA_DR_LOG", "")
    _counter = {"n": 0}

    def _log(msg: str) -> None:
        if _log_path:
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    from libero.libero.envs.env_wrapper import ControlEnv

    _orig_reset = ControlEnv.reset

    def _patched_reset(self):
        ret = _orig_reset(self)
        idx = _counter["n"]
        _counter["n"] += 1
        rng = np.random.default_rng(_seed_base + idx)
        model = self.env.sim.model
        parts = ["ep={}".format(idx)]

        if _dr_physics:
            f_scale = float(rng.uniform(_f_lo, _f_hi))
            m_scale = float(rng.uniform(_m_lo, _m_hi))
            before_f = float(model.geom_friction[:, 0].mean())
            before_m = float(model.body_mass[:].mean())
            model.geom_friction[:, :] *= f_scale
            model.body_mass[:] *= m_scale
            after_f = float(model.geom_friction[:, 0].mean())
            after_m = float(model.body_mass[:].mean())
            parts.append("f_scale={:.4f} m_scale={:.4f} friction_mean {:.5f}->{:.5f} "
                          "mass_mean {:.5f}->{:.5f}".format(f_scale, m_scale, before_f, after_f, before_m, after_m))

        if _dr_visual:
            rgba_scale = float(rng.uniform(_rgba_lo, _rgba_hi))
            light_scale = float(rng.uniform(_light_lo, _light_hi))
            before_rgba = float(model.geom_rgba[:, :3].mean())
            before_light = float(model.light_diffuse[:, :].mean()) if model.nlight > 0 else 0.0
            model.geom_rgba[:, :3] = np.clip(model.geom_rgba[:, :3] * rgba_scale, 0.0, 1.0)
            if model.nlight > 0:
                model.light_diffuse[:, :] = np.clip(model.light_diffuse[:, :] * light_scale, 0.0, 1.0)
            after_rgba = float(model.geom_rgba[:, :3].mean())
            after_light = float(model.light_diffuse[:, :].mean()) if model.nlight > 0 else 0.0
            parts.append("rgba_scale={:.4f} light_scale={:.4f} rgba_mean {:.5f}->{:.5f} "
                          "light_mean {:.5f}->{:.5f}".format(rgba_scale, light_scale, before_rgba, after_rgba,
                                                              before_light, after_light))

        if _dr_physics or _dr_visual:
            self.env.sim.forward()
        _log(" ".join(parts))
        return ret

    ControlEnv.reset = _patched_reset
    _log("[dr_sitecustomize] armed: physics={} friction=[{},{}] mass=[{},{}] "
         "visual={} rgba=[{},{}] light=[{},{}] seed_base={}".format(
             _dr_physics, _f_lo, _f_hi, _m_lo, _m_hi,
             _dr_visual, _rgba_lo, _rgba_hi, _light_lo, _light_hi, _seed_base))
