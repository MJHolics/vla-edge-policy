# 환경 구축 기록 — 걸린 함정과 해결

> 2026-08-18 · Windows 11 + WSL2 Ubuntu 24.04 · RTX 4080 SUPER 16GB
> 재현: `bash setup_vla.sh` (WSL 안에서)

## 왜 WSL2인가 — 선택이 아니라 제약이었다

| 이유 | 사실 |
|---|---|
| **디스크** | Windows `C:` 여유 **70G / 930G(93% 사용)**. WSL2 루트는 **939G 여유** |
| **LIBERO** | lerobot의 `libero` extra가 `hf-libero`에 의존하는데 **`sys_platform == "linux"` 조건부**다. Windows 네이티브로는 설치 자체가 안 된다 |
| **렌더링** | MuJoCo/robosuite 헤드리스 렌더링에 OSMesa·EGL이 필요 — 리눅스 쪽이 훨씬 단순 |

GPU 패스스루는 정상이다(`nvidia-smi`가 WSL 안에서 4080 SUPER를 인식).

## 함정 4개

### 1. WSL 안에서 `nohup`으로 띄우면 프로세스가 같이 죽는다

```bash
wsl -d Ubuntu -- bash -lc 'nohup ~/setup.sh &'   # ❌ 로그 파일조차 안 생긴다
```

명령이 반환되는 순간 WSL이 **마지막 프로세스가 끝났다고 보고 배포판을 종료**한다. `nohup`은
SIGHUP만 막지 배포판 종료를 막지 못한다. 호출하는 쪽에서 **프로세스를 붙잡고 있어야** 한다
(포그라운드로 실행하되 호출자가 백그라운드에서 대기).

### 2. Git Bash가 `/root/...`를 Windows 경로로 바꾼다 (MSYS 경로 변환)

```bash
wsl -d Ubuntu -- bash /root/setup.sh
# → bash: C:/Program Files/Git/root/setup.sh: No such file or directory (exit 127)
```

MSYS는 유닉스처럼 생긴 **인자**를 Windows 경로로 번역한다. `~`도 마찬가지로 호출 측에서 먼저 확장된다
(`~/setup.sh` → `C:/Users/apple/setup.sh`). 해결은 **작은따옴표 안에서 리눅스가 확장하게** 두는 것:

```bash
wsl -d Ubuntu -- bash -lc 'bash "$HOME/setup.sh"'   # ✅
```

### 3. `torch`를 cu124로 핀했는데 lerobot이 덮어썼다

설치 중간 로그에는 `torch 2.6.0+cu124`가 찍혔는데, `pip list` 최종 상태는 **`torch 2.11.0+cu130`**이었다.
`pip install lerobot`이 더 높은 버전을 끌어온 것이다. **중간 로그를 믿고 넘어가면 안 된다 — 최종 상태를 다시 확인해야 한다.**

확인해 보니 문제는 없었다(드라이버 591.86이 CUDA 13을 지원):

```
torch 2.11.0+cu130 · cuda build 13.0 · available True
device NVIDIA GeForce RTX 4080 SUPER · VRAM 16.0GB · bf16 matmul OK
```

**결론: cu124 핀은 무의미했다.** lerobot을 먼저 깔고 torch 상태를 검증하는 순서가 맞다.

### 4. "cmake가 없다"는데 cmake는 멀쩡했다 — venv의 **깨진 shim**이 가리고 있었다

`lerobot[libero]`가 `egl_probe`·`hf-egl-probe` 휠 빌드에서 죽었다:

```
subprocess.CalledProcessError: Command '['cmake', '--version']' returned non-zero exit status 1
```

셸에서는 `cmake --version`이 3.28.3을 잘 뱉는다. **에러 메시지를 그대로 믿었으면 apt로 cmake를
다시 깔며 시간을 버렸을 것이다.** 전체 로그를 보니 진짜 원인은 딴 데 있었다:

```
File "/root/vla-venv/bin/cmake", line 3, in <module>
    from cmake import cmake
ModuleNotFoundError: No module named 'cmake'
```

venv의 `bin/cmake`가 **파이썬 shim**인데 그게 부르는 `cmake` 모듈이 없다.
`cmake-4.1.3.dist-info`는 남아 있는데 모듈은 없는 **반쯤 설치된 상태**였고,
venv의 `bin/`이 PATH 앞에 오니 **정상인 `/usr/bin/cmake`를 가려 버린** 것이다.

```bash
pip uninstall -y cmake     # 깨진 shim 제거 → /usr/bin/cmake 3.28.3 사용
```

> **교훈**: 빌드 에러의 종료코드는 "무엇이 실패했나"만 알려주고 "왜"는 안 알려준다.
> `... returned non-zero exit status 1`은 **명령을 못 찾은 게 아니라 찾아서 실행했는데 실패한 것**이라
> `which`로 확인되는 것과 실제 실행되는 것이 다를 수 있다.

### 5. 파이프를 태우면 pip 실패가 exit 0으로 보고된다

```bash
pip install ... 2>&1 | tail -25    # ❌ tail의 종료코드(0)가 잡힌다
```

이 설치는 **실패했는데 exit 0으로 완료 보고**됐고, 그래서 다음 단계에서 `import`가 깨지고 나서야
알았다. 파이프를 쓸 거면 `PIPESTATUS`를 보거나 파이프를 걸지 않는다.

### 6. `sed`의 `$p`가 Git Bash에서 깨진다

`sed -n "/pattern/,\$p"`가 `unexpected ,`로 실패한다(셸 계층이 겹치며 `$`가 먹힌다).
`awk "/pattern/,0"`으로 바꾸면 된다.

### 7. `wsl -- bash -lc '...'` 안에서 **셸 변수 할당이 비어 버린다**

```bash
wsl -d Ubuntu -- bash -lc 'F=/root/x.py; wc -l $F'
# → wc: invalid zero-length file name   ($F가 비었다)
```

두 번 물렸다(`$SP`, `$F`). `$HOME`처럼 이미 존재하는 변수는 확장되는데 **새로 할당한 변수는 비어 나온다.**
셸 계층이 겹치면서 생기는 문제이고, 원인을 파기보다 **규칙으로 피하는 게 싸다 — 리터럴 경로만 쓴다.**

### 8. heredoc에 파이썬을 밀어 넣으면 따옴표·`
`이 깨진다

세 번 물렸다. `wsl -- bash -lc 'python - <<PY ... PY'` 형태에서
- f-string 안의 `\"` 가 `unexpected character after line continuation character`로 깨지고
- `
`이 **진짜 개행으로 치환돼** f-string이 미완성으로 끊긴다.

**규칙: 한 줄짜리가 아니면 파일로 만들어 `/mnt/c`에서 복사해 실행한다.**
(한글 경로 `로보틱스/`도 `/mnt/c` 경유로는 문제없이 읽힌다.)

### 9. 공식 체크포인트와 공식 env가 **서로 안 맞는다** (카메라 이름·개수)

`lerobot/smolvla_libero`(공식, 23.6k dl)를 `--env.type=libero`(공식)로 평가하면 바로 죽는다:

```
ValueError: Feature mismatch between dataset/environment and policy config.
- Missing features: ['observation.images.camera1', 'camera2', 'camera3']
- Extra features:   ['observation.images.image', 'observation.images.image2']
```

정책은 **카메라 3대**(`camera1/2/3`, 각 3×256×256)를 기대하는데 LIBERO env는 **2대**(`image`, `image2`)를 준다.
같은 조직이 올린 정책과 환경인데 키 이름조차 다르다. **"공식이니까 맞겠지"가 성립하지 않는다.**

해결은 **한 겹이면 된다**:

1. **이름 맞추기** — `--rename_map='{"observation.images.image":"observation.images.camera1", ...}'`
   검증 함수(`validate_visual_features_consistency`)는 **한쪽이 부분집합이면 통과**시킨다.
   `{camera1,camera2} ⊆ {camera1,camera2,camera3}`이 되므로 여기서 통과한다.

### ⚠ 2026-08-19 정정 — 여기에 `--policy.empty_cameras=1`도 같이 넣었는데, 틀렸다

당시엔 "없는 camera3을 `-1` 패딩 + `mask=0`으로 채워야 한다"고 적고 그렇게 돌렸다.
**두 손잡이를 동시에 돌렸고, 에러가 사라지자 둘 다 필요하다고 믿었다.** 아니었다.

- 학습 데이터셋 `lerobot/libero`의 카메라는 **2대뿐**(`image`, `image2`)이다. camera3은 데이터에 없다.
- `train_config.json`의 **`policy.empty_cameras = 0`**이고, `modeling_smolvla.prepare_images`는
  그 값이 0이면 없는 카메라를 **패딩하지 않고 즉시 탈출**한다
  (`if num_empty_cameras >= self.config.empty_cameras: break`).
- → **학습은 이미지 2장으로 이뤄졌다.** `empty_cameras=1`은 학습에 존재한 적 없는
  패딩 이미지를 매 추론에 넣는 **학습/평가 불일치**였다.

A/B(각 50에피소드): `empty_cameras=0` **64.0%** vs `=1` **60.0%** — Wilson 구간이 겹쳐
해롭다고 단정하지는 않는다. 다만 학습 조건이 0이고 없는 카메라는 비전 토큰만 늘리므로
`run_eval.sh`·`sweep_denoise.py`·`bench_policy_latency.py`의 기본값을 **0으로 되돌렸다**.

> **교훈**: 에러가 사라진 것은 **어느 손잡이가 고쳤는지에 대한 증거가 아니다.**
> 두 개를 같이 돌렸으면, 하나씩 되돌려 확인해야 한다. 이 하나가 이후 모든 측정에 얹혀 있었다
> (지연 벤치는 아예 코드에서 `empty_cameras`를 1로 **강제**하고 있었다).

## 확정된 환경

```
WSL2 Ubuntu 24.04 · Python 3.12.3 · venv ~/vla-venv
torch 2.11.0+cu130 · torchvision 0.26.0 · numpy 2.2.6 · gymnasium 1.3.0
lerobot 0.6.1
```

## lerobot 0.6.1이 실제로 담고 있는 것 (직접 확인)

추측하지 않고 `pkgutil`로 열어 봤다.

**정책 21종**: `act` · `diffusion` · `smolvla` · `pi0` · `pi05` · `pi0_fast` · `pi_gemma` · `groot` ·
`eo1` · `evo1` · `vla_jepa` · `wall_x` · `xvla` · `molmoact2` · `lingbot_va` · `multi_task_dit` ·
`fastwam` · `rtc` · `tdmpc` · `vqbet` · `gaussian_actor`

**시뮬레이션 env 6종**: `libero` · `metaworld` · `robocasa` · `robomme` · `robotwin` · `vlabench`

→ **LIBERO가 내장이라 자체 환경을 만들 필요가 없다.** 공개 수치와 대조 가능한 상태로 시작한다.
