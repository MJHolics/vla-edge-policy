# P4 — 정책을 ROS2로 감싸 메시징 오버헤드와 폐루프 주기를 잰다

> 측정일 2026-09-02 · WSL2 Ubuntu 24.04 + ROS2 Jazzy(ros-base) · RTX 4080 SUPER 16GB
> 정책 `lerobot/smolvla_libero`(450M, num_steps 기본값 10)
> 코드: `ros2_ws/`(`vla_ros_bench_msgs`, `vla_ros_bench`) · 재현: 아래 §5

PLAN.md의 로드맵 P4("ROS2 노드로 래핑 — 정책을 토픽 인터페이스 뒤에 두고 주기 지터 측정")를
**두 단계로 완료했다**: ① 서비스(request/response)로 왕복 지연을 분해(§2~§3) ② 타이머+토픽으로
실제 폐루프 주기 지터를 측정(§3.5). 둘을 나눈 이유는 §4에 적었다.

---

## 0. 한 줄

**ROS2 서비스 계층이 추가하는 지연은 왕복시간의 0.5% 미만이다(mean 1.52ms, 추론 312.86ms 대비).**
지연을 지배하는 건 여전히 정책 추론이지 미들웨어가 아니다. 이번 실행의 추론 지연(num_steps=10
기본값, mean 312.86ms)은 `RESULTS.md`가 이미 문서화한 세션 간 변동폭(같은 설정에서 p50
255.5~414.5ms) 안에 들어오며, 그 변동의 원인은 이 프로젝트가 아직 완전히 규명하지 못했다(§3).

---

## 1. 왜 이 실험인가

`RESULTS.md`까지의 측정은 전부 **같은 파이썬 프로세스 안에서 정책 함수를 직접 호출**한 지연이다.
실제 로봇 스택에서는 정책이 보통 별도 노드로 떠서 메시지로 관측을 받고 액션을 돌려준다 —
그 경계를 넘는 비용(직렬화, 프로세스 간 통신, ROS2 미들웨어)은 지금까지 한 번도 재지 않았다.
오늘 조사한 로보티즈 채용 공고가 "ROS/ROS2 기반 로봇 개발 경험"을 지원자격으로 요구하는 것을
보고, 이 프로젝트가 로드맵에 이미 적어 둔 P4를 실제로 구현하기로 했다.

## 2. 측정 구조

```
bench_client (rclpy)                    policy_server (rclpy)
  obs_sample.npz 로드                      시작 시 1회: build_policy_and_batch()
  (360x360x3 uint8 이미지 2장, state 8차원,      → policy, 고정 preprocess된 batch
   고정 — 매 호출 동일 데이터)                    → in-process 기준선 60회 측정(§3)
        │  client_send_stamp = now()
        ├─ /vla/infer_action (rosidl 서비스) ──▶  server_recv_stamp = now()
        │                                         이미지 2장 실제 디코딩(전송 검증용)
        │                                         policy.predict_action_chunk(batch)
        │                                         torch.cuda.synchronize()
        │                                         server_done_stamp = now()
        ◀── action[7], horizon, 두 타임스탬프 ──┤
  client_recv_stamp = now()
  round_trip = client_recv − client_send
  server_proc = server_done − server_recv (서버가 직접 잰 값, 클록은 한 대라 스큐 없음)
  messaging_overhead = round_trip − server_proc
```

**추론 입력은 고정 batch를 재사용한다.** 클라이언트가 보내는 이미지는 실제 360x360x3 바이트를
싣고 실제로 디코딩되지만(전송·역직렬화 비용은 진짜로 발생), 정책 호출 자체는 서버가 시작할 때
만든 배치를 쓴다 — 매번 새로 전처리하면 "메시징 오버헤드"가 아니라 "관측마다 다른 전처리 시간"을
재게 되기 때문이다(`tools/bench_policy_latency.py`가 고정 batch를 쓰는 것과 같은 이유).
이건 재구성이 아니라 통제된 실험 설계이고, 여기 명시해 둔다.

**round_trip − server_proc으로 오버헤드를 뽑은 것이 이 실험에서 가장 중요한 설계 결정이다.**
같은 호출 안에서 두 값을 함께 받기 때문에, 회차마다 요동치는 GPU 상태(§3)가 오버헤드
쪽으로 새지 않는다. 서로 다른 두 실행(예: 별도 in-process 벤치 vs ROS2 벤치)의 평균을 빼면
그 요동이 그대로 "오버헤드"로 오인될 수 있는데, 실제로 그 함정에 걸렸다 — §3 참조.

## 3. 결과

| 구간 | n | mean | p50 | p95 | p99 | CV | 인용 가능 |
|---|---:|---:|---:|---:|---:|---:|---|
| 왕복(클라이언트 체감) | 60 | 314.38ms | 300.17ms | 424.83ms | 465.09ms | 19.2% | ✅ |
| 서버측 추론(콜백 내부, GPU 동기화 포함) | 60 | 312.86ms | 298.71ms | 423.39ms | 463.40ms | 19.3% | ✅ |
| **메시징 오버헤드(왕복 − 서버추론)** | 60 | **1.52ms** | 1.55ms | 1.70ms | 1.73ms | **7.9%** | ✅ |

원자료: `results/ros2_roundtrip.json`.

### 추론 지연 자체가 어느 범위에 속하는지 — RESULTS.md와 대조

이 실행은 `num_steps`를 따로 지정하지 않아 정책 기본값(**10**)을 그대로 썼다. §0에서 인용한
"낮은 모드 49~60ms·높은 모드 120~153ms"는 `RESULTS.md`가 num_steps=1 기준으로 정리한 값이고,
**num_steps=10 자체는 그 표에서도 세션마다 p50 255.5ms~414.5ms까지 흔들린다**(§2 "절대값이
실행마다 튄다" 표). 이번 서버측 추론 평균(312.86ms)·별도 in-process 기준선(343.09ms,
`results/inprocess_baseline_separate_run.json`)은 **그 이미 문서화된 범위 안에 있다** — 새로
발견한 이상치가 아니라 기존에 규명하지 못한 세션 간 변동폭(§2가 "이봉 분포"로 추적하다 만 것)의
연장선이다.

측정 직후 `nvidia-smi`를 보니 GPU가 **P8(유휴 전력 상태, 16W/288W)**에 있었다. §2는 이 변동의
원인을 캐시 온도/커널 제출 경로 쪽으로 좁혀 가다 완결하지 못했는데, WSL2에서 `nvidia-smi -lgc`로
클럭을 고정할 수 없다는 것도 이미 확인된 제약이다. **이건 가설이지 검증된 인과관계가 아니다** —
클럭을 고정해 A/B로 대조하지 않았고, 이번 실행은 그 미해결 변동성에 새 관측 하나를 더한 것으로
남겨 둔다.

**그래도 §0의 결론(오버헤드 비중 <0.5%)은 이 흔들림에 영향받지 않는다.** 오버헤드를 절대 시간
차이가 아니라 같은 호출 안에서의 차이로 뽑았기 때문에, 추론 자체가 300ms든 120ms든 오버헤드는
여전히 같은 1~2ms대일 것으로 예상된다(직렬화하는 페이로드 크기가 고정이므로). 다만 이것도
아직 확인 안 된 추정이다 — 클럭이 고정된 환경에서 재실행해 오버헤드가 정말 불변인지 보는 게
다음 검증 대상이다.

## 3.5 토픽/타이머 폐루프 — "주기 지터"를 마저 쟀다 (2026-09-02, 이어서)

§4에서 서비스를 고른 이유를 적으면서 "토픽이 필요한 질문은 다르다"고 미뤄뒀던 실험을
마저 했다. `loop_publisher`가 `create_timer(1/20, on_tick)`으로 20Hz를 요청하고, 콜백
안에서 동기적으로 추론한다(가장 단순하고 흔한 컨트롤러 패턴). `loop_subscriber`는
`/vla/action_stream`을 구독해 도착 간격을 독립적으로 잰다.

| 구간 | n | mean | p50 | p95 | CV |
|---|---:|---:|---:|---:|---:|
| 틱-간 간격(퍼블리셔 자체 측정) | 29 | 348.12ms | 339.42ms | 406.22ms | 9.4% |
| 콜백 소요(추론+동기화) | 30 | 347.75ms | 339.32ms | 405.81ms | 9.3% |
| 구독자 도착 간격(독립 측정) | 30 | 341.36ms | 337.36ms | — | 8.4% |

목표 20Hz(주기 50ms) 대비 **실제 달성 2.87Hz — 6.96배 느리다.** 틱-간 간격이 콜백 소요와
사실상 같다는 것(348.12 vs 347.75ms)은 **ROS2 타이머가 오버런을 누적하지 않고 그냥 콜백이
끝날 때마다 바로 다음 콜백을 돌린다**는 뜻이다(SingleThreadedExecutor 기본 동작) — 밀린
틱을 나중에 몰아서 실행하는 것도, 건너뛰는 것도 아니고 그냥 "될 때마다" 돈다.

**구독자 쪽 도착 간격(341.36ms)이 퍼블리셔 자체 측정(348.12ms)과 2% 이내로 거의 같다** — §3의
서비스 실험과 같은 결론이다: 토픽 계층이 더하는 지연도 무시할 만하다. 이 실험의 CV(8~9%)가
서비스 실험(19~24%)보다 뚜렷이 낮은 것도 눈에 띈다 — 후속 조사 없이 관측만 남긴다(가설: 서비스는
클라이언트 프로세스의 스케줄링·왕복이 한 겹 더 끼지만, 타이머 루프는 같은 프로세스 안에서 반복돼
변동 요인이 하나 적다).

**이걸로 P4("정책을 토픽 인터페이스 뒤에 두고 주기 지터 측정")가 실제로 완료됐다** — §4의 서비스
실험과 이 절의 토픽 실험이 서로 다른 질문(왕복 지연 분해 vs 폐루프 달성 주기)에 각각 답한다.

## 4. 서비스 vs 토픽 — 뭘 골랐고 왜인가

로드맵 문구는 "토픽 인터페이스"였다. 실제로는 **서비스**(request/response)로 구현했다:

- 이 실험이 묻는 질문은 "한 번의 관측→액션 왕복에 미들웨어가 얼마를 더 쓰는가"다. 서비스는
  요청·응답을 한 쌍으로 묶어 주기 때문에 클라이언트가 별도 상관관계 로직(seq 매칭, 콜백 큐)
  없이 왕복시간을 직접 잴 수 있다. 토픽(퍼블리셔/구독자)은 비동기라 "이 액션이 어느 관측에
  대한 응답인가"를 스탬프로 따로 맞춰야 하고, 그 매칭 로직 자체가 오차의 원천이 된다.
- **토픽이 필요한 질문은 다르다** — 로드맵이 말한 "주기 지터"는 고정 주기 타이머가 관측을
  흘리고 정책 노드가 그 흐름을 따라가는 **폐루프 제어**(20Hz LIBERO 전제와 같은 구조)를
  가정한다. 이건 서비스로는 잘 안 맞고, 타이머 콜백 + 퍼블리셔/구독자 조합이 맞다.
- **§3.5에서 실제로 돌려봤다.** 20Hz 타이머 + 동기 콜백(가장 단순한 패턴)에서 ROS2가 오버런을
  어떻게 흡수하는지 확인했다 — 큐잉도 스킵도 아니고, **콜백이 끝나는 대로 바로 다음 콜백을
  돌린다**(달성 2.87Hz, 목표의 6.96배 느림). "따라가지 못할 때 스킵/큐잉/지난 액션 재사용 중
  뭘 정책으로 쓸지"는 아직 다루지 않았다 — 지금 확인한 건 **아무 정책도 안 걸었을 때의 기본
  동작**이다. 다음 스텝(미완, ⬜)은 별도 정책(예: 데드라인 넘으면 스킵)을 콜백 안에 넣어
  비교하는 것.

## 5. 재현

```bash
# 1) ROS2 워크스페이스는 WSL 네이티브 경로에서 빌드한다 — Windows 마운트(한글 경로)에서
#    colcon/CMake(rosidl_generate_interfaces)가 "list index out of range"로 깨진다(§6 참조).
cp -r ros2_ws/src ~/ros2_ws/src
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws && colcon build --symlink-install

# 2) 고정 관측 캡처(최초 1회)
export MUJOCO_GL=egl
cd <repo>/ros2_ws && /root/vla-venv/bin/python3 capture_observation.py

# 3) 서버 (터미널 1)
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
/root/vla-venv/bin/python3 -m vla_ros_bench.policy_server

# 4) 클라이언트 (터미널 2) — <n> <warmup>
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
/root/vla-venv/bin/python3 -m vla_ros_bench.bench_client 60 15

# 5) 토픽/타이머 폐루프(§3.5) — 구독자 먼저, 그 다음 퍼블리셔
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
/root/vla-venv/bin/python3 -m vla_ros_bench.loop_subscriber &
/root/vla-venv/bin/python3 -m vla_ros_bench.loop_publisher
```

## 6. 에러 대처 기록

| # | 증상 | 원인 | 해결 |
|---|---|---|---|
| 1 | `colcon build`가 `vla_ros_bench_msgs`에서 `CMake Error ... list index: 1 out of range` | Windows 마운트(`/mnt/c/...AI개발/...`)의 **한글 경로**가 rosidl의 CMake 리스트 처리를 깨뜨림(SETUP.md가 문서화한 "한글 경로 문제"가 rosidl에서도 재현) | 소스를 WSL 네이티브 경로(`~/ros2_ws/src`)로 복사해 빌드. 코드의 단일 진실 소스는 Windows 쪽에 유지하고, 빌드 전 동기화만 한다 |
| 2 | `cv_bridge.cv2_to_imgmsg(...)` → `KeyError: 16` | `ros-jazzy-cv-bridge`의 컴파일된 boost 확장(`cv_bridge_boost`)이 apt의 system numpy로 빌드돼 있는데, venv의 numpy 2.2.6과 ABI가 달라 encoding↔cv-type 매핑 테이블이 깨짐(import 시점엔 numpy가 "1.x용으로 컴파일된 모듈" 경고만 찍고 죽지 않아서 늦게 발견됨) | cv_bridge를 아예 안 쓴다. `sensor_msgs/Image`의 `height/width/encoding/step/data` 필드를 rgb8 uint8 HWC 기준으로 직접 채우고 파싱(`numpy_to_imgmsg`/`imgmsg_to_numpy`) — 단순한 인코딩에는 이 편이 의존성도 줄고 ABI 문제도 없다 |
| 3 | `KeyError: 'observation.state'` (raw obs에서 바로 찾으려 함) | `preprocess_observation` 직후의 raw 관측엔 `observation.robot_state`(중첩 dict: eef pos/mat 등)만 있고, 정책이 쓰는 평탄화된 `observation.state`는 **env_preprocessor + policy preprocessor**를 거쳐야 나온다(SETUP.md 에러#4와 같은 뿌리) | state 벡터·task 문자열은 raw obs를 직접 파싱하지 않고 `build_policy_and_batch()`가 만드는 공식 전처리 배치에서 그대로 뽑는다 |

## 7. 정직한 범위

- 시뮬레이션 관측(LIBERO)을 고정 페이로드로 재사용했다. 실제 카메라 스트림의 프레임 간
  변화·압축 포맷(예: JPEG)은 다루지 않았다 — 페이로드 크기와 형태가 다르면 직렬화 비용도 달라진다.
- 같은 머신 안의 두 ROS2 노드(loopback)만 쟀다. 네트워크를 타는 분산 배포(다른 머신의 온보드
  컴퓨터와 컨트롤러)는 재지 않았다 — 그 경우 오버헤드는 지금보다 훨씬 클 것이다.
- GPU 클럭 상태(P8)가 결과에 미친 영향은 가설이지 A/B로 검증하지 않았다.
- §3.5는 오버런에 **아무 대응 정책도 걸지 않은** 기본 동작만 쟀다. 스킵·큐잉·지난 액션 재사용 같은
  실제 대응 전략의 비교는 다루지 않았다.
- 시뮬레이션 정책(LIBERO)을 ROS2로 감싼 것이지 **실물 로봇·실시간 임베디드 배포 경험이 아니다** —
  이 프로젝트 전체가 이미 명시한 범위(RTX 4080·WSL2·시뮬레이션)를 벗어나지 않는다.
