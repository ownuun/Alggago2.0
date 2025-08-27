# AlggaGo 전용 훈련소 시스템

## 개요

AlggaGo 전용 훈련소 시스템은 일반공격과 틈새공격 성공률이 미달되었을 때 각각 전용 환경에서 전문적으로 훈련시키는 시스템입니다. 이를 통해 특정 전략의 성공률을 향상시키고, 전체적인 게임 성능을 개선할 수 있습니다.

## 시스템 구성

### 1. 전용 훈련 환경

#### `RegularAttackTrainingEnv` (일반공격 전용 훈련소)
- **목적**: 상대 돌을 직접 제거하는 일반공격 성공률 향상
- **특징**:
  - 일반공격만 강제 (틈새공격 비활성화)
  - 더 가까운 초기 돌 배치로 일반공격 기회 증가
  - 일반공격 성공 시 더 높은 보상
  - 자신의 돌 손실에 대한 낮은 페널티

#### `SplitAttackTrainingEnv` (틈새공격 전용 훈련소)
- **목적**: 두 돌 사이에 자신의 돌을 위치시키는 틈새공격 성공률 향상
- **특징**:
  - 틈새공격만 강제 (일반공격 비활성화)
  - 더 넓은 초기 돌 배치로 틈새공격 기회 증가
  - 틈새공격 성공 시 더 높은 보상
  - 틈새공격 실패 시 낮은 페널티

### 2. 성능 분석 시스템

- **가운틀 로그 분석**: `gauntlet_log.csv` 파일을 분석하여 성공률 계산
- **임계값 설정**: 
  - 일반공격 성공률 임계값: 30% (기본값)
  - 틈새공격 성공률 임계값: 20% (기본값)
- **자동 판단**: 성공률이 임계값 미달 시 자동으로 전용 훈련 실행

### 3. 훈련 관리 시스템

#### `SpecializedTrainingManager`
- **기능**:
  - 성능 분석 및 훈련 필요성 판단
  - 전용 훈련 환경 생성 및 관리
  - 훈련 결과 시각화
  - 하이브리드 모델 생성

## 사용법

### 1. 독립 실행

```bash
# 기본 설정으로 전용 훈련 실행
python run_specialized_training.py

# 특정 모델을 기반으로 훈련
python run_specialized_training.py --model rl_models_competitive/model_a_100000_0.050.zip

# 임계값 조정
python run_specialized_training.py --threshold-regular 0.4 --threshold-split 0.3

# 훈련 타임스텝 조정
python run_specialized_training.py --timesteps 100000

# 결과 시각화 포함
python run_specialized_training.py --visualize
```

### 2. 메인 훈련과 통합

```bash
# 메인 훈련 실행 (자동으로 전용 훈련소 시스템 포함)
python train.py
```

메인 훈련이 완료되면 자동으로 성능을 분석하고 필요시 전용 훈련을 실행합니다.

### 3. 프로그래밍 방식 사용

```python
from specialized_training_manager import SpecializedTrainingManager

# 매니저 초기화
manager = SpecializedTrainingManager("path/to/base/model.zip")

# 성능 분석
regular_rate, split_rate = manager.analyze_performance()

# 전용 훈련 실행
if manager.needs_regular_training(regular_rate):
    manager.train_regular_attack()

if manager.needs_split_training(split_rate):
    manager.train_split_attack()
```

## 파일 구조

```
AlggaGo1.9/
├── specialized_training_envs.py      # 전용 훈련 환경
├── specialized_training_manager.py   # 훈련 관리 시스템
├── run_specialized_training.py       # 독립 실행 스크립트
├── train.py                          # 메인 훈련 (통합됨)
├── rl_models_specialized/            # 전용 훈련 모델 저장소
│   ├── regular_specialized_final.zip
│   ├── split_specialized_final.zip
│   ├── hybrid_model.zip
│   └── training_results.png
└── README_SPECIALIZED_TRAINING.md    # 이 파일
```

## 설정 옵션

### 성공률 임계값
- `regular_success_threshold`: 일반공격 성공률 임계값 (기본값: 0.3)
- `split_success_threshold`: 틈새공격 성공률 임계값 (기본값: 0.2)

### 훈련 설정
- `training_timesteps`: 전용 훈련 타임스텝 (기본값: 50000)
- `eval_freq`: 평가 빈도 (기본값: 1000)
- `eval_episodes`: 평가 에피소드 수 (기본값: 100)

## 성능 모니터링

### 로그 파일
- `rl_models_specialized/logs_regular_specialized.csv`: 일반공격 훈련 로그
- `rl_models_specialized/logs_split_specialized.csv`: 틈새공격 훈련 로그

### 시각화
- `rl_models_specialized/training_results.png`: 훈련 결과 그래프

## 하이브리드 모델

두 전용 훈련이 모두 완료되면 하이브리드 모델이 자동으로 생성됩니다. 이 모델은:
- 기본 모델의 구조를 유지
- 전용 훈련에서 학습한 전략적 지식을 통합
- 일반공격과 틈새공격을 모두 효과적으로 수행

## 주의사항

1. **의존성**: `specialized_training_envs.py`와 `specialized_training_manager.py` 파일이 필요합니다.
2. **메모리**: 전용 훈련은 추가 메모리를 사용할 수 있습니다.
3. **시간**: 전용 훈련은 추가 시간이 소요됩니다 (기본 50,000 타임스텝).
4. **호환성**: 기존 AlggaGo 환경과 완전히 호환됩니다.

## 문제 해결

### ImportError 발생 시
```bash
# 필요한 파일들이 있는지 확인
ls specialized_training_envs.py specialized_training_manager.py
```

### 성능 분석 실패 시
```bash
# 가운틀 로그 파일 확인
ls rl_logs_competitive/gauntlet_log.csv
```

### 훈련 실패 시
```bash
# 로그 확인
cat rl_models_specialized/logs_regular_specialized.csv
cat rl_models_specialized/logs_split_specialized.csv
```

## 예시 출력

```
=== AlggaGo 전용 훈련소 시스템 ===
자동으로 최신 모델을 선택했습니다: rl_models_competitive/model_a_100000_0.050.zip

=== 성능 분석 ===
[SpecializedTrainingManager] 성능 분석 결과:
  - 일반공격 성공률: 0.250 (임계값: 0.300)
  - 틈새공격 성공률: 0.150 (임계값: 0.200)

=== 훈련 필요성 분석 ===
일반공격 훈련 필요: ✅ 예
틈새공격 훈련 필요: ✅ 예

전용 훈련을 시작하시겠습니까? (y/N): y

=== 전용 훈련 시작 ===
[SpecializedTrainingManager] 일반공격 전용 훈련 시작...
[SpecializedTrainingManager] 틈새공격 전용 훈련 시작...
[SpecializedTrainingManager] 하이브리드 모델 생성 중...

✅ 전용 훈련 완료!
생성된 모델들:
  - regular: rl_models_specialized/regular_specialized_final.zip
  - split: rl_models_specialized/split_specialized_final.zip
  - hybrid: rl_models_specialized/hybrid_model.zip

모델들은 'rl_models_specialized' 폴더에 저장되었습니다.
```

## 라이선스

이 시스템은 AlggaGo 프로젝트의 일부로 동일한 라이선스를 따릅니다.
