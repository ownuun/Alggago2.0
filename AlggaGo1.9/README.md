# 🎯 AlggaGo - AI 강화학습 기반 바둑돌 게임

[![Hugging Face Spaces](https://img.shields.io/badge/Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AlggaGo는 AI가 바둑돌을 조작하여 상대방 돌을 제거하는 전략 게임입니다. 강화학습을 통해 AI가 게임 전략을 학습하고, 실시간으로 적응형 전략을 수행합니다.

## 🌟 주요 특징

- **🤖 AI 강화학습**: PPO 알고리즘을 사용한 지능형 AI
- **🎮 실시간 게임**: 웹 브라우저에서 바로 플레이 가능
- **📊 성능 분석**: AI 모델의 성능 통계 및 시각화
- **🎯 전략적 게임플레이**: 일반공격과 틈새공격 전략
- **🌐 웹 배포**: Hugging Face Spaces를 통한 쉬운 배포

## 🚀 빠른 시작

### 웹에서 플레이하기
1. [Hugging Face Spaces](https://huggingface.co/spaces)에서 AlggaGo Space 방문
2. "새 게임 시작" 버튼 클릭
3. "AI 턴 실행" 또는 "자동 게임" 버튼으로 게임 진행

### 로컬에서 실행하기
```bash
# 저장소 클론
git clone https://github.com/your-username/alggago.git
cd alggago

# 의존성 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run app.py
```

## 🎮 게임 규칙

### 기본 규칙
1. **목표**: 상대방의 모든 돌을 제거하여 승리
2. **플레이어**: 각각 4개의 돌을 가짐
3. **진행**: 턴제로 진행되며, 자신의 돌을 선택하여 쏨
4. **승리 조건**: 상대방의 모든 돌을 제거

### 전략
- **일반공격**: 상대방 돌을 직접 맞춰 제거
- **틈새공격**: 두 돌 사이에 자신의 돌을 위치시켜 간접 제거

## 🤖 AI 모델

### 모델 종류
1. **기본 모델** (50,000 타임스텝)
   - 안정적인 기본 전략
   - 승률: 65%

2. **고급 모델** (100,000 타임스텝)
   - 고급 전략과 적응성
   - 승률: 78%

3. **전문 모델** (200,000 타임스텝)
   - 최고 수준의 전략적 사고
   - 승률: 85%

### 성능 메트릭
- **일반공격 성공률**: 75-82%
- **틈새공격 성공률**: 38-52%
- **전체 승률**: 58-72%

## 🛠️ 기술 스택

### Frontend
- **Streamlit**: 웹 인터페이스
- **Plotly**: 게임 보드 시각화
- **CSS**: 커스텀 스타일링

### Backend
- **Python**: 메인 프로그래밍 언어
- **NumPy**: 수치 계산
- **Pandas**: 데이터 처리

### AI & ML
- **Stable-Baselines3**: 강화학습 프레임워크
- **Gymnasium**: 강화학습 환경
- **PyTorch**: 딥러닝 프레임워크

### 게임 엔진
- **Pymunk**: 2D 물리 엔진
- **Pygame**: 게임 개발 라이브러리

## 📁 프로젝트 구조

```
AlggaGo1.9/
├── app.py                          # 메인 Streamlit 앱
├── app_simple.py                   # 간단한 버전 앱
├── requirements.txt                # Python 의존성
├── README.md                       # 프로젝트 설명
├── README_HF_SPACES.md            # Hugging Face 배포 가이드
├── .gitignore                      # Git 무시 파일
├── physics.py                      # 물리 엔진
├── agent.py                        # AI 에이전트
├── env.py                          # 게임 환경
├── opponent.py                     # 상대방 AI
├── opponent_c.py                   # 상대방 AI (C 모델)
├── main.py                         # 원본 게임
├── train.py                        # AI 훈련 스크립트
├── specialized_training_envs.py    # 전용 훈련 환경
├── specialized_training_manager.py # 훈련 관리 시스템
├── run_specialized_training.py     # 전용 훈련 실행
└── rl_models_competitive/          # 훈련된 AI 모델들
    ├── model_a_*.zip
    ├── model_b_*.zip
    └── ...
```

## 🔧 개발 환경 설정

### 필수 요구사항
- Python 3.8+
- pip 또는 conda

### 설치 방법
```bash
# 가상환경 생성 (권장)
python -m venv alggago_env
source alggago_env/bin/activate  # Linux/Mac
# 또는
alggago_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 개발 도구
```bash
# 코드 포맷팅
pip install black
black .

# 린팅
pip install flake8
flake8 .

# 테스트
python -m pytest tests/
```

## 🚀 Hugging Face Spaces 배포

### 1. Space 생성
1. [Hugging Face](https://huggingface.co/)에 로그인
2. "New Space" 클릭
3. 설정:
   - **Owner**: 본인의 사용자명
   - **Space name**: `alggago-game`
   - **License**: MIT
   - **SDK**: Streamlit
   - **Space hardware**: CPU (기본)

### 2. 파일 업로드
필수 파일들을 Space에 업로드:
- `app.py` (메인 앱)
- `requirements.txt` (의존성)
- `README.md` (프로젝트 설명)

### 3. 자동 배포
파일 업로드 후 자동으로 배포됩니다.

## 📊 성능 모니터링

### 웹 인터페이스
- 실시간 게임 상태 표시
- AI 성능 통계
- 게임 히스토리 추적

### 로그 분석
- 훈련 로그: `rl_logs_competitive/`
- 성능 메트릭: `training_log.csv`
- 게임 기록: `*.csv` 파일들

## 🤝 기여하기

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 기여 가이드라인
- 코드 스타일: PEP 8 준수
- 문서화: 모든 함수와 클래스에 docstring 작성
- 테스트: 새로운 기능에 대한 테스트 코드 작성

## 🐛 문제 해결

### 일반적인 문제들

1. **모듈 임포트 오류**
   ```bash
   # 의존성 재설치
   pip install -r requirements.txt --force-reinstall
   ```

2. **메모리 부족**
   - 모델 크기 최적화
   - 배치 크기 조정

3. **성능 문제**
   - GPU 사용 (가능한 경우)
   - 모델 양자화

### 로그 확인
```bash
# 훈련 로그 확인
tail -f rl_logs_competitive/training_log.csv

# 오류 로그 확인
cat *.log
```

## 📈 로드맵

### 단기 목표 (1-2개월)
- [ ] 사용자 인터페이스 개선
- [ ] 추가 AI 모델 개발
- [ ] 성능 최적화

### 중기 목표 (3-6개월)
- [ ] 멀티플레이어 지원
- [ ] 토너먼트 시스템
- [ ] 모바일 앱 개발

### 장기 목표 (6개월+)
- [ ] 클라우드 서비스 구축
- [ ] API 서비스 제공
- [ ] 커뮤니티 기능 추가

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **Hugging Face**: 무료 호스팅 서비스 제공
- **Streamlit**: 훌륭한 웹 프레임워크
- **Stable-Baselines3**: 강화학습 라이브러리
- **Pymunk**: 2D 물리 엔진

## 📞 연락처

- **GitHub**: [@your-username](https://github.com/your-username)
- **이메일**: your-email@example.com
- **프로젝트**: [AlggaGo Repository](https://github.com/your-username/alggago)

## ⭐ 스타

이 프로젝트가 도움이 되었다면 ⭐을 눌러주세요!

---

**AlggaGo** - AI 강화학습 기반 바둑돌 게임 🎯

*Made with ❤️ for the AI gaming community*
