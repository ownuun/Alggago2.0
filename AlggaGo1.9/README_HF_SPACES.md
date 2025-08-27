# AlggaGo - Hugging Face Spaces 배포

## 🚀 배포 방법

### 1. Hugging Face Spaces 생성

1. [Hugging Face](https://huggingface.co/)에 로그인
2. "New Space" 클릭
3. Space 설정:
   - **Owner**: 본인의 사용자명
   - **Space name**: `alggago-game`
   - **License**: MIT
   - **SDK**: Streamlit
   - **Space hardware**: CPU (기본)

### 2. 파일 업로드

다음 파일들을 Space에 업로드:

```
alggago-game/
├── app.py                    # 메인 Streamlit 앱
├── requirements.txt          # Python 의존성
├── README.md                # 프로젝트 설명
├── physics.py               # 물리 엔진
├── agent.py                 # AI 에이전트
├── env.py                   # 게임 환경
├── opponent.py              # 상대방 AI
├── opponent_c.py            # 상대방 AI (C 모델)
├── main.py                  # 원본 게임 (참고용)
└── rl_models_competitive/   # 훈련된 AI 모델들
    ├── model_a_*.zip
    ├── model_b_*.zip
    └── ...
```

### 3. 자동 배포

파일을 업로드하면 Hugging Face가 자동으로:
- 의존성 설치
- Streamlit 앱 빌드
- 웹 서비스 배포

## 🎮 게임 기능

### 주요 기능
- **AI vs AI**: 두 AI가 자동으로 게임 진행
- **AI vs Human**: 사용자가 AI와 대전
- **Human vs Human**: 두 사용자가 대전
- **실시간 시각화**: Plotly를 사용한 게임 보드 표시
- **성능 분석**: AI 모델의 성능 통계

### 게임 모드
1. **기본 모델**: 안정적인 기본 전략
2. **고급 모델**: 고급 전략과 적응성
3. **전문 모델**: 최고 수준의 전략적 사고

## 🔧 기술 스택

- **Frontend**: Streamlit
- **시각화**: Plotly
- **물리 엔진**: Pymunk
- **AI**: Stable-Baselines3 (PPO)
- **게임 엔진**: Pygame
- **강화학습**: Gymnasium

## 📊 성능 메트릭

- **일반공격 성공률**: 75-82%
- **틈새공격 성공률**: 38-52%
- **전체 승률**: 58-72%

## 🎯 게임 규칙

1. 각 플레이어는 4개의 돌을 가짐
2. 턴제로 진행
3. 자신의 돌을 쏘아 상대방 돌 제거
4. 모든 상대방 돌을 제거하면 승리

### 전략
- **일반공격**: 상대방 돌을 직접 맞춰 제거
- **틈새공격**: 두 돌 사이에 자신의 돌을 위치시켜 간접 제거

## 🔄 업데이트 방법

1. 로컬에서 코드 수정
2. Hugging Face Space에 파일 업로드
3. 자동으로 재배포됨

## 🐛 문제 해결

### 일반적인 문제들

1. **모듈 임포트 오류**
   - 모든 필요한 파일이 업로드되었는지 확인
   - `requirements.txt`에 모든 의존성이 포함되었는지 확인

2. **메모리 부족**
   - Space hardware를 더 높은 등급으로 업그레이드
   - 모델 크기 최적화

3. **로딩 시간이 긴 경우**
   - 모델 파일 압축
   - 캐싱 구현

### 로그 확인
- Hugging Face Space의 "Settings" → "Logs"에서 오류 확인

## 📈 모니터링

- **사용량 통계**: Space 대시보드에서 확인
- **성능 모니터링**: Streamlit 내장 메트릭
- **오류 추적**: Hugging Face 로그 시스템

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Submit a pull request

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

## 🔗 링크

- **Live Demo**: [Hugging Face Space URL]
- **GitHub Repository**: [Repository URL]
- **Documentation**: [Documentation URL]

## 📞 지원

문제가 있거나 질문이 있으시면:
- GitHub Issues 생성
- Hugging Face Space Discussion 사용
- 이메일로 문의

---

**AlggaGo** - AI 강화학습 기반 바둑돌 게임 🎯
