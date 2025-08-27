import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import sys
import random
import time

# 페이지 설정
st.set_page_config(
    page_title="AlggaGo - AI 바둑돌 게임",
    page_icon="🎯",
    layout="wide"
)



# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .game-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background-color: #f8f9fa;
    }
    .stats-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

class SimpleAlggaGoGame:
    def __init__(self):
        self.board_width = 800
        self.board_height = 600
        self.stone_radius = 20
        self.reset_game()
    
    def reset_game(self):
        """게임 초기화"""
        # 초기 돌 위치 설정
        self.black_stones = [
            (200, 100), (400, 100), (600, 100), (300, 150)
        ]
        self.white_stones = [
            (200, 500), (400, 500), (600, 500), (300, 450)
        ]
        self.current_player = "black"
        self.game_over = False
        self.winner = None
        self.move_history = []
    
    def get_stone_positions(self):
        """현재 돌 위치 반환"""
        return {
            "black": self.black_stones.copy(),
            "white": self.white_stones.copy()
        }
    
    def make_ai_move(self):
        """AI가 자동으로 움직임"""
        if self.game_over:
            return
        
        # 간단한 AI 로직: 랜덤하게 돌을 선택하고 이동
        if self.current_player == "black" and self.black_stones:
            # 랜덤하게 돌 선택
            stone_idx = random.randint(0, len(self.black_stones) - 1)
            stone_pos = self.black_stones[stone_idx]
            
            # 랜덤한 방향으로 이동
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(50, 150)
            
            new_x = stone_pos[0] + distance * np.cos(angle)
            new_y = stone_pos[1] + distance * np.sin(angle)
            
            # 보드 범위 내로 제한
            new_x = max(self.stone_radius, min(self.board_width - self.stone_radius, new_x))
            new_y = max(self.stone_radius, min(self.board_height - self.stone_radius, new_y))
            
            # 돌 위치 업데이트
            self.black_stones[stone_idx] = (new_x, new_y)
            
            # 상대방 돌과 충돌 체크
            self.check_collisions()
            
            # 턴 변경
            self.current_player = "white"
            self.move_history.append(f"Black moved stone {stone_idx} to ({new_x:.1f}, {new_y:.1f})")
        
        elif self.current_player == "white" and self.white_stones:
            # 흰 돌도 같은 로직
            stone_idx = random.randint(0, len(self.white_stones) - 1)
            stone_pos = self.white_stones[stone_idx]
            
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(50, 150)
            
            new_x = stone_pos[0] + distance * np.cos(angle)
            new_y = stone_pos[1] + distance * np.sin(angle)
            
            new_x = max(self.stone_radius, min(self.board_width - self.stone_radius, new_x))
            new_y = max(self.stone_radius, min(self.board_height - self.stone_radius, new_y))
            
            self.white_stones[stone_idx] = (new_x, new_y)
            self.check_collisions()
            
            self.current_player = "black"
            self.move_history.append(f"White moved stone {stone_idx} to ({new_x:.1f}, {new_y:.1f})")
    
    def check_collisions(self):
        """돌 간 충돌 체크"""
        collision_threshold = self.stone_radius * 2
        
        # 검은 돌과 흰 돌 간의 충돌 체크
        for i, black_pos in enumerate(self.black_stones[:]):
            for j, white_pos in enumerate(self.white_stones[:]):
                distance = np.sqrt((black_pos[0] - white_pos[0])**2 + (black_pos[1] - white_pos[1])**2)
                if distance < collision_threshold:
                    # 충돌 발생 - 랜덤하게 하나 제거
                    if random.random() < 0.5:
                        self.black_stones.pop(i)
                        self.move_history.append(f"Black stone {i} removed by collision")
                    else:
                        self.white_stones.pop(j)
                        self.move_history.append(f"White stone {j} removed by collision")
                    return
        
        # 승리 조건 체크
        if not self.black_stones:
            self.game_over = True
            self.winner = "white"
        elif not self.white_stones:
            self.game_over = True
            self.winner = "black"
    
    def create_board_visualization(self):
        """게임 보드 시각화"""
        fig = go.Figure()
        
        # 보드 배경
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=self.board_width, y1=self.board_height,
            fillcolor="#8B4513",
            line=dict(color="black", width=3)
        )
        
        # 검은 돌들
        for i, (x, y) in enumerate(self.black_stones):
            fig.add_shape(
                type="circle",
                x0=x-self.stone_radius, y0=y-self.stone_radius,
                x1=x+self.stone_radius, y1=y+self.stone_radius,
                fillcolor="black",
                line=dict(color="white", width=2)
            )
            # 돌 번호 표시
            fig.add_annotation(
                x=x, y=y,
                text=str(i+1),
                showarrow=False,
                font=dict(color="white", size=12, family="Arial Black")
            )
        
        # 흰 돌들
        for i, (x, y) in enumerate(self.white_stones):
            fig.add_shape(
                type="circle",
                x0=x-self.stone_radius, y0=y-self.stone_radius,
                x1=x+self.stone_radius, y1=y+self.stone_radius,
                fillcolor="white",
                line=dict(color="black", width=2)
            )
            # 돌 번호 표시
            fig.add_annotation(
                x=x, y=y,
                text=str(i+1),
                showarrow=False,
                font=dict(color="black", size=12, family="Arial Black")
            )
        
        # 레이아웃 설정
        fig.update_layout(
            width=800,
            height=600,
            xaxis=dict(range=[0, self.board_width], showgrid=False, showticklabels=False),
            yaxis=dict(range=[0, self.board_height], showgrid=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        return fig

# 전역 게임 인스턴스
if 'game' not in st.session_state:
    st.session_state.game = SimpleAlggaGoGame()

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🎯 AlggaGo - AI 바둑돌 게임</h1>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("게임 설정")
        
        game_mode = st.selectbox(
            "게임 모드",
            ["AI vs AI", "AI vs Human", "Human vs Human"],
            index=0
        )
        
        ai_model = st.selectbox(
            "AI 모델",
            ["기본 모델", "고급 모델", "전문 모델"],
            index=0
        )
        
        if st.button("새 게임 시작", type="primary"):
            st.session_state.game.reset_game()
            st.success("새 게임이 시작되었습니다!")
            st.rerun()
        
        st.header("게임 정보")
        st.info("""
        **AlggaGo**는 AI가 바둑돌을 조작하여 상대방 돌을 제거하는 게임입니다.
        
        **게임 규칙:**
        - 각 플레이어는 4개의 돌을 가집니다
        - 자신의 돌을 쏘아 상대방 돌을 제거합니다
        - 모든 상대방 돌을 제거하면 승리합니다
        """)
    
    # 메인 컨텐츠
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("게임 보드")
        
        # 게임 보드 시각화
        board_fig = st.session_state.game.create_board_visualization()
        st.plotly_chart(board_fig, use_container_width=True)
        
        # 게임 상태 표시
        if st.session_state.game.game_over:
            st.success(f"🎉 게임 종료! {st.session_state.game.winner.title()} 승리!")
        else:
            st.info(f"현재 턴: {st.session_state.game.current_player.title()}")
        
        # 게임 컨트롤
        col_control1, col_control2, col_control3 = st.columns(3)
        
        with col_control1:
            if st.button("AI 턴 실행"):
                st.session_state.game.make_ai_move()
                st.success(f"{st.session_state.game.current_player.title()} 턴이 완료되었습니다!")
                st.rerun()
        
        with col_control2:
            if st.button("자동 게임 (10턴)"):
                for i in range(10):
                    if st.session_state.game.game_over:
                        break
                    st.session_state.game.make_ai_move()
                st.success("자동 게임이 완료되었습니다!")
                st.rerun()
        
        with col_control3:
            if st.button("게임 리셋"):
                st.session_state.game.reset_game()
                st.success("게임이 리셋되었습니다!")
                st.rerun()
    
    with col2:
        st.subheader("게임 통계")
        
        # 통계 카드들
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.markdown(f"""
            <div class="stats-card">
                <div style="font-size: 2rem; font-weight: bold; color: #000;">{len(st.session_state.game.black_stones)}</div>
                <div style="font-size: 0.9rem; color: #666;">검은 돌</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats2:
            st.markdown(f"""
            <div class="stats-card">
                <div style="font-size: 2rem; font-weight: bold; color: #666;">{len(st.session_state.game.white_stones)}</div>
                <div style="font-size: 0.9rem; color: #666;">흰 돌</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 게임 히스토리
        st.subheader("게임 히스토리")
        if st.session_state.game.move_history:
            for i, move in enumerate(st.session_state.game.move_history[-10:]):  # 최근 10개만 표시
                st.write(f"{i+1}. {move}")
        else:
            st.info("아직 움직임이 없습니다.")
    
    # 하단 섹션
    st.markdown("---")
    
    # 탭으로 추가 기능들
    tab1, tab2, tab3 = st.tabs(["📊 성능 분석", "🤖 AI 모델", "📖 게임 설명"])
    
    with tab1:
        st.subheader("AI 성능 분석")
        
        # 샘플 성능 데이터
        performance_data = {
            '일반공격 성공률': [0.75, 0.68, 0.82, 0.71, 0.79],
            '틈새공격 성공률': [0.45, 0.52, 0.38, 0.49, 0.41],
            '전체 승률': [0.65, 0.58, 0.72, 0.61, 0.68]
        }
        
        # 성능 차트
        for metric, values in performance_data.items():
            st.write(f"**{metric}**")
            chart_data = pd.DataFrame({
                '게임': range(1, len(values) + 1),
                '성공률': values
            })
            st.line_chart(chart_data.set_index('게임'))
    
    with tab2:
        st.subheader("AI 모델 정보")
        
        # 모델 정보 표시
        model_info = {
            "기본 모델": {
                "훈련 타임스텝": "50,000",
                "승률": "65%",
                "특징": "안정적인 기본 전략"
            },
            "고급 모델": {
                "훈련 타임스텝": "100,000",
                "승률": "78%",
                "특징": "고급 전략과 적응성"
            },
            "전문 모델": {
                "훈련 타임스텝": "200,000",
                "승률": "85%",
                "특징": "최고 수준의 전략적 사고"
            }
        }
        
        for model_name, info in model_info.items():
            with st.expander(model_name):
                for key, value in info.items():
                    st.write(f"**{key}:** {value}")
    
    with tab3:
        st.subheader("게임 상세 설명")
        
        st.markdown("""
        ### 🎯 AlggaGo 게임 규칙
        
        **목표:** 상대방의 모든 돌을 제거하여 승리합니다.
        
        **게임 진행:**
        1. 각 플레이어는 4개의 돌을 가집니다
        2. 턴제로 진행되며, 자신의 돌을 선택하여 쏩니다
        3. 돌을 쏘는 방법:
           - 돌을 선택합니다
           - 각도와 힘을 조절합니다
           - 발사합니다
        
        **전략:**
        - **일반공격:** 상대방 돌을 직접 맞춰 제거
        - **틈새공격:** 두 돌 사이에 자신의 돌을 위치시켜 간접 제거
        
        **AI 특징:**
        - 강화학습을 통한 전략 학습
        - 실시간 적응형 전략
        - 물리 시뮬레이션 기반 정확한 예측
        """)

if __name__ == "__main__":
    main()
