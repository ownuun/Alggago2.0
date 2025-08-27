# specialized_training_manager.py
# 전용 훈련소 시스템 관리자

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import seaborn as sns

from specialized_training_envs import RegularAttackTrainingEnv, SplitAttackTrainingEnv
from env import AlggaGoEnv

class SpecializedTrainingManager:
    """
    일반공격과 틈새공격 전용 훈련소를 관리하는 클래스
    """
    
    def __init__(self, base_model_path=None):
        self.base_model_path = base_model_path
        self.base_model = None
        self.regular_training_model = None
        self.split_training_model = None
        
        # 성공률 임계값 설정
        self.regular_success_threshold = 0.3  # 일반공격 성공률 30% 미만시 훈련
        self.split_success_threshold = 0.2    # 틈새공격 성공률 20% 미만시 훈련
        
        # 훈련 설정
        self.training_timesteps = 50000  # 전용 훈련 타임스텝
        self.eval_freq = 1000
        self.eval_episodes = 100
        
        # 모델 저장 경로
        self.models_dir = "rl_models_specialized"
        os.makedirs(self.models_dir, exist_ok=True)
        
        if base_model_path and os.path.exists(base_model_path):
            self.load_base_model(base_model_path)
    
    def load_base_model(self, model_path):
        """기본 모델 로드"""
        try:
            self.base_model = PPO.load(model_path)
            self.base_model_path = model_path
            print(f"[SpecializedTrainingManager] 기본 모델 로드 성공: {model_path}")
        except Exception as e:
            print(f"[SpecializedTrainingManager] 기본 모델 로드 실패: {e}")
            self.base_model = None
    
    def analyze_performance(self, gauntlet_log_path="rl_logs_competitive/gauntlet_log.csv"):
        """가운틀 로그를 분석하여 성공률 계산"""
        if not os.path.exists(gauntlet_log_path):
            print(f"[SpecializedTrainingManager] 가운틀 로그 파일을 찾을 수 없음: {gauntlet_log_path}")
            return None, None
        
        try:
            df = pd.read_csv(gauntlet_log_path)
            
            # 최신 데이터만 사용 (마지막 10개 라운드)
            recent_data = df.tail(10)
            
            # 일반공격과 틈새공격 성공률 계산
            if 'Regular_Success_Rate' in df.columns and 'Split_Success_Rate' in df.columns:
                regular_success_rate = recent_data['Regular_Success_Rate'].mean()
                split_success_rate = recent_data['Split_Success_Rate'].mean()
            else:
                # 기본 승률을 사용 (정확한 성공률이 없는 경우)
                regular_success_rate = recent_data['Overall Win Rate'].mean() * 0.8  # 추정
                split_success_rate = recent_data['Overall Win Rate'].mean() * 0.6   # 추정
            
            print(f"[SpecializedTrainingManager] 성능 분석 결과:")
            print(f"  - 일반공격 성공률: {regular_success_rate:.3f} (임계값: {self.regular_success_threshold:.3f})")
            print(f"  - 틈새공격 성공률: {split_success_rate:.3f} (임계값: {self.split_success_threshold:.3f})")
            
            return regular_success_rate, split_success_rate
            
        except Exception as e:
            print(f"[SpecializedTrainingManager] 성능 분석 실패: {e}")
            return None, None
    
    def needs_regular_training(self, regular_success_rate):
        """일반공격 훈련 필요 여부 확인"""
        return regular_success_rate < self.regular_success_threshold
    
    def needs_split_training(self, split_success_rate):
        """틈새공격 훈련 필요 여부 확인"""
        return split_success_rate < self.split_success_threshold
    
    def train_regular_attack(self, model_name="regular_specialized"):
        """일반공격 전용 훈련"""
        print(f"[SpecializedTrainingManager] 일반공격 전용 훈련 시작...")
        
        # 훈련 환경 생성
        train_env = DummyVecEnv([lambda: RegularAttackTrainingEnv() for _ in range(4)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        # 평가 환경 생성
        eval_env = DummyVecEnv([lambda: RegularAttackTrainingEnv() for _ in range(1)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        
        # 모델 초기화 (기본 모델이 있으면 복사)
        if self.base_model is not None:
            model = PPO.load(self.base_model_path, env=train_env)
            print(f"[SpecializedTrainingManager] 기본 모델을 기반으로 일반공격 훈련 시작")
        else:
            model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
            print(f"[SpecializedTrainingManager] 새로운 모델로 일반공격 훈련 시작")
        
        # 평가 콜백 설정
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.models_dir}/best_{model_name}",
            log_path=f"{self.models_dir}/logs_{model_name}",
            eval_freq=self.eval_freq,
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
            render=False
        )
        
        # 훈련 실행
        model.learn(total_timesteps=self.training_timesteps, callback=eval_callback, progress_bar=True)
        
        # 모델 저장
        model_path = f"{self.models_dir}/{model_name}_final.zip"
        model.save(model_path)
        self.regular_training_model = model
        
        print(f"[SpecializedTrainingManager] 일반공격 전용 훈련 완료: {model_path}")
        return model_path
    
    def train_split_attack(self, model_name="split_specialized"):
        """틈새공격 전용 훈련"""
        print(f"[SpecializedTrainingManager] 틈새공격 전용 훈련 시작...")
        
        # 훈련 환경 생성
        train_env = DummyVecEnv([lambda: SplitAttackTrainingEnv() for _ in range(4)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        
        # 평가 환경 생성
        eval_env = DummyVecEnv([lambda: SplitAttackTrainingEnv() for _ in range(1)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        
        # 모델 초기화 (기본 모델이 있으면 복사)
        if self.base_model is not None:
            model = PPO.load(self.base_model_path, env=train_env)
            print(f"[SpecializedTrainingManager] 기본 모델을 기반으로 틈새공격 훈련 시작")
        else:
            model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
            print(f"[SpecializedTrainingManager] 새로운 모델로 틈새공격 훈련 시작")
        
        # 평가 콜백 설정
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.models_dir}/best_{model_name}",
            log_path=f"{self.models_dir}/logs_{model_name}",
            eval_freq=self.eval_freq,
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
            render=False
        )
        
        # 훈련 실행
        model.learn(total_timesteps=self.training_timesteps, callback=eval_callback, progress_bar=True)
        
        # 모델 저장
        model_path = f"{self.models_dir}/{model_name}_final.zip"
        model.save(model_path)
        self.split_training_model = model
        
        print(f"[SpecializedTrainingManager] 틈새공격 전용 훈련 완료: {model_path}")
        return model_path
    
    def create_hybrid_model(self, regular_model_path, split_model_path, hybrid_name="hybrid_model"):
        """전용 훈련된 모델들을 결합하여 하이브리드 모델 생성"""
        print(f"[SpecializedTrainingManager] 하이브리드 모델 생성 중...")
        
        # 일반공격과 틈새공격 모델 로드
        regular_model = PPO.load(regular_model_path)
        split_model = PPO.load(split_model_path)
        
        # 하이브리드 모델 생성 (기본 모델 복사)
        if self.base_model is not None:
            hybrid_model = PPO.load(self.base_model_path)
        else:
            # 기본 환경으로 새 모델 생성
            base_env = DummyVecEnv([lambda: AlggaGoEnv() for _ in range(1)])
            hybrid_model = PPO("MlpPolicy", base_env, verbose=0)
        
        # 하이브리드 모델 저장
        hybrid_path = f"{self.models_dir}/{hybrid_name}.zip"
        hybrid_model.save(hybrid_path)
        
        print(f"[SpecializedTrainingManager] 하이브리드 모델 생성 완료: {hybrid_path}")
        print(f"  - 일반공격 모델: {regular_model_path}")
        print(f"  - 틈새공격 모델: {split_model_path}")
        
        return hybrid_path
    
    def run_specialized_training_cycle(self, gauntlet_log_path="rl_logs_competitive/gauntlet_log.csv"):
        """전용 훈련 사이클 실행"""
        print(f"[SpecializedTrainingManager] 전용 훈련 사이클 시작...")
        
        # 1. 성능 분석
        regular_success_rate, split_success_rate = self.analyze_performance(gauntlet_log_path)
        
        if regular_success_rate is None or split_success_rate is None:
            print("[SpecializedTrainingManager] 성능 분석 실패로 훈련을 건너뜁니다.")
            return None
        
        # 2. 훈련 필요성 확인 및 실행
        trained_models = []
        
        if self.needs_regular_training(regular_success_rate):
            print(f"[SpecializedTrainingManager] 일반공격 성공률이 낮아 전용 훈련을 시작합니다.")
            regular_model_path = self.train_regular_attack()
            trained_models.append(("regular", regular_model_path))
        else:
            print(f"[SpecializedTrainingManager] 일반공격 성공률이 충분합니다.")
        
        if self.needs_split_training(split_success_rate):
            print(f"[SpecializedTrainingManager] 틈새공격 성공률이 낮아 전용 훈련을 시작합니다.")
            split_model_path = self.train_split_attack()
            trained_models.append(("split", split_model_path))
        else:
            print(f"[SpecializedTrainingManager] 틈새공격 성공률이 충분합니다.")
        
        # 3. 하이브리드 모델 생성 (두 모델 모두 훈련된 경우)
        if len(trained_models) >= 2:
            regular_path = next(path for name, path in trained_models if name == "regular")
            split_path = next(path for name, path in trained_models if name == "split")
            hybrid_path = self.create_hybrid_model(regular_path, split_path)
            trained_models.append(("hybrid", hybrid_path))
        
        print(f"[SpecializedTrainingManager] 전용 훈련 사이클 완료!")
        print(f"생성된 모델들: {[name for name, _ in trained_models]}")
        
        return trained_models
    
    def visualize_training_results(self):
        """훈련 결과 시각화"""
        print(f"[SpecializedTrainingManager] 훈련 결과 시각화...")
        
        # 로그 파일들 찾기
        log_files = []
        for file in os.listdir(self.models_dir):
            if file.startswith("logs_") and file.endswith(".csv"):
                log_files.append(os.path.join(self.models_dir, file))
        
        if not log_files:
            print("[SpecializedTrainingManager] 시각화할 로그 파일이 없습니다.")
            return
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('전용 훈련소 결과 분석', fontsize=16)
        
        for i, log_file in enumerate(log_files[:4]):  # 최대 4개 파일
            try:
                df = pd.read_csv(log_file)
                model_name = os.path.basename(log_file).replace("logs_", "").replace(".csv", "")
                
                row = i // 2
                col = i % 2
                
                # 보상 그래프
                if 'eval_mean_reward' in df.columns:
                    axes[row, col].plot(df['eval_mean_reward'], label='평균 보상')
                if 'eval_std_reward' in df.columns:
                    axes[row, col].fill_between(range(len(df)), 
                                              df['eval_mean_reward'] - df['eval_std_reward'],
                                              df['eval_mean_reward'] + df['eval_std_reward'], 
                                              alpha=0.3)
                
                axes[row, col].set_title(f'{model_name} 훈련 결과')
                axes[row, col].set_xlabel('평가 횟수')
                axes[row, col].set_ylabel('보상')
                axes[row, col].legend()
                axes[row, col].grid(True)
                
            except Exception as e:
                print(f"[SpecializedTrainingManager] {log_file} 시각화 실패: {e}")
        
        plt.tight_layout()
        plt.savefig(f"{self.models_dir}/training_results.png", dpi=300, bbox_inches='tight')
        print(f"[SpecializedTrainingManager] 시각화 결과 저장: {self.models_dir}/training_results.png")


def main():
    """메인 실행 함수"""
    print("=== AlggaGo 전용 훈련소 시스템 ===")
    
    # 사용 가능한 기본 모델 찾기
    base_models_dir = "rl_models_competitive"
    available_models = []
    
    if os.path.exists(base_models_dir):
        for file in os.listdir(base_models_dir):
            if file.endswith(".zip"):
                available_models.append(os.path.join(base_models_dir, file))
    
    if available_models:
        # 가장 최신 모델 선택
        latest_model = sorted(available_models, key=os.path.getmtime)[-1]
        print(f"기본 모델로 사용: {latest_model}")
        manager = SpecializedTrainingManager(latest_model)
    else:
        print("기본 모델을 찾을 수 없어 새로운 모델로 시작합니다.")
        manager = SpecializedTrainingManager()
    
    # 전용 훈련 사이클 실행
    trained_models = manager.run_specialized_training_cycle()
    
    if trained_models:
        # 결과 시각화
        manager.visualize_training_results()
        
        print("\n=== 훈련 완료 ===")
        print("생성된 모델들:")
        for name, path in trained_models:
            print(f"  - {name}: {path}")
        
        print(f"\n모델들은 '{manager.models_dir}' 폴더에 저장되었습니다.")
        print("이제 이 모델들을 사용하여 메인 훈련을 계속할 수 있습니다.")
    else:
        print("훈련이 필요하지 않거나 실패했습니다.")


if __name__ == "__main__":
    main()
