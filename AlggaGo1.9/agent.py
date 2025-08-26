import numpy as np
import os
from stable_baselines3 import PPO
from pymunk import Vec2d
from physics import scale_force
import re
from opponent import get_regular_action, get_split_shot_action

# --- [추가] 파일 기준 경로 유틸 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def rel_path(*parts):
    return os.path.join(BASE_DIR, *parts)

MODEL_SAVE_DIR = rel_path("rl_models_competitive")

class MainRLAgent:
    """
    메인 강화 학습 모델의 로딩 및 추론 로직을 담당하는 클래스.
    """
    def __init__(self, model_path=None):
        self.model = None
        # --- [수정] 모델 경로를 클래스 변수로 저장 ---
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            try:
                self.model = PPO.load(model_path)
                print(f"[MainRLAgent] 모델 로드 성공: {model_path}")
            except Exception as e:
                print(f"[MainRLAgent] 모델 로드 실패: {model_path} - {e}")
                self.model = None
        elif model_path:
            print(f"[MainRLAgent] 지정된 모델 경로를 찾을 수 없음: {model_path}")

    def select_action(self, observation: np.ndarray):
        """
        주어진 관측을 바탕으로 전체 행동 벡터(5개 값)를 예측합니다.
        AlggaGo1.0 모델의 경우 호환성을 보정합니다.
        """
        if self.model is None:
            print(f"[MainRLAgent] 모델 없음: 순수 rule-based 행동")
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        observation_reshaped = np.expand_dims(observation, axis=0)
        action_raw, _states = self.model.predict(observation_reshaped, deterministic=True)
        action_vector = action_raw[0]
        
        # --- [✅ 핵심 수정] AlggaGo1.0 모델 호환성 처리 ---
        if self.model_path and "AlggaGo1.0" in self.model_path:
            # 1.0 모델이 3개의 값을 반환하는 경우, 5개 값으로 변환
            if action_vector.size == 3:
                print("[MainRLAgent] AlggaGo1.0 호환 모드: '일반 공격'을 강제하고 3-val 출력을 5-val로 변환합니다.")
                # 전략 값: [일반공격 선호도, 틈새공격 선호도] -> [1.0, -1.0]으로 일반 공격 강제
                pref_regular = 1.0
                pref_split = -1.0
                # 모델이 반환한 3개의 파라미터 값
                raw_index, raw_angle, raw_force = action_vector
                # 5개 값의 새로운 행동 벡터 생성
                return np.array([pref_regular, pref_split, raw_index, raw_angle, raw_force], dtype=np.float32)

        # 그 외 최신 모델들은 그대로 5개 값 반환
        return action_vector

def apply_action_to_stone(full_action: np.ndarray, stones: list, target_color_tuple: tuple):
    """
    모델이 예측한 전체 행동 벡터를 해석하여 돌에 impulse를 적용합니다.
    """
    if full_action is None:
        return

    player_stones = [s for s in stones if s.color[:3] == target_color_tuple]
    opponent_stones = [s for s in stones if s.color[:3] != target_color_tuple]

    if not player_stones:
        print(f"[ERROR] apply_action_to_stone: 대상 돌이 없습니다.")
        return

    # --- [✅ 핵심 수정] argmax를 softmax 샘플링 로직으로 교체 ---
    strategy_preferences = np.asarray(full_action[:2], dtype=np.float32)
    
    # 안정화된 소프트맥스 확률 계산 (train.py와 동일한 로직)
    max_pref = float(np.max(strategy_preferences))
    exp_p = np.exp(strategy_preferences - max_pref)
    probs = exp_p / (np.sum(exp_p) + 1e-8)

    # 수치 이상 시 안전하게 argmax로 폴백
    if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        strategy_choice = int(np.argmax(strategy_preferences))
    else:
        # 계산된 확률에 따라 행동을 랜덤하게 선택
        strategy_choice = int(np.random.choice(2, p=probs))
    # --- 수정 끝 ---

    if strategy_choice == 1: # 1번 전략: 틈새 공격
        rule_action = get_split_shot_action(player_stones, opponent_stones)
        if rule_action is None:
            rule_action = get_regular_action(player_stones, opponent_stones)
            print("[Action Strategy] 모델이 '틈새 공격'을 원했으나, 불가능하여 '일반 공격'으로 전환")
        else:
            print("[Action Strategy] 모델이 '틈새 공격(Split Shot)'을 선택")
    else: # 0번 전략: 일반 공격
        rule_action = get_regular_action(player_stones, opponent_stones)
        print("[Action Strategy] 모델이 '일반 공격(Regular Action)'을 선택")

    if rule_action is None:
        print(f"[ERROR] apply_action_to_stone: Rule-based 행동 계산 실패")
        return
        
    raw_index, raw_angle, raw_force = full_action[2:]
    
    index_weight = np.clip(raw_index, -1.0, 1.0)
    angle_offset = np.clip(raw_angle, -1.0, 1.0)
    force_offset = np.clip(raw_force, -1.0, 1.0)
    
    exploration_range = {"index": 1.0, "angle": np.pi / 4, "force": 0.5}

    final_index_weight = index_weight * exploration_range['index']
    final_angle_offset = angle_offset * exploration_range['angle']
    
    final_index_weight = index_weight * exploration_range['index']
    final_angle_offset = angle_offset * exploration_range['angle']
    final_force_offset = force_offset * exploration_range['force']

    print(f"[MainRLAgent] AI 오차 예측 - index_w: {final_index_weight:.3f}, angle_o: {final_angle_offset:.3f}, force_o: {final_force_offset:.3f}")

    rule_idx, rule_angle, rule_force = rule_action

    if len(player_stones) > 1:
        # 2. train.py와 동일하게 scaling 부분 삭제
        idx_offset = int(np.round(final_index_weight))
        final_idx = np.clip(rule_idx + idx_offset, 0, len(player_stones) - 1)
    else:
        final_idx = 0

    final_angle = rule_angle + final_angle_offset
    final_force = np.clip(rule_force + final_force_offset, 0.0, 1.0)

    print(f"[apply_action] Rule: idx={rule_idx}, angle={rule_angle:.2f}, force={rule_force:.2f}")
    print(f"[apply_action] Final: idx={final_idx}, angle={final_angle:.2f}, force={final_force:.2f}")

    stone_to_move = player_stones[final_idx]
    direction = Vec2d(1, 0).rotated(final_angle)
    scaled_force = scale_force(final_force)
    impulse = direction * scaled_force
    stone_to_move.body.apply_impulse_at_world_point(impulse, stone_to_move.body.position)

def choose_ai():
    """
    사용자 입력을 받아 메인 에이전트 모델을 선택합니다.
    (이 함수는 수정할 필요가 없습니다)
    """
    print("🤖 AI 행동 방식을 선택하세요:")
    
    model_pattern = re.compile(r"model_(a|b)_(\d+)_([0-9.]+)\.zip")
    available_models = []

    if os.path.exists(MODEL_SAVE_DIR):
        for filename in sorted(os.listdir(MODEL_SAVE_DIR)):
            match = model_pattern.match(filename)
            if match:
                model_char = match.group(1).upper()
                timesteps = int(match.group(2))
                entropy = float(match.group(3))
                full_path = os.path.join(MODEL_SAVE_DIR, filename)
                available_models.append({
                    "char": model_char, "timesteps": timesteps,
                    "entropy": entropy, "path": full_path
                })
    
    available_models.sort(key=lambda x: x["timesteps"])

    print("   0) 순수 Rule-based (AI 없음)")
    model_options = {"0": None}
    
    if not available_models:
        print("   (사용 가능한 강화학습 모델이 없습니다.)")
    else:
        for i, model_info in enumerate(available_models):
            option_num = i + 1
            print(f"   {option_num}) 강화학습 모델 (Model {model_info['char']} - {model_info['timesteps']} steps, ent={model_info['entropy']:.3f})")
            model_options[str(option_num)] = model_info["path"]

    choice = input("선택 (0" + "".join(f"/{i+1}" for i in range(len(available_models))) + "): ").strip()
    model_path = model_options.get(choice)
    
    if choice in model_options:
        if model_path:
            print(f"[AI 선택] 강화학습 모델 '{os.path.basename(model_path)}'을(를) 사용합니다.")
        else:
            print("[AI 선택] 순수 Rule-based를 사용합니다.")
    else:
        print("[AI 선택] 유효하지 않은 선택입니다. 순수 Rule-based를 사용합니다.")
        model_path = None

    return MainRLAgent(model_path=model_path)