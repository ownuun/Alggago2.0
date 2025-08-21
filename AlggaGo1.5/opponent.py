import numpy as np
import random

# 'is_friendly_fire_risk' 함수는 경로 계산 로직이므로 완전히 제거되었습니다.

def get_regular_action(player_stones, opponent_stones):
    """'일반 공격' 기준 행동: 플레이어의 돌과 상대 돌을 각각 무작위로 선택해 공격"""
    # 공격할 아군 돌이나 상대 돌이 없으면 행동을 할 수 없으므로 None을 반환합니다.
    if not player_stones or not opponent_stones:
        return None

    # 공격할 아군 돌(shooter)과 그 인덱스(p_idx)를 무작위로 선택합니다.
    p_idx, shooter = random.choice(list(enumerate(player_stones)))
    
    # 공격할 상대 돌(target)을 무작위로 선택합니다.
    target = random.choice(opponent_stones)
    
    # 선택된 돌들을 기반으로 공격 각도를 계산합니다.
    diff = target.body.position - shooter.body.position
    angle = diff.angle
    
    # (돌 인덱스, 공격 각도, 힘) 튜플을 반환합니다. 힘은 최대로 설정합니다.
    return (p_idx, angle, 1.0)

def get_split_shot_action(player_stones, opponent_stones):
    """'틈새 공격' 기준 행동"""
    if len(opponent_stones) < 2: return None
    try:
        # 예측 가능하게 첫 두 돌 사용
        stone1, stone2 = opponent_stones[0], opponent_stones[1] 
        target_pos = (stone1.body.position + stone2.body.position) / 2
        
        sorted_player_stones = sorted(player_stones, key=lambda s: (s.body.position - target_pos).length)
        stone_to_shoot = sorted_player_stones[0]
        
        direction_vector = target_pos - stone_to_shoot.body.position
        
        if direction_vector.length > 0:
            rule_idx = player_stones.index(stone_to_shoot)
            rule_angle = direction_vector.angle
            distance_to_target = direction_vector.length
            
            DAMPING = 0.1
            LN_DAMPING = np.log(DAMPING)
            MIN_IMPULSE = 20.0
            MAX_IMPULSE = 2000.0
            
            required_impulse = -distance_to_target * LN_DAMPING
            normalized_force = (required_impulse - MIN_IMPULSE) / (MAX_IMPULSE - MIN_IMPULSE)
            rule_force = np.clip(normalized_force, 0.0, 1.0)
            
            return (rule_idx, rule_angle, rule_force)
            
    except (ValueError, IndexError):
        return None
        
    return None