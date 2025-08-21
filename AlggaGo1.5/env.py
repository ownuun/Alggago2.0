# env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import pygame 
from pymunk import Vec2d
from opponent import get_regular_action, get_split_shot_action

from physics import (
    create_stone,
    WIDTH, HEIGHT, MARGIN,
    STONE_RADIUS,
    scale_force,
    all_stones_stopped
)

MAX_PHYSICS_STEPS_PER_ACTION = 600

class AlggaGoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 0: 전략 선택 (-1: 일반 공격, 1: 틈새 공격)
        # 1: 쏘는 돌 인덱스 수정값
        # 2: 각도 수정값
        # 3: 힘 수정값
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-1.0] * 25, dtype=np.float32),
            high=np.array([float(max(WIDTH, HEIGHT))] * 25, dtype=np.float32),
            shape=(25,),
            dtype=np.float32
        )
        self.stones = []
        self.space = None
        self.current_player = "black"
        self.exploration_range = {"index": 1.0, "angle": np.pi, "force": 0.5}
        self.reset()

    # === 추가: 외부에서 행동 범위를 설정하기 위한 함수 ===
    def set_exploration_range(self, index_range, angle_range, force_range):
        self.exploration_range['index'] = index_range
        self.exploration_range['angle'] = angle_range
        self.exploration_range['force'] = force_range

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.space:
            for body in list(self.space.bodies): self.space.remove(body)
            for shape in list(self.space.shapes): self.space.remove(shape)
        self.space = pymunk.Space(); self.space.gravity=(0,0); self.space.damping=0.1
        self.stones.clear()
        static_body = self.space.static_body
        corners = [(MARGIN, MARGIN), (WIDTH-MARGIN, MARGIN), (WIDTH-MARGIN, HEIGHT-MARGIN), (MARGIN, HEIGHT-MARGIN)]
        for i in range(4):
            a, b = corners[i], corners[(i+1)%4]
            seg=pymunk.Segment(static_body, a, b, 1); seg.sensor=True; self.space.add(seg)
        y_top_center = MARGIN + STONE_RADIUS*2 + 66.5
        y_bottom_center = HEIGHT - MARGIN - STONE_RADIUS*2 - 66.5
        MIN_DIST = STONE_RADIUS * 3.0
        def is_far_enough(x, y, p):
            for px,py in p:
                if np.linalg.norm([x-px, y-py]) < MIN_DIST: return False
            return True
        w_pos = []; b_pos = []
        while len(w_pos) < 4:
            x = np.random.uniform(MARGIN+STONE_RADIUS, WIDTH-MARGIN-STONE_RADIUS)
            y = y_top_center + np.random.uniform(-40, 40)
            if is_far_enough(x, y, w_pos): w_pos.append((x,y)); self.stones.append(create_stone(self.space, (x, y), (255,255,255)))
        while len(b_pos) < 4:
            x = np.random.uniform(MARGIN+STONE_RADIUS, WIDTH-MARGIN-STONE_RADIUS)
            y = y_bottom_center + np.random.uniform(-40, 40)
            if is_far_enough(x, y, b_pos): b_pos.append((x,y)); self.stones.append(create_stone(self.space, (x, y), (0,0,0)))
        self.initial_black_count = 4; self.initial_white_count = 4
        # 랜덤하게 시작 플레이어 결정 (공정성 확보)
        if options and "initial_player" in options:
            self.current_player = options["initial_player"]
        else:
            self.current_player = np.random.choice(["black", "white"])
        return self._get_obs(), {"initial_player": self.current_player}

    def _get_obs(self):
        obs = []
        sorted_stones = sorted(self.stones, key=lambda s: (s.color[:1], s.body.position.x))
        
        # [수정] 처리할 돌의 개수를 최대 8개로 제한합니다.
        for shape in sorted_stones[:8]:
            if shape.body in self.space.bodies:
                x, y = shape.body.position
                stone_is_white = shape.color[:3] == (255,255,255)
                is_mine = 1.0 if (self.current_player=='white' and stone_is_white) or (self.current_player=='black' and not stone_is_white) else 0.0
                if self.current_player == "black": y = HEIGHT - y
                obs.extend([float(x), float(y), float(is_mine)])
            else:
                # 이 경우는 거의 발생하지 않지만 안전을 위해 유지
                obs.extend([-1.0, -1.0, -1.0])
                
        while len(obs) < 8 * 3: # 24
            obs.extend([-1.0, -1.0, -1.0])
            
        obs.append(1.0 if self.current_player == "white" else 0.0) # 25번째 정보
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        terminated = False; truncated = False; reward = 0.0; info = {}
        
        player_color_tuple = (255,255,255) if self.current_player == "white" else (0,0,0)
        player_stones = [s for s in self.stones if s.color[:3] == player_color_tuple]
        opponent_stones = [s for s in self.stones if s.color[:3] != player_color_tuple]

        if not player_stones or not opponent_stones:
            return self._get_obs(), 0.0, True, False, {}

        # [수정] AI의 출력을 4개로 해석
        strategy_choice, raw_index, raw_angle, raw_force = np.squeeze(action)
        
        # AI의 전략 선택에 따라 기준 행동 결정
        if strategy_choice >= 0:
            rule_action = get_split_shot_action(player_stones, opponent_stones)
        else:
            rule_action = get_regular_action(player_stones, opponent_stones)

        # 기준 행동이 없으면 (예: 틈새 공격 불가 시) 일반 공격으로 대체
        if rule_action is None:
            rule_action = get_regular_action(player_stones, opponent_stones)

        if rule_action is None: # 그래도 없으면 턴 종료
            return self._get_obs(), 0.0, True, False, {}
        
        # --- 이하 로직은 기존과 거의 동일 ---
        index_weight = raw_index * self.exploration_range['index']
        angle_offset = raw_angle * self.exploration_range['angle']
        force_offset = raw_force * self.exploration_range['force']

        rule_idx, rule_angle, rule_force = rule_action
        
        if len(player_stones) > 1:
            idx_offset = int(np.round(index_weight * (len(player_stones)-1)/2))
            final_idx = np.clip(rule_idx + idx_offset, 0, len(player_stones)-1)
        else: final_idx = 0
        
        final_angle = rule_angle + angle_offset
        final_force = np.clip(rule_force + force_offset, 0.0, 1.0)
        
        selected_stone_to_shoot = player_stones[final_idx]
        direction = Vec2d(1, 0).rotated(final_angle)
        impulse = direction * scale_force(final_force)
        selected_stone_to_shoot.body.apply_impulse_at_world_point(impulse, selected_stone_to_shoot.body.position)

        physics_step_count = 0
        while not all_stones_stopped(self.stones) and physics_step_count < MAX_PHYSICS_STEPS_PER_ACTION:
            self.space.step(1/60.0); physics_step_count += 1
            for shape in self.stones[:]:
                if not (MARGIN<shape.body.position.x<WIDTH-MARGIN and MARGIN<shape.body.position.y<HEIGHT-MARGIN):
                    self.space.remove(shape, shape.body); self.stones.remove(shape)
        current_black = sum(1 for s in self.stones if s.color[:3] == (0,0,0))
        current_white = sum(1 for s in self.stones if s.color[:3] == (255,255,255))
        black_removed = self.initial_black_count - current_black
        white_removed = self.initial_white_count - current_white
        reward = 0.0

        # 틈새 파고들기 전략 보상
        moved_stone_final_pos = selected_stone_to_shoot.body.position

        opponent_color_tuple = (0,0,0) if self.current_player == "white" else (255,255,255)
        opponent_stones = [s for s in self.stones if s.color[:3] == opponent_color_tuple]

        # 이 전략에 대한 보상을 0으로 초기화
        wedge_reward = 0.0

        if len(opponent_stones) >= 2:
            opponent_stones.sort(key=lambda s: (s.body.position - moved_stone_final_pos).length)
            
            p1 = opponent_stones[0].body.position
            p2 = opponent_stones[1].body.position
            
            # 내 돌(p3)에서 두 상대 돌(p1, p2)을 잇는 선분까지의 최단 거리를 계산
            p3 = moved_stone_final_pos
            v = p2 - p1
            w = p3 - p1
            
            # t는 p1-p2 선분상에 p3의 수직 발이 떨어지는 위치 (0~1 사이 값)
            t = w.dot(v) / (v.dot(v) + 1e-6) # 0으로 나누는 것 방지
            t = max(0, min(1, t)) # t를 0과 1 사이로 제한하여 선분 안에 있도록 함
            
            closest_point_on_segment = p1 + t * v
            dist_to_segment = (p3 - closest_point_on_segment).length
            
            dist_between_opponents = (p1 - p2).length
            
            # 틈새 전략을 시도할 만한 상황인지 먼저 판단 (상대 돌이 너무 멀지 않을 때)
            if dist_between_opponents < WIDTH / 2:
                # '틈새'로 인정하는 거리 (상대 돌 사이 거리의 15% 이내)
                wedge_threshold = dist_between_opponents * 0.15
                
                # 틈새 안으로 성공적으로 들어갔는지 확인
                if dist_to_segment < wedge_threshold:
                    # 틈새 경로에 가까울수록 더 큰 보상
                    wedge_reward = (1 - (dist_to_segment / wedge_threshold)) * 0.5
                else:
                    # 틈새를 노렸으나 인정 거리 밖으로 벗어난 경우 -2점 페널티
                    wedge_reward = -0.5
        
        reward += wedge_reward

        # 내 턴에 내 돌 죽음 
        if self.current_player == 'black':
            if black_removed == 1:
                reward -= 2
            elif black_removed == 2:
                reward -= 4
            elif black_removed == 3:
                reward -= 6
        else: # 'white'
            if white_removed == 1:
                reward -= 2
            elif white_removed == 2:
                reward -= 4
            elif white_removed == 3:
                reward -= 6

        # 상대 돌 제거 보상
        removed_count = 0
        if self.current_player == "black":
            removed_count = white_removed
        elif self.current_player == "white":
            removed_count = black_removed

        if removed_count == 1:
            reward += 1
        elif removed_count == 2:
            reward += 3
        elif removed_count == 3:
            reward += 5
        elif removed_count >= 4:
            reward += 7 

       # 게임 종료 시 승리 보상 (대칭, 마진 기반: 1.5 + 3 * margin)
        if current_black == 0 and current_white > 0:
            terminated = True
            info['winner'] = 'white'
            margin = current_white  # 남은 백돌 수 = 마진
            W = 1.5 + 3.0 * margin
            reward += W if self.current_player == 'white' else -W

        elif current_white == 0 and current_black > 0:
            terminated = True
            info['winner'] = 'black'
            margin = current_black  # 남은 흑돌 수 = 마진
            W = 1.5 + 3.0 * margin
            reward += W if self.current_player == 'black' else -W

        self.initial_black_count = current_black; self.initial_white_count = current_white
        if not terminated: self.current_player = "white" if self.current_player == "black" else "black"
        info["current_black_stones"] = current_black; info["current_white_stones"] = current_white
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, screen):
        screen.fill((150, 150, 150))
        pygame.draw.rect(screen, (210, 180, 140), pygame.Rect(MARGIN, MARGIN, WIDTH - 2*MARGIN, HEIGHT - 2*MARGIN))
        for shape in self.stones:
            pygame.draw.circle(screen, shape.color[:3], (int(shape.body.position.x), int(shape.body.position.y)), STONE_RADIUS)
    def close(self): pass