#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlggaGo 전용 훈련소 시스템 실행 스크립트

사용법:
    python run_specialized_training.py [옵션]

옵션:
    --model PATH     기본 모델 경로 (기본값: 가장 최신 모델)
    --log PATH       가운틀 로그 파일 경로 (기본값: rl_logs_competitive/gauntlet_log.csv)
    --threshold-regular VALUE  일반공격 성공률 임계값 (기본값: 0.3)
    --threshold-split VALUE    틈새공격 성공률 임계값 (기본값: 0.2)
    --timesteps VALUE          훈련 타임스텝 (기본값: 50000)
    --help                     도움말 표시

예시:
    python run_specialized_training.py
    python run_specialized_training.py --model rl_models_competitive/model_a_100000_0.050.zip
    python run_specialized_training.py --threshold-regular 0.4 --threshold-split 0.3
"""

import os
import sys
import argparse
from specialized_training_manager import SpecializedTrainingManager


def find_latest_model(models_dir="rl_models_competitive"):
    """가장 최신 모델 찾기"""
    if not os.path.exists(models_dir):
        return None
    
    model_files = []
    for file in os.listdir(models_dir):
        if file.endswith(".zip"):
            file_path = os.path.join(models_dir, file)
            model_files.append((file_path, os.path.getmtime(file_path)))
    
    if not model_files:
        return None
    
    # 수정 시간 기준으로 정렬하여 가장 최신 모델 반환
    latest_model = sorted(model_files, key=lambda x: x[1])[-1][0]
    return latest_model


def main():
    parser = argparse.ArgumentParser(
        description="AlggaGo 전용 훈련소 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="기본 모델 경로 (기본값: 가장 최신 모델)"
    )
    
    parser.add_argument(
        "--log", 
        type=str, 
        default="rl_logs_competitive/gauntlet_log.csv",
        help="가운틀 로그 파일 경로"
    )
    
    parser.add_argument(
        "--threshold-regular", 
        type=float, 
        default=0.3,
        help="일반공격 성공률 임계값 (기본값: 0.3)"
    )
    
    parser.add_argument(
        "--threshold-split", 
        type=float, 
        default=0.2,
        help="틈새공격 성공률 임계값 (기본값: 0.2)"
    )
    
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=50000,
        help="훈련 타임스텝 (기본값: 50000)"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="훈련 결과 시각화"
    )
    
    args = parser.parse_args()
    
    print("=== AlggaGo 전용 훈련소 시스템 ===")
    
    # 기본 모델 경로 결정
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model()
        if model_path:
            print(f"자동으로 최신 모델을 선택했습니다: {model_path}")
        else:
            print("사용 가능한 모델을 찾을 수 없습니다. 새로운 모델로 시작합니다.")
            model_path = None
    else:
        if not os.path.exists(model_path):
            print(f"지정된 모델 파일을 찾을 수 없습니다: {model_path}")
            return 1
    
    # 전용 훈련소 매니저 초기화
    manager = SpecializedTrainingManager(model_path)
    
    # 임계값 설정
    manager.regular_success_threshold = args.threshold_regular
    manager.split_success_threshold = args.threshold_split
    manager.training_timesteps = args.timesteps
    
    print(f"설정:")
    print(f"  - 일반공격 성공률 임계값: {manager.regular_success_threshold:.3f}")
    print(f"  - 틈새공격 성공률 임계값: {manager.split_success_threshold:.3f}")
    print(f"  - 훈련 타임스텝: {manager.training_timesteps:,}")
    print(f"  - 로그 파일: {args.log}")
    
    # 성능 분석
    print(f"\n=== 성능 분석 ===")
    regular_success_rate, split_success_rate = manager.analyze_performance(args.log)
    
    if regular_success_rate is None or split_success_rate is None:
        print("❌ 성능 분석에 실패했습니다.")
        return 1
    
    # 훈련 필요성 확인
    needs_regular = manager.needs_regular_training(regular_success_rate)
    needs_split = manager.needs_split_training(split_success_rate)
    
    print(f"\n=== 훈련 필요성 분석 ===")
    print(f"일반공격 훈련 필요: {'✅ 예' if needs_regular else '❌ 아니오'}")
    print(f"틈새공격 훈련 필요: {'✅ 예' if needs_split else '❌ 아니오'}")
    
    if not needs_regular and not needs_split:
        print("\n✅ 모든 성공률이 충분합니다. 전용 훈련이 필요하지 않습니다.")
        return 0
    
    # 사용자 확인
    if needs_regular or needs_split:
        print(f"\n전용 훈련을 시작하시겠습니까? (y/N): ", end="")
        response = input().strip().lower()
        if response not in ['y', 'yes']:
            print("전용 훈련을 취소했습니다.")
            return 0
    
    # 전용 훈련 실행
    print(f"\n=== 전용 훈련 시작 ===")
    trained_models = manager.run_specialized_training_cycle(args.log)
    
    if trained_models:
        print(f"\n✅ 전용 훈련 완료!")
        print(f"생성된 모델들:")
        for name, path in trained_models:
            print(f"  - {name}: {path}")
        
        if args.visualize:
            print(f"\n=== 결과 시각화 ===")
            manager.visualize_training_results()
        
        print(f"\n모델들은 '{manager.models_dir}' 폴더에 저장되었습니다.")
        return 0
    else:
        print(f"\n❌ 전용 훈련에 실패했습니다.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n전용 훈련이 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n전용 훈련 중 오류가 발생했습니다: {e}")
        sys.exit(1)
