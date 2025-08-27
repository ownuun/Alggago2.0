import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import sys
import os

# FastAPI 앱 생성
app = FastAPI(title="AlggaGo Original Game", version="1.0.0")

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def game_page():
    """원래 AlggaGo 게임 웹 페이지"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🎯 AlggaGo - 원래 게임</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 600px;
            }
            .header {
                margin-bottom: 30px;
            }
            .header h1 {
                color: #333;
                font-size: 2.5rem;
                margin: 0;
            }
            .game-info {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .btn {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 15px 30px;
                margin: 10px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                transition: background 0.3s;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover {
                background: #45a049;
            }
            .btn.download {
                background: #2196F3;
            }
            .btn.download:hover {
                background: #1976D2;
            }
            .features {
                text-align: left;
                margin: 20px 0;
            }
            .features h3 {
                color: #333;
                margin-bottom: 10px;
            }
            .features ul {
                color: #666;
                line-height: 1.6;
            }
            .warning {
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 AlggaGo</h1>
                <p>AI 강화학습 기반 바둑돌 게임</p>
            </div>
            
            <div class="game-info">
                <h3>🎮 원래 게임 특징</h3>
                <div class="features">
                    <ul>
                        <li><strong>새 창 실행:</strong> 데스크톱 게임 창이 별도로 열림</li>
                        <li><strong>물리 엔진:</strong> Pymunk 기반 실제 물리 시뮬레이션</li>
                        <li><strong>AI 모델:</strong> 강화학습 훈련된 AI 에이전트</li>
                        <li><strong>전략 게임:</strong> 바둑돌을 이용한 전략적 대결</li>
                        <li><strong>실시간 플레이:</strong> 마우스로 직접 조작</li>
                    </ul>
                </div>
            </div>
            
            <div class="warning">
                <strong>⚠️ 주의사항:</strong> 이 게임은 데스크톱 애플리케이션입니다. 
                웹 브라우저에서는 실행할 수 없으므로 로컬에서 다운로드하여 실행해주세요.
            </div>
            
            <div>
                <a href="/download" class="btn download">📥 게임 다운로드</a>
                <a href="/docs" class="btn">📚 API 문서</a>
                <a href="https://github.com/ownuun/Alggago2.0" class="btn" target="_blank">🐙 GitHub</a>
            </div>
            
            <div style="margin-top: 30px; color: #666;">
                <p><strong>실행 방법:</strong></p>
                <ol style="text-align: left; display: inline-block;">
                    <li>게임 다운로드</li>
                    <li>Python 3.8+ 설치</li>
                    <li>의존성 설치: <code>pip install -r requirements.txt</code></li>
                    <li>게임 실행: <code>python main.py</code></li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/download")
async def download_game():
    """게임 파일 다운로드 페이지"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AlggaGo 게임 다운로드</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 600px;
            }
            .download-section {
                margin: 20px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            .btn {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 15px 30px;
                margin: 10px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover {
                background: #45a049;
            }
            .instructions {
                text-align: left;
                margin: 20px 0;
                padding: 20px;
                background: #e8f5e8;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📥 AlggaGo 게임 다운로드</h1>
            
            <div class="download-section">
                <h3>🎮 게임 실행 파일</h3>
                <p>원래 AlggaGo 게임을 로컬에서 실행하세요.</p>
                <a href="https://github.com/ownuun/Alggago2.0/archive/refs/heads/main.zip" class="btn" target="_blank">
                    📦 전체 프로젝트 다운로드 (ZIP)
                </a>
            </div>
            
            <div class="instructions">
                <h3>📋 설치 및 실행 방법</h3>
                <ol>
                    <li><strong>다운로드:</strong> 위 버튼을 클릭하여 ZIP 파일 다운로드</li>
                    <li><strong>압축 해제:</strong> 다운로드한 ZIP 파일을 원하는 폴더에 압축 해제</li>
                    <li><strong>Python 설치:</strong> Python 3.8 이상이 설치되어 있어야 합니다</li>
                    <li><strong>의존성 설치:</strong> 터미널/명령 프롬프트에서 다음 명령어 실행:
                        <br><code>pip install pygame pymunk gymnasium stable-baselines3 torch numpy</code></li>
                    <li><strong>게임 실행:</strong> AlggaGo1.9 폴더에서 다음 명령어 실행:
                        <br><code>python main.py</code></li>
                </ol>
            </div>
            
            <div>
                <a href="/" class="btn">🏠 홈으로 돌아가기</a>
                <a href="/docs" class="btn">📚 API 문서</a>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/game-info")
async def get_game_info():
    """게임 정보 API"""
    return {
        "game_name": "AlggaGo",
        "version": "1.9",
        "description": "AI 강화학습 기반 바둑돌 게임",
        "type": "desktop_game",
        "features": [
            "새 창에서 실행되는 데스크톱 게임",
            "Pymunk 물리 엔진",
            "강화학습 AI 모델",
            "실시간 마우스 조작",
            "전략적 바둑돌 게임"
        ],
        "requirements": [
            "Python 3.8+",
            "pygame",
            "pymunk", 
            "gymnasium",
            "stable-baselines3",
            "torch",
            "numpy"
        ],
        "download_url": "https://github.com/ownuun/Alggago2.0/archive/refs/heads/main.zip",
        "github_url": "https://github.com/ownuun/Alggago2.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
