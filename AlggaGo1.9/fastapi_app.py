from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import json
import random
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel
import asyncio
from datetime import datetime

# FastAPI 앱 생성
app = FastAPI(
    title="AlggaGo Game API",
    description="AI 강화학습 기반 바둑돌 게임 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Physics-lite server simulation (Option B) ---
TICK_HZ = 30
FRICTION = 0.98
ELASTICITY = 0.9
MIN_SPEED = 2.0

class SimStone:
    def __init__(self, x: float, y: float, color: str):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.color = color

class GameState:
    def __init__(self):
        self.board_width = 800
        self.board_height = 600
        self.stone_radius = 20
        self.reset_game()
        self._tick_task = None
        self._clients: List[WebSocket] = []

    def reset_game(self):
        self.black: List[SimStone] = [
            SimStone(200, 100, "black"), SimStone(400, 100, "black"),
            SimStone(600, 100, "black"), SimStone(300, 150, "black")
        ]
        self.white: List[SimStone] = [
            SimStone(200, 500, "white"), SimStone(400, 500, "white"),
            SimStone(600, 500, "white"), SimStone(300, 450, "white")
        ]
        self.current_player = "black"
        self.game_over = False
        self.winner = None
        self.move_history: List[str] = []
        self.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def serialize(self) -> Dict[str, Any]:
        return {
            "black": [[s.x, s.y] for s in self.black],
            "white": [[s.x, s.y] for s in self.white],
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "board_width": self.board_width,
            "board_height": self.board_height,
            "stone_radius": self.stone_radius,
        }

    # Simple circle collision with walls and stones
    def _integrate(self):
        # Update positions
        for stones in (self.black, self.white):
            for s in stones:
                s.x += s.vx
                s.y += s.vy
                s.vx *= FRICTION
                s.vy *= FRICTION
                # wall bounce
                r = self.stone_radius
                if s.x < r:
                    s.x = r
                    s.vx = -s.vx * ELASTICITY
                if s.x > self.board_width - r:
                    s.x = self.board_width - r
                    s.vx = -s.vx * ELASTICITY
                if s.y < r:
                    s.y = r
                    s.vy = -s.vy * ELASTICITY
                if s.y > self.board_height - r:
                    s.y = self.board_height - r
                    s.vy = -s.vy * ELASTICITY

        # stone-stone simplistic collision (elastic swap along axis)
        def collide(list_a: List[SimStone], list_b: List[SimStone]):
            r2 = (self.stone_radius * 2) ** 2
            for a in list_a:
                for b in list_b:
                    dx = b.x - a.x
                    dy = b.y - a.y
                    d2 = dx*dx + dy*dy
                    if d2 > 0 and d2 < r2:
                        a.vx, b.vx = b.vx, a.vx
                        a.vy, b.vy = b.vy, a.vy
                        # separate
                        dist = np.sqrt(d2) if d2 > 0 else 0.0001
                        overlap = (self.stone_radius*2 - dist) / 2
                        nx, ny = dx/dist, dy/dist
                        a.x -= nx * overlap
                        a.y -= ny * overlap
                        b.x += nx * overlap
                        b.y += ny * overlap

        collide(self.black, self.black)
        collide(self.white, self.white)
        collide(self.black, self.white)

        # stop tiny speeds
        for stones in (self.black, self.white):
            for s in stones:
                if abs(s.vx) < MIN_SPEED and abs(s.vy) < MIN_SPEED:
                    s.vx = s.vy = 0.0

    async def start_ticks(self, broadcaster):
        if self._tick_task:
            return
        async def _run():
            try:
                while True:
                    self._integrate()
                    await broadcaster()
                    await asyncio.sleep(1.0 / TICK_HZ)
            except asyncio.CancelledError:
                pass
        self._tick_task = asyncio.create_task(_run())

    def apply_shot(self, player: str, stone_index: int, angle: float, force: float):
        if self.game_over:
            return
        stones = self.black if player == "black" else self.white
        if not (0 <= stone_index < len(stones)):
            return
        s = stones[stone_index]
        s.vx += force * np.cos(angle)
        s.vy += force * np.sin(angle)
        self.current_player = "white" if self.current_player == "black" else "black"

# 전역 상태
state = GameState()

# WebSocket 매니저
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
    async def broadcast(self, message: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

async def broadcast_state():
    payload = json.dumps({"type": "state", "data": state.serialize()})
    await manager.broadcast(payload)

# API 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "🎯 AlggaGo Game API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "game_state": "/api/game/state",
            "new_game": "/api/game/new",
            "ai_move": "/api/game/ai-move",
            "auto_game": "/api/game/auto",
            "stats": "/api/stats",
            "websocket": "/ws"
        }
    }

@app.get("/api/game/state")
async def get_game_state():
    return state.serialize()

@app.post("/api/game/new")
async def new_game():
    state.reset_game()
    await broadcast_state()
    return {"message": "새 게임이 시작되었습니다!", "game_state": state.serialize()}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# WebSocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # ensure ticks running
    await state.start_ticks(broadcast_state)
    try:
        await ws.send_text(json.dumps({"type": "state", "data": state.serialize()}))
        while True:
            text = await ws.receive_text()
            msg = json.loads(text)
            t = msg.get("type")
            if t == "shot":
                state.apply_shot(
                    player=msg.get("player", state.current_player),
                    stone_index=int(msg.get("stone_index", 0)),
                    angle=float(msg.get("angle", 0.0)),
                    force=float(msg.get("force", 50.0)),
                )
            elif t == "reset":
                state.reset_game()
                await ws.send_text(json.dumps({"type": "state", "data": state.serialize()}))
    except WebSocketDisconnect:
        manager.disconnect(ws)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/game", response_class=HTMLResponse)
async def game_page():
    html_content = """
    <!DOCTYPE html>
    <html lang=\"ko\">
    <head>
        <meta charset=\"UTF-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>AlggaGo - Real-time</title>
        <style>
            body{margin:0;background:#222;color:#fff;font-family:Arial,Helvetica,sans-serif}
            .wrap{max-width:1000px;margin:0 auto;padding:16px}
            canvas{background:#8B4513;border:3px solid #333;border-radius:10px;display:block;margin:12px auto}
            .row{display:flex;gap:8px;justify-content:center}
            button{padding:10px 14px;border:none;border-radius:6px;background:#4CAF50;color:#fff;cursor:pointer}
            input{padding:8px;border-radius:6px;border:1px solid #555;background:#333;color:#fff}
            .stat{display:flex;gap:16px;justify-content:center;margin-top:8px;color:#ddd}
        </style>
    </head>
    <body>
        <div class=\"wrap\">
            <h2>AlggaGo - 서버 시뮬레이션(WebSocket)</h2>
            <canvas id=\"board\" width=\"800\" height=\"600\"></canvas>
            <div class=\"row\">
                <label>Stone # <input id=\"stoneIdx\" type=\"number\" value=\"0\" min=\"0\" step=\"1\" style=\"width:80px\"/></label>
                <label>Angle(rad) <input id=\"angle\" type=\"number\" value=\"0.0\" step=\"0.1\" style=\"width:100px\"/></label>
                <label>Force <input id=\"force\" type=\"number\" value=\"60\" step=\"1\" style=\"width:80px\"/></label>
                <button id=\"shoot\">Shoot</button>
                <button id=\"reset\">Reset</button>
            </div>
            <div class=\"stat\"><span id=\"turn\">turn: -</span><span id=\"winner\"></span></div>
        </div>
        <script>
            const canvas = document.getElementById('board');
            const ctx = canvas.getContext('2d');
            const r = 20;
            let ws;
            let state = null;

            function draw(){
                if(!state){return}
                ctx.clearRect(0,0,canvas.width,canvas.height);
                function drawSet(arr,color){
                    ctx.fillStyle = color==='black'? '#000':'#fff';
                    ctx.strokeStyle = '#000';
                    arr.forEach((p,i)=>{
                        ctx.beginPath();
                        ctx.arc(p[0],p[1],r,0,Math.PI*2);
                        ctx.fill();
                        ctx.stroke();
                        ctx.fillStyle = color==='black'? '#fff':'#000';
                        ctx.font = '12px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText(i+1, p[0], p[1]+4);
                        ctx.fillStyle = color==='black'? '#000':'#fff';
                    });
                }
                drawSet(state.black,'black');
                drawSet(state.white,'white');
                document.getElementById('turn').textContent = 'turn: ' + state.current_player;
                document.getElementById('winner').textContent = state.winner? ('winner: '+state.winner):'';
            }

            function connect(){
                ws = new WebSocket((location.protocol==='https:'?'wss':'ws') + '://' + location.host + '/ws');
                ws.onopen = ()=>{};
                ws.onmessage = (ev)=>{
                    const msg = JSON.parse(ev.data);
                    if(msg.type==='state'){
                        state = msg.data; draw();
                    }
                };
                ws.onclose = ()=>{ setTimeout(connect, 1000);};
            }
            connect();

            document.getElementById('shoot').onclick = ()=>{
                const stone_index = parseInt(document.getElementById('stoneIdx').value)||0;
                const angle = parseFloat(document.getElementById('angle').value)||0;
                const force = parseFloat(document.getElementById('force').value)||60;
                ws?.send(JSON.stringify({type:'shot', player: (state?.current_player||'black'), stone_index, angle, force}));
            };
            document.getElementById('reset').onclick = ()=> ws?.send(JSON.stringify({type:'reset'}));
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
