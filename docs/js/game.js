// game.js - 메인 게임 로직
class AlggaGoGame {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        // 물리 엔진 초기화
        this.physics = new PhysicsEngine();
        
        // 게임 상태
        this.gameState = 'ready'; // ready, playing, paused, gameOver
        this.currentTurn = 'black'; // black, white
        this.dragging = false;
        this.dragStart = null;
        this.dragEnd = null;
        this.selectedStone = null;
        
        // 점수
        this.blackScore = 0;
        this.whiteScore = 0;
        
        // 이벤트 리스너 설정
        this.setupEventListeners();
        
        // 게임 루프 시작
        this.gameLoop();
    }
    
    setupEventListeners() {
        // 마우스 이벤트
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        
        // 터치 이벤트 (모바일 지원)
        this.canvas.addEventListener('touchstart', (e) => this.onTouchStart(e));
        this.canvas.addEventListener('touchmove', (e) => this.onTouchMove(e));
        this.canvas.addEventListener('touchend', (e) => this.onTouchEnd(e));
    }
    
    getMousePos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    getTouchPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const touch = e.touches[0];
        return {
            x: touch.clientX - rect.left,
            y: touch.clientY - rect.top
        };
    }
    
    onMouseDown(e) {
        if (this.gameState !== 'playing') return;
        
        const pos = this.getMousePos(e);
        const stone = this.getStoneAtPosition(pos.x, pos.y);
        
        if (stone && stone.color === this.currentTurn) {
            this.dragging = true;
            this.dragStart = pos;
            this.selectedStone = stone;
        }
    }
    
    onMouseMove(e) {
        if (!this.dragging) return;
        
        const pos = this.getMousePos(e);
        this.dragEnd = pos;
    }
    
    onMouseUp(e) {
        if (!this.dragging) return;
        
        const pos = this.getMousePos(e);
        this.dragEnd = pos;
        this.shootStone();
        
        this.dragging = false;
        this.dragStart = null;
        this.dragEnd = null;
        this.selectedStone = null;
    }
    
    onTouchStart(e) {
        e.preventDefault();
        const pos = this.getTouchPos(e);
        const stone = this.getStoneAtPosition(pos.x, pos.y);
        
        if (stone && stone.color === this.currentTurn) {
            this.dragging = true;
            this.dragStart = pos;
            this.selectedStone = stone;
        }
    }
    
    onTouchMove(e) {
        e.preventDefault();
        if (!this.dragging) return;
        
        const pos = this.getTouchPos(e);
        this.dragEnd = pos;
    }
    
    onTouchEnd(e) {
        e.preventDefault();
        if (!this.dragging) return;
        
        const pos = this.getTouchPos(e);
        this.dragEnd = pos;
        this.shootStone();
        
        this.dragging = false;
        this.dragStart = null;
        this.dragEnd = null;
        this.selectedStone = null;
    }
    
    getStoneAtPosition(x, y) {
        return this.physics.stones.find(stone => {
            const dx = stone.position.x - x;
            const dy = stone.position.y - y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            return distance <= STONE_RADIUS;
        });
    }
    
    shootStone() {
        if (!this.selectedStone || !this.dragStart || !this.dragEnd) return;
        
        const impulse = this.physics.calculateImpulse(this.dragStart, this.dragEnd);
        this.physics.applyImpulse(this.selectedStone, impulse);
        
        // 턴 변경
        this.currentTurn = this.currentTurn === 'black' ? 'white' : 'black';
        
        // UI 업데이트
        this.updateStatus();
    }
    
    startGame() {
        this.gameState = 'playing';
        this.currentTurn = 'black';
        this.blackScore = 0;
        this.whiteScore = 0;
        
        // 돌들 초기화
        this.physics.resetStones();
        
        this.updateStatus();
    }
    
    resetGame() {
        this.startGame();
    }
    
    updateStatus() {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            const turnText = this.currentTurn === 'black' ? '흑돌' : '백돌';
            statusElement.textContent = `${turnText} 차례`;
        }
    }
    
    checkGameOver() {
        const blackStones = this.physics.getStonesByColor('black');
        const whiteStones = this.physics.getStonesByColor('white');
        
        if (blackStones.length === 0) {
            this.gameState = 'gameOver';
            this.whiteScore++;
            this.showGameResult('white');
            return true;
        }
        
        if (whiteStones.length === 0) {
            this.gameState = 'gameOver';
            this.blackScore++;
            this.showGameResult('black');
            return true;
        }
        
        return false;
    }
    
    showGameResult(winner) {
        const winnerText = winner === 'black' ? '흑돌 승리!' : '백돌 승리!';
        alert(winnerText);
        
        // 3초 후 자동으로 새 게임 시작
        setTimeout(() => {
            this.startGame();
        }, 3000);
    }
    
    drawBoard() {
        // 바둑판 배경
        this.ctx.fillStyle = '#d2b48c';
        this.ctx.fillRect(MARGIN, MARGIN, WIDTH - 2 * MARGIN, HEIGHT - 2 * MARGIN);
        
        // 격자선 그리기
        this.drawGrid();
    }
    
    drawGrid() {
        const cell = (WIDTH - 2 * MARGIN) / 18;
        
        this.ctx.strokeStyle = '#8b4513';
        this.ctx.lineWidth = 1;
        
        // 세로 격자선
        for (let i = 0; i <= 18; i++) {
            const x = MARGIN + i * cell;
            const lineWidth = (i === 0 || i === 18) ? 3 : 1;
            this.ctx.lineWidth = lineWidth;
            
            this.ctx.beginPath();
            this.ctx.moveTo(x, MARGIN);
            this.ctx.lineTo(x, HEIGHT - MARGIN);
            this.ctx.stroke();
        }
        
        // 가로 격자선
        for (let j = 0; j <= 18; j++) {
            const y = MARGIN + j * cell;
            const lineWidth = (j === 0 || j === 18) ? 3 : 1;
            this.ctx.lineWidth = lineWidth;
            
            this.ctx.beginPath();
            this.ctx.moveTo(MARGIN, y);
            this.ctx.lineTo(WIDTH - MARGIN, y);
            this.ctx.stroke();
        }
        
        // 화점 그리기
        this.ctx.fillStyle = '#000000';
        const starPoints = [3, 9, 15];
        starPoints.forEach(i => {
            starPoints.forEach(j => {
                const x = MARGIN + i * cell;
                const y = MARGIN + j * cell;
                this.ctx.beginPath();
                this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
                this.ctx.fill();
            });
        });
    }
    
    drawStones() {
        this.physics.stones.forEach(stone => {
            const x = stone.position.x;
            const y = stone.position.y;
            
            // 돌 그림자
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            this.ctx.beginPath();
            this.ctx.arc(x + 2, y + 2, STONE_RADIUS, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // 돌 그라데이션
            const gradient = this.ctx.createRadialGradient(
                x - STONE_RADIUS/3, y - STONE_RADIUS/3, 0,
                x, y, STONE_RADIUS
            );
            
            if (stone.color === 'white') {
                gradient.addColorStop(0, '#ffffff');
                gradient.addColorStop(1, '#cccccc');
            } else {
                gradient.addColorStop(0, '#666666');
                gradient.addColorStop(1, '#000000');
            }
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, STONE_RADIUS, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // 돌 테두리
            this.ctx.strokeStyle = '#333333';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
    }
    
    drawDragLine() {
        if (!this.dragging || !this.dragStart || !this.dragEnd) return;
        
        // 드래그 선 그리기
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(this.dragStart.x, this.dragStart.y);
        this.ctx.lineTo(this.dragEnd.x, this.dragEnd.y);
        this.ctx.stroke();
        
        // 화살표 그리기
        this.drawArrow(this.dragStart, this.dragEnd);
    }
    
    drawArrow(start, end) {
        const headLength = 15;
        const angle = Math.atan2(end.y - start.y, end.x - start.x);
        
        this.ctx.strokeStyle = '#ff0000';
        this.ctx.lineWidth = 3;
        
        this.ctx.beginPath();
        this.ctx.moveTo(end.x, end.y);
        this.ctx.lineTo(
            end.x - headLength * Math.cos(angle - Math.PI / 6),
            end.y - headLength * Math.sin(angle - Math.PI / 6)
        );
        this.ctx.moveTo(end.x, end.y);
        this.ctx.lineTo(
            end.x - headLength * Math.cos(angle + Math.PI / 6),
            end.y - headLength * Math.sin(angle + Math.PI / 6)
        );
        this.ctx.stroke();
    }
    
    render() {
        // 캔버스 클리어
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 바둑판 그리기
        this.drawBoard();
        
        // 돌들 그리기
        this.drawStones();
        
        // 드래그 선 그리기
        this.drawDragLine();
    }
    
    gameLoop() {
        // 물리 업데이트
        this.physics.update();
        
        // 게임 상태 체크
        if (this.gameState === 'playing') {
            if (this.physics.allStonesStopped()) {
                this.checkGameOver();
            }
        }
        
        // 렌더링
        this.render();
        
        // 다음 프레임 요청
        requestAnimationFrame(() => this.gameLoop());
    }
}

// 전역 변수로 export
window.AlggaGoGame = AlggaGoGame;
