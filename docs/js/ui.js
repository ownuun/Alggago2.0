// ui.js - UI 컨트롤 및 이벤트 처리
class GameUI {
    constructor() {
        this.game = null;
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // 게임 시작 버튼
        const startBtn = document.getElementById('startBtn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startGame());
        }
        
        // 다시 시작 버튼
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetGame());
        }
        
        // 튜토리얼 버튼
        const tutorialBtn = document.getElementById('tutorialBtn');
        if (tutorialBtn) {
            tutorialBtn.addEventListener('click', () => this.showTutorial());
        }
        
        // 키보드 이벤트
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
    }
    
    initializeGame() {
        const canvas = document.getElementById('gameCanvas');
        if (!canvas) {
            console.error('Canvas element not found!');
            return;
        }
        
        this.game = new AlggaGoGame(canvas);
        this.updateButtonStates();
    }
    
    startGame() {
        if (this.game) {
            this.game.startGame();
            this.updateButtonStates();
        }
    }
    
    resetGame() {
        if (this.game) {
            this.game.resetGame();
            this.updateButtonStates();
        }
    }
    
    showTutorial() {
        this.showTutorialModal();
    }
    
    showTutorialModal() {
        // 간단한 튜토리얼 모달
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;
        
        const content = document.createElement('div');
        content.style.cssText = `
            background-color: #34495e;
            padding: 30px;
            border-radius: 10px;
            max-width: 500px;
            color: white;
            text-align: center;
        `;
        
        content.innerHTML = `
            <h2>AlggaGo 튜토리얼</h2>
            <div style="margin: 20px 0;">
                <h3>게임 방법</h3>
                <ol style="text-align: left; line-height: 1.8;">
                    <li>검은 돌을 클릭하고 아래쪽으로 드래그하세요</li>
                    <li>드래그 거리에 따라 발사 힘이 결정됩니다</li>
                    <li>상대의 모든 돌을 밖으로 내보내면 승리!</li>
                    <li>흑돌과 백돌이 번갈아가며 플레이합니다</li>
                </ol>
            </div>
            <div style="margin: 20px 0;">
                <h3>조작법</h3>
                <ul style="text-align: left; line-height: 1.8;">
                    <li><strong>마우스:</strong> 클릭 + 드래그</li>
                    <li><strong>터치:</strong> 터치 + 드래그 (모바일)</li>
                    <li><strong>ESC:</strong> 게임 일시정지</li>
                </ul>
            </div>
            <button id="closeTutorial" style="
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            ">닫기</button>
        `;
        
        modal.appendChild(content);
        document.body.appendChild(modal);
        
        // 닫기 버튼 이벤트
        const closeBtn = document.getElementById('closeTutorial');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        // 모달 외부 클릭으로 닫기
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }
    
    handleKeyPress(e) {
        switch (e.key) {
            case 'Escape':
                e.preventDefault();
                this.togglePause();
                break;
            case 'r':
            case 'R':
                if (e.ctrlKey) {
                    e.preventDefault();
                    this.resetGame();
                }
                break;
            case ' ':
                e.preventDefault();
                if (this.game && this.game.gameState === 'ready') {
                    this.startGame();
                }
                break;
        }
    }
    
    togglePause() {
        if (!this.game) return;
        
        if (this.game.gameState === 'playing') {
            this.game.gameState = 'paused';
            this.updateStatus('게임 일시정지');
        } else if (this.game.gameState === 'paused') {
            this.game.gameState = 'playing';
            this.updateStatus(`${this.game.currentTurn === 'black' ? '흑돌' : '백돌'} 차례`);
        }
        
        this.updateButtonStates();
    }
    
    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }
    
    updateButtonStates() {
        const startBtn = document.getElementById('startBtn');
        const resetBtn = document.getElementById('resetBtn');
        const tutorialBtn = document.getElementById('tutorialBtn');
        
        if (!this.game) {
            if (startBtn) startBtn.disabled = true;
            if (resetBtn) resetBtn.disabled = true;
            return;
        }
        
        const isPlaying = this.game.gameState === 'playing';
        const isPaused = this.game.gameState === 'paused';
        
        if (startBtn) {
            startBtn.disabled = isPlaying || isPaused;
            startBtn.textContent = isPaused ? '계속하기' : '게임 시작';
        }
        
        if (resetBtn) {
            resetBtn.disabled = this.game.gameState === 'ready';
        }
        
        if (tutorialBtn) {
            tutorialBtn.disabled = isPlaying;
        }
    }
    
    showGameResult(winner, blackScore, whiteScore) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;
        
        const content = document.createElement('div');
        content.style.cssText = `
            background-color: #34495e;
            padding: 30px;
            border-radius: 10px;
            max-width: 400px;
            color: white;
            text-align: center;
        `;
        
        const winnerText = winner === 'black' ? '흑돌 승리!' : '백돌 승리!';
        const winnerColor = winner === 'black' ? '#000000' : '#ffffff';
        
        content.innerHTML = `
            <h2 style="color: ${winnerColor};">${winnerText}</h2>
            <div style="margin: 20px 0;">
                <p>최종 점수</p>
                <p>흑돌: ${blackScore} | 백돌: ${whiteScore}</p>
            </div>
            <div style="margin: 20px 0;">
                <button id="playAgain" style="
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                    margin-right: 10px;
                ">다시하기</button>
                <button id="closeResult" style="
                    background-color: #7f8c8d;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                ">닫기</button>
            </div>
        `;
        
        modal.appendChild(content);
        document.body.appendChild(modal);
        
        // 다시하기 버튼
        const playAgainBtn = document.getElementById('playAgain');
        playAgainBtn.addEventListener('click', () => {
            document.body.removeChild(modal);
            this.resetGame();
        });
        
        // 닫기 버튼
        const closeBtn = document.getElementById('closeResult');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        // 3초 후 자동으로 다시하기
        setTimeout(() => {
            if (document.body.contains(modal)) {
                document.body.removeChild(modal);
                this.resetGame();
            }
        }, 3000);
    }
    
    showLoading() {
        const loading = document.createElement('div');
        loading.id = 'loading';
        loading.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        `;
        
        loading.innerHTML = `
            <div style="
                background-color: #34495e;
                padding: 30px;
                border-radius: 10px;
                color: white;
                text-align: center;
            ">
                <h3>게임 로딩 중...</h3>
                <div style="
                    width: 50px;
                    height: 50px;
                    border: 5px solid #3498db;
                    border-top: 5px solid transparent;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin: 20px auto;
                "></div>
            </div>
        `;
        
        // CSS 애니메이션 추가
        const style = document.createElement('style');
        style.textContent = `
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(loading);
    }
    
    hideLoading() {
        const loading = document.getElementById('loading');
        if (loading) {
            document.body.removeChild(loading);
        }
    }
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    const ui = new GameUI();
    
    // 로딩 표시
    ui.showLoading();
    
    // Matter.js 로드 완료 후 게임 초기화
    setTimeout(() => {
        ui.hideLoading();
        ui.initializeGame();
    }, 1000);
    
    // 전역 변수로 export
    window.gameUI = ui;
});
