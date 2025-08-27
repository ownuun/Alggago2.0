// physics.js - Matter.js 기반 물리 엔진
const { Engine, Render, World, Bodies, Body, Vector, Composite } = Matter;

// 게임 설정 (Python physics.py와 동일)
const WIDTH = 800;
const HEIGHT = 800;
const MARGIN = 50;
const STONE_RADIUS = 25;
const STONE_MASS = 1;

// 드래그 관련 상수
const MAX_DRAG_LENGTH = 100;
const FORCE_MULTIPLIER = 20;
const MIN_FORCE = 20;

class PhysicsEngine {
    constructor() {
        this.engine = Engine.create();
        this.world = this.engine.world;
        
        // 중력 제거 (AlggaGo는 중력이 없는 게임)
        this.world.gravity.y = 0;
        
        // 경계 벽 생성
        this.createWalls();
        
        // 돌들을 저장할 배열
        this.stones = [];
        
        // 충돌 감지 설정
        this.setupCollisions();
    }
    
    createWalls() {
        const wallThickness = 10;
        const wallOptions = { 
            isStatic: true, 
            render: { fillStyle: '#34495e' } 
        };
        
        // 상단 벽
        this.topWall = Bodies.rectangle(WIDTH/2, MARGIN/2, WIDTH - 2*MARGIN, wallThickness, wallOptions);
        
        // 하단 벽
        this.bottomWall = Bodies.rectangle(WIDTH/2, HEIGHT - MARGIN/2, WIDTH - 2*MARGIN, wallThickness, wallOptions);
        
        // 좌측 벽
        this.leftWall = Bodies.rectangle(MARGIN/2, HEIGHT/2, wallThickness, HEIGHT - 2*MARGIN, wallOptions);
        
        // 우측 벽
        this.rightWall = Bodies.rectangle(WIDTH - MARGIN/2, HEIGHT/2, wallThickness, HEIGHT - 2*MARGIN, wallOptions);
        
        Composite.add(this.world, [this.topWall, this.bottomWall, this.leftWall, this.rightWall]);
    }
    
    setupCollisions() {
        // 충돌 이벤트 리스너
        Matter.Events.on(this.engine, 'collisionStart', (event) => {
            // 충돌 효과음 재생 (나중에 구현)
            console.log('Collision detected!');
        });
    }
    
    createStone(x, y, color) {
        const stone = Bodies.circle(x, y, STONE_RADIUS, {
            mass: STONE_MASS,
            restitution: 1.0, // 탄성
            friction: 0.9,
            render: {
                fillStyle: color === 'white' ? '#ffffff' : '#000000',
                strokeStyle: '#333333',
                lineWidth: 2
            }
        });
        
        // 돌에 메타데이터 추가
        stone.color = color;
        stone.isStone = true;
        
        Composite.add(this.world, stone);
        this.stones.push(stone);
        
        return stone;
    }
    
    resetStones() {
        // 기존 돌들 제거
        this.stones.forEach(stone => {
            Composite.remove(this.world, stone);
        });
        this.stones = [];
        
        // 격자 계산
        const cell = (WIDTH - 2 * MARGIN) / 18;
        const xPositions = [];
        for (let i = 3; i <= 15; i += 4) {
            xPositions.push(MARGIN + cell * i);
        }
        
        const yTop = MARGIN + STONE_RADIUS * 2 + 66.5;
        const yBottom = HEIGHT - MARGIN - STONE_RADIUS * 2 - 66.5;
        
        // 백돌 (상단)
        xPositions.forEach(x => {
            this.createStone(x, yTop, 'white');
        });
        
        // 흑돌 (하단)
        xPositions.forEach(x => {
            this.createStone(x, yBottom, 'black');
        });
        
        return { white: 4, black: 4 };
    }
    
    applyImpulse(stone, impulse) {
        Body.applyForce(stone, stone.position, impulse);
    }
    
    allStonesStopped() {
        const velocityThreshold = 0.5;
        return this.stones.every(stone => {
            const velocity = stone.velocity;
            return Math.abs(velocity.x) < velocityThreshold && Math.abs(velocity.y) < velocityThreshold;
        });
    }
    
    getStonesByColor(color) {
        return this.stones.filter(stone => stone.color === color);
    }
    
    removeStone(stone) {
        Composite.remove(this.world, stone);
        const index = this.stones.indexOf(stone);
        if (index > -1) {
            this.stones.splice(index, 1);
        }
    }
    
    update() {
        Engine.update(this.engine, 1000 / 60); // 60fps
        
        // 화면 밖으로 나간 돌들 제거
        this.stones = this.stones.filter(stone => {
            const pos = stone.position;
            if (pos.x < 0 || pos.x > WIDTH || pos.y < 0 || pos.y > HEIGHT) {
                Composite.remove(this.world, stone);
                return false;
            }
            return true;
        });
    }
    
    // 드래그 힘 계산
    calculateImpulse(dragStart, dragEnd) {
        const dragVector = Vector.sub(dragStart, dragEnd);
        const length = Vector.magnitude(dragVector);
        
        if (length === 0) return Vector.create(0, 0);
        
        const clampedLength = Math.min(length, MAX_DRAG_LENGTH);
        const normalizedVector = Vector.normalise(dragVector);
        const force = (clampedLength * FORCE_MULTIPLIER + MIN_FORCE) / 1000; // 스케일 조정
        
        return Vector.mult(normalizedVector, force);
    }
}

// 전역 변수로 export
window.PhysicsEngine = PhysicsEngine;
window.GAME_CONSTANTS = {
    WIDTH, HEIGHT, MARGIN, STONE_RADIUS, STONE_MASS,
    MAX_DRAG_LENGTH, FORCE_MULTIPLIER, MIN_FORCE
};
