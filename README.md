# {{project_name}} - LangGraph Multi-Agent Scaffold

이 프로젝트는 LangGraph 기반의 Router → SQL Agent / Risk Agent 분기 멀티 에이전트 스캐폴드입니다.

## 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 실행 개요
- 라우터에 메시지를 전달하면 인텐트(TEXT_TO_SQL, RISK_SENSING, GENERAL)로 분류되어 각 에이전트로 분기합니다.
- 간단 실행: `python -m src.graph` 참고.

## 디렉터리
- `data/ddl`: TDL/DDL JSON 예시
- `data/sql`: 초기 SQL 스크립트
- `src/router`: 라우터/인텐트/상태
- `src/sql`: SQL 에이전트 구성 요소
- `src/risk`: Risk Sensing 에이전트 구성 요소
- `src/utils`: 공통 유틸/LLM 헬퍼

## Docker
```bash
docker build -t lg-multi-agent .
docker run --rm --env-file .env lg-multi-agent
```
