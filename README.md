Linkid-AI - LangGraph Multi-Agent

## 주요 기능
- 부모-아이 대화 입력을 기반으로 멀티 노드 병렬 처리
  - `sentiment_labeler`: DPICS 스타일 라벨링 생성
  - `highlight_extractor`: 발화 하이라이트 자동 추출(라인 인덱스 기반)
  - `parenting_advice`: 라벨과 하이라이트를 참조한 한국어 코칭 조언 생성
- LangGraph로 병렬 실행 후 단일 노드에서 조합해 결과 제공

## 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 선택: 예시 파일이 있다면 복사
# cp .env.example .env
```

## 환경 변수
필수/선택 환경 변수는 다음과 같습니다. 사용하려는 모델 제공자에 맞춰 설정하세요.

- MODEL_PROVIDER: `openai` | `anthropic` | `google` | `ollama` (기본: `openai`)
- MODEL_NAME: 기본 모델명 (예: OpenAI `gpt-4o-mini`)
- MINI_MODEL_NAME: 경량 모델명 (하이라이트 추출에 사용)
- OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY: 제공자별 API 키
- OLLAMA_BASE_URL: Ollama 사용 시 (기본: `http://localhost:11434`)

민감정보는 `.env`에 보관하고 Git에 커밋하지 마세요. 실행 전 로드됩니다.

## 실행 개요
- 메시지를 그래프에 전달하면 `sentiment_labeler`와 `highlight_extractor`가 병렬 실행되고 결과가 `parenting_advice`로 합류합니다.
- 간단 실행:
  - 모듈 실행: `python -m src.graph "부모: ...\n아이: ..."`
  - 스크립트 실행: `python src/graph.py "부모: ...\n아이: ..."`

## 디렉터리
- `data/ddl`: TDL/DDL JSON 예시
- `data/sql`: 초기 SQL 스크립트
- `src/router`: 그래프/상태 정의
- `src/expert`: 에이전트 노드 구현(하이라이트, 라벨링, 코칭)
- `src/utils`: 공통 유틸 및 LLM 헬퍼
- `src/vs`: TDL/DDL 헬퍼

## Docker
```bash
docker build -t linkid-multi-agent .
docker run --rm --env-file .env linkid-multi-agent
```

## 보안 주의
- `.env` 파일과 API 키는 절대 커밋하지 마세요.
- CI/CD나 배포 시 런타임 환경 변수로 주입하거나 `--env-file`을 사용하세요.
