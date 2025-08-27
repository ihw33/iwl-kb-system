# IWL Knowledge Base System

## 🎯 목적
IWL(IdeaWorkLab) v5.0의 Knowledge Base 시스템 - RAG 기반 교육 콘텐츠 관리 및 검색

## 🏗️ 아키텍처

### 핵심 구성요소
1. **Content Management** - 교육 콘텐츠 관리
2. **Vector Database** - 임베딩 저장 및 검색 (Pinecone/Chroma)
3. **Embedding Pipeline** - 콘텐츠 벡터화
4. **Search API** - 의미 기반 검색
5. **RAG System** - Retrieval-Augmented Generation

## 📁 프로젝트 구조

```
iwl-kb-system/
├── docs/                  # 문서 및 설계
│   ├── architecture.md   # 시스템 아키텍처
│   ├── api-spec.md       # API 명세
│   └── setup.md          # 설정 가이드
├── src/                  # 소스 코드
│   ├── embeddings/       # 임베딩 생성
│   ├── vectordb/         # Vector DB 관리
│   ├── search/           # 검색 엔진
│   ├── rag/             # RAG 파이프라인
│   └── api/             # REST API
├── content/             # 교육 콘텐츠
│   ├── courses/         # 코스 자료
│   ├── tutorials/       # 튜토리얼
│   ├── references/      # 참고 자료
│   └── templates/       # 템플릿
├── tests/               # 테스트
├── scripts/             # 유틸리티 스크립트
└── config/              # 설정 파일
```

## 🚀 주요 기능

### Phase 1: 기반 구축 (현재)
- [ ] Vector DB 선택 및 설정 (Pinecone vs Chroma)
- [ ] 기본 임베딩 파이프라인
- [ ] 콘텐츠 인덱싱 시스템
- [ ] 기본 검색 API

### Phase 2: RAG 구현
- [ ] LLM 연동 (GPT-4, Claude, Gemini)
- [ ] 컨텍스트 검색 최적화
- [ ] 프롬프트 템플릿 설계
- [ ] 응답 품질 평가

### Phase 3: 고급 기능
- [ ] 다국어 지원
- [ ] 실시간 콘텐츠 업데이트
- [ ] 사용자 맞춤 검색
- [ ] 학습 경로 추천

## 🔧 기술 스택

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Vector DB**: Pinecone / ChromaDB
- **Embeddings**: OpenAI Ada-002 / Sentence Transformers
- **LLM**: OpenAI GPT-4 / Anthropic Claude / Google Gemini

### Infrastructure
- **Container**: Docker
- **Orchestration**: Kubernetes (Production)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana

## 📊 성능 목표

| 메트릭 | 목표 | 현재 |
|--------|------|------|
| 검색 응답 시간 | < 200ms | - |
| 임베딩 처리량 | > 100 docs/min | - |
| RAG 응답 시간 | < 3s | - |
| 정확도 (Recall@10) | > 90% | - |

## 🔄 통합 포인트

### IWL Main Platform (iwl-v5-rebuild)
- API Gateway를 통한 검색 요청
- 사용자 컨텍스트 전달
- 학습 진도 동기화

### AI Engine Hub (ai-engine-hub)
- LLM 서비스 연동
- 프롬프트 최적화
- 응답 후처리

### AI Orchestra (ai-orchestra-v02)
- 자동 콘텐츠 생성 워크플로우
- 품질 검증 파이프라인
- 배포 자동화

## 📝 API 예시

### 검색 요청
```python
POST /api/search
{
    "query": "Python 비동기 프로그래밍",
    "filters": {
        "level": "intermediate",
        "language": "ko"
    },
    "top_k": 5
}
```

### RAG 질의
```python
POST /api/rag/query
{
    "question": "FastAPI에서 dependency injection은 어떻게 동작하나요?",
    "context": {
        "user_level": "advanced",
        "preferred_style": "practical"
    }
}
```

## 🚦 개발 현황

### ✅ 완료
- [x] 레포지토리 생성
- [x] 기본 구조 설계

### 🔄 진행 중
- [ ] Vector DB 선정
- [ ] 기본 API 구현

### ⏳ 계획
- [ ] RAG 파이프라인
- [ ] 성능 최적화
- [ ] 프로덕션 배포

## 🤝 기여 방법

1. Issue 생성 또는 선택
2. Feature 브랜치 생성
3. 구현 및 테스트
4. PR 생성
5. 코드 리뷰
6. 머지

## 📚 참고 문서

- [Master Build Tasks (Issue #34)](https://github.com/ihw33/iwl-v5-rebuild/issues/34)
- [IWL v5 Main Platform](https://github.com/ihw33/iwl-v5-rebuild)
- [AI Engine Hub](https://github.com/ihw33/ai-engine-hub)
- [AI Orchestra v02](https://github.com/ihw33/ai-orchestra-v02)

## 📞 연락처

- **Owner**: Thomas (ihw33)
- **Repository**: https://github.com/ihw33/iwl-kb-system
- **Issue Tracker**: https://github.com/ihw33/iwl-kb-system/issues

---

*Last Updated: 2025-08-27*
*Part of IWL v5.0 Ecosystem*