# Runtime Bootstrap (FastAPI + Redis + Postgres + MinIO + SGLang)

## 1) Keep single project env

```powershell
conda activate LabSOPGuard
```

## 2) Install locked runtime deps

```powershell
pip install -r requirements-lock.txt
```

## 3) Start service stack (docker)

```powershell
docker compose -f docker-compose.runtime.yml up -d
```

## 4) API health check

```powershell
curl http://localhost:8080/health
```

## 5) Run one VLA runtime request

```powershell
curl -X POST http://localhost:8080/v1/runtime/run `
  -H "Content-Type: application/json" `
  -d "{\"sample_id\":\"demo_api_001\",\"instruction\":\"pick sample container and place carefully\",\"rgb_path\":\"D:/LabEmbodiedVLA/LabSOPGuard/data/interim/labeling/frames/first_person_复杂长操作_normal_correct_001/first_person_复杂长操作_normal_correct_001__f000000.jpg\",\"robot_config\":\"configs/robot/bridge.yaml\"}"
```

## 6) Local non-service runtime (fallback)

```powershell
python .\lab_titration_vla\deploy\runtime_pipeline\run_runtime_pipeline.py `
  --sample-id runtime_demo `
  --instruction "pick sample container and place carefully" `
  --rgb-path "D:\LabEmbodiedVLA\LabSOPGuard\data\interim\labeling\frames\first_person_复杂长操作_normal_correct_001\first_person_复杂长操作_normal_correct_001__f000000.jpg"
```

## Notes

- Current worker is a scaffold task queue endpoint; next step is to offload heavy VLM calls.
- SGLang service is included as runtime placeholder and should be pinned to your chosen model image in production.
- Keep all large video/raw outputs outside Git and use MinIO or local mounted storage.

