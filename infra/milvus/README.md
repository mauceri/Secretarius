# Milvus for Secretarius

This folder keeps Milvus orchestration in `~/Secretarius` while storing data outside the project.

## Why this layout
- Keep infra config versioned with Secretarius.
- Keep Milvus data persistent and isolated from source code.
- Avoid accidental commits of DB files.

## 1) Prepare environment file
```bash
cd /home/mauceric/Secretarius
cp infra/milvus/.env.example infra/milvus/.env
```

Edit `infra/milvus/.env` if needed, especially:
- `MILVUS_DATA_ROOT` (recommended external path, e.g. `/home/mauceric/milvus-data`)
- ports if conflicts exist.

## 2) Create external data directories
```bash
mkdir -p /home/mauceric/milvus-data/{etcd,minio,milvus}
```

## 3) Start Milvus stack
```bash
docker compose --env-file infra/milvus/.env -f infra/milvus/compose.yml up -d
```

## 4) Check health
```bash
curl -s http://localhost:9091/healthz
```

## 5) Stop stack
```bash
docker compose --env-file infra/milvus/.env -f infra/milvus/compose.yml down
```

## Optional migration from old volumes
If you want to reuse old data from `/home/mauceric/milvus/volumes`:
1. stop any old Milvus containers first.
2. copy the directories:
```bash
cp -a /home/mauceric/milvus/volumes/etcd /home/mauceric/milvus-data/
cp -a /home/mauceric/milvus/volumes/minio /home/mauceric/milvus-data/
cp -a /home/mauceric/milvus/volumes/milvus /home/mauceric/milvus-data/
```
3. verify ownership/permissions, then start with the new compose.
