# ClipCC

[English](README.md) | 繁體中文

跨平台、Docker 化的影片分類 API，使用 [OpenCLIP](https://github.com/mlfoundations/open_clip)（ViT-L-14）對影片內容進行文字標籤比對。上傳影片檔案並提供一組描述性標籤，即可取得每個標籤與影片內容的匹配信心分數。

**應用範例：** 上傳行車記錄器畫面，搭配標籤如 `"drunk driving"`、`"normal driving"`、`"distracted driving"`，API 會回傳哪個標籤最符合影片內容。

---

## 目錄

- [環境需求](#環境需求)
- [快速開始](#快速開始)
- [API 參考文件](#api-參考文件)
  - [POST /api/v1/classify](#post-apiv1classify)
  - [GET /live](#get-live)
  - [GET /ready](#get-ready)
- [組態設定](#組態設定)
- [身份驗證](#身份驗證)
- [GPU 支援](#gpu-支援)
- [自訂模型建置](#自訂模型建置)
- [執行測試](#執行測試)
- [專案結構](#專案結構)
- [運作原理](#運作原理)
- [常見問答](#常見問答)

---

## 環境需求

你只需要安裝 **Docker** 和 **Docker Compose**。其他所有依賴（Python、ffmpeg、PyTorch、CLIP 模型）都已封裝在 Docker 映像檔中。

| 需求 | 最低版本 | 檢查方式 |
|---|---|---|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+（V2） | `docker compose version` |

**安裝 Docker：**
- **macOS：** [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- **Windows：** [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- **Linux：** [Docker Engine](https://docs.docker.com/engine/install/) + [Docker Compose 外掛](https://docs.docker.com/compose/install/linux/)

---

## 快速開始

### 1. 複製儲存庫

```bash
git clone <your-repo-url>
cd clipCC
```

### 2. 建置 Docker 映像檔（CPU 版本）

```bash
docker compose --profile cpu build
```

首次建置需要 **5-15 分鐘**，會下載以下內容：
- Python 3.11 slim 基礎映像檔（約 50 MB）
- ffmpeg 及系統函式庫（約 100 MB）
- PyTorch CPU 版本（約 200 MB）
- OpenCLIP 及相關依賴（約 100 MB）
- ViT-L-14 模型權重（約 900 MB）— **內建於映像檔中，執行時不需要額外下載**

後續僅修改程式碼的建置會很快（Docker 快取層機制）。

### 3. 啟動伺服器

```bash
docker compose --profile cpu up
```

等待出現以下日誌訊息：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

模型會在啟動時載入記憶體，CPU 上大約需要 **10-30 秒**。

### 4. 測試連線

開啟新的終端機視窗：

```bash
# 檢查伺服器是否存活
curl http://localhost:8000/live
# {"status":"ok"}

# 檢查模型是否已載入
curl http://localhost:8000/ready
# {"status":"ready","model":"ViT-L-14","pretrained":"laion2b_s32b_b82k","device":"cpu"}
```

### 5. 分類影片

建立測試影片（或使用你自己的 .mp4 檔案）：

```bash
# 用 ffmpeg 產生一段 5 秒的測試影片
ffmpeg -y -f lavfi -i testsrc=duration=5:size=320x240:rate=10 \
  -c:v libx264 -pix_fmt yuv420p test_video.mp4
```

傳送至 API：

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["test pattern","outdoor scene","person walking"]' \
  -F "fps=1.0" \
  -F "aggregation=mean"
```

回應結果：

```json
{
  "best_match": {
    "label": "test pattern",
    "confidence": 1.0
  },
  "scores": [
    {
      "label": "test pattern",
      "confidence": 1.0,
      "raw_similarity": 0.3496
    },
    {
      "label": "outdoor scene",
      "confidence": 0.0,
      "raw_similarity": 0.1736
    },
    {
      "label": "person walking",
      "confidence": 0.0,
      "raw_similarity": 0.1737
    }
  ],
  "metadata": {
    "frames_analyzed": 5,
    "video_duration_seconds": 5.0,
    "model": "ViT-L-14",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 1.4,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Not suitable for safety-critical decisions."
  }
}
```

### 6. 停止伺服器

```bash
docker compose --profile cpu down
```

---

## API 參考文件

### POST /api/v1/classify

對影片進行文字標籤分類。

**Content-Type：** `multipart/form-data`

#### 請求欄位

| 欄位 | 類型 | 必填 | 預設值 | 說明 |
|---|---|---|---|---|
| `video` | 檔案 | 是 | - | 影片檔案。支援格式：`.mp4`、`.avi`、`.mov`、`.mkv` |
| `labels` | 字串（JSON） | 是 | - | 3-10 個文字標籤的 JSON 陣列。範例：`'["driving","parking","crash"]'` |
| `prompt_template` | 字串 | 否 | `"a video of {}"` | CLIP 文字提示模板。`{}` 會被替換為每個標籤。最長 500 字元。 |
| `fps` | 浮點數 | 否 | `1.0` | 影格取樣率。範圍：0.1-5.0。數值越高 = 取樣越多影格 = 處理較慢但可能更準確。 |
| `aggregation` | 字串 | 否 | `"mean"` | 分數聚合方式：`"mean"` 或 `"max"` |

#### 聚合方式

**`mean`**（預設）— 對所有取樣影格的信心分數取平均值。適合回答「這段影片主要呈現什麼內容？」信心分數加總為 1.0。

**`max`** — 針對每個標籤，獨立回傳所有影格中的最高信心分數。適合回答「影片中是否在某個時間點出現過這個內容？」分數不會加總為 1.0。每個分數會附帶 `peak_frame_index` 和 `approx_timestamp_seconds`，標示產生最高分的影格。

#### 提示模板

CLIP 會將影片影格與文字提示進行比對。預設情況下，每個標籤會被包裝成 `"a video of {label}"`。你可以自訂模板：

```bash
# 預設："a video of {label}"
-F 'labels=["driving","parking"]'

# 針對特定場景的自訂模板：
-F 'labels=["driving","parking"]'
-F 'prompt_template=a dashcam video showing {}'
# 產生："a dashcam video showing driving"、"a dashcam video showing parking"
```

更好的提示模板可以提升分類準確度。模板中必須恰好包含一個 `{}` 佔位符。

#### 回應 — Mean 模式

```json
{
  "best_match": {
    "label": "normal driving",
    "confidence": 0.45
  },
  "scores": [
    {
      "label": "drunk driving",
      "confidence": 0.15,
      "raw_similarity": 0.27
    },
    {
      "label": "normal driving",
      "confidence": 0.45,
      "raw_similarity": 0.31
    },
    {
      "label": "distracted driving",
      "confidence": 0.40,
      "raw_similarity": 0.30
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "ViT-L-14",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 12.3,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Not suitable for safety-critical decisions."
  }
}
```

#### 回應 — Max 模式

```json
{
  "best_match": {
    "label": "normal driving",
    "confidence": 0.72
  },
  "scores": [
    {
      "label": "drunk driving",
      "confidence": 0.38,
      "raw_similarity": 0.29,
      "peak_frame_index": 142,
      "approx_timestamp_seconds": 142.0
    },
    {
      "label": "normal driving",
      "confidence": 0.72,
      "raw_similarity": 0.34,
      "peak_frame_index": 85,
      "approx_timestamp_seconds": 85.0
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "ViT-L-14",
    "device": "cuda",
    "aggregation": "max",
    "processing_time_seconds": 2.1,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Max-mode scores are independent peaks per label and do not sum to 1. Not suitable for safety-critical decisions."
  }
}
```

#### 回應欄位說明

| 欄位 | 說明 |
|---|---|
| `best_match.label` | 信心分數最高的標籤 |
| `best_match.confidence` | 最佳匹配的信心值 |
| `scores[].label` | 原始標籤文字 |
| `scores[].confidence` | 相對信心分數（對 CLIP logit 縮放後的餘弦相似度取 softmax） |
| `scores[].raw_similarity` | 未縮放的餘弦相似度（影格嵌入與文字嵌入之間） |
| `scores[].peak_frame_index` | （僅 Max 模式）產生最高分的影格索引（從 0 開始） |
| `scores[].approx_timestamp_seconds` | （僅 Max 模式）最高分影格的近似時間戳記（`影格索引 / fps`） |
| `metadata.frames_analyzed` | 從影片中取樣的總影格數 |
| `metadata.video_duration_seconds` | 輸入影片的時長 |
| `metadata.model` | 使用的 OpenCLIP 模型架構 |
| `metadata.device` | 計算裝置（`cpu` 或 `cuda`） |
| `metadata.aggregation` | 使用的聚合方式 |
| `metadata.processing_time_seconds` | 推論管線的實際執行時間 |
| `metadata.disclaimer` | 提醒分數為相對值，非絕對值 |

#### 錯誤回應

所有錯誤都會回傳包含易讀 `detail` 訊息的 JSON。

| 狀態碼 | 條件 | 範例 |
|---|---|---|
| 401 | API 金鑰遺失或無效 | `"Invalid or missing API key..."` |
| 413 | 檔案超過大小限制 | `"File size 620.0 MB exceeds the maximum allowed size of 500.0 MB."` |
| 415 | 不支援的影片格式 | `"File format '.webm' is not supported."` |
| 422 | 影片過長 | `"Video duration 512.0s exceeds the maximum..."` |
| 422 | 影格數過多 | `"Extracting 1500 frames...exceeds the maximum of 300 frames."` |
| 422 | 解析度過高 | `"Video resolution 7680x4320 exceeds the maximum supported resolution of 3840x2160."` |
| 422 | 標籤驗證失敗 | `"Number of labels must be between 3 and 10 (inclusive)."` |
| 422 | 提示超過 CLIP token 限制 | `"Prompt '...' has 83 tokens and will be truncated..."` |
| 422 | 無效的 fps | `"FPS value 10.0 is invalid. Must be between 0.1 and 5.0."` |
| 429 | 同時上傳數過多 | `"Too many uploads in progress. Please retry in a moment."` |
| 429 | 同時推論數過多 | `"Server is processing the maximum number of videos..."` |
| 504 | 處理逾時 | `"Inference timed out after 300.0s."` |

### GET /live

存活探測端點。如果程序正在執行，回傳 `200`。不需要身份驗證，不受並行限制影響。

```json
{"status": "ok"}
```

### GET /ready

就緒探測端點。如果模型已載入且可以提供服務，回傳 `200` 及模型詳細資訊。模型仍在載入中時回傳 `503`。如有設定身份驗證，此端點會檢查驗證。

```json
{
  "status": "ready",
  "model": "ViT-L-14",
  "pretrained": "laion2b_s32b_b82k",
  "device": "cpu"
}
```

---

## 組態設定

所有設定透過環境變數控制。可在 `docker-compose.yml`、`.env` 檔案中設定，或透過 `docker run -e` 傳入。

複製範例檔案：
```bash
cp .env.example .env
# 依需求編輯 .env
```

| 變數 | 預設值 | 說明 |
|---|---|---|
| `MAX_FILE_SIZE_MB` | `500` | 上傳檔案大小上限（MB） |
| `MAX_DURATION_SECONDS` | `300` | 影片時長上限（5 分鐘） |
| `MAX_FRAMES` | `300` | 每次請求可擷取的最大影格數 |
| `DEFAULT_FPS` | `1.0` | 請求未指定時的預設影格取樣率 |
| `BATCH_SIZE` | `32` | 每批推論處理的影格數 |
| `MAX_CONCURRENT_REQUESTS` | `2`（CPU）/ `1`（GPU） | 最大同時推論請求數。GPU 預設為 1 以避免顯存不足。 |
| `MAX_UPLOAD_CONCURRENCY` | `MAX_CONCURRENT_REQUESTS + 2` | 最大同時上傳解析數。限制暫存磁碟使用量。 |
| `API_KEY` | （未設定） | 若有設定，所有對 `/api/v1/classify` 和 `/ready` 的請求都必須在標頭中包含 `X-API-Key` |
| `ALLOW_UNAUTHENTICATED` | `false` | 若未設定 `API_KEY`，此值必須為 `true`。伺服器在沒有明確驗證設定時會拒絕啟動。 |
| `FFMPEG_TIMEOUT_SECONDS` | `120` | ffmpeg/ffprobe 子程序的逾時時間 |
| `REQUEST_TIMEOUT_SECONDS` | `300` | 整個推論管線的端對端逾時時間 |
| `CLIP_CACHE_DIR` | `/app/models` | 模型權重儲存目錄。必須與建置時的快取路徑一致。 |

---

## 身份驗證

ClipCC 採用**預設拒絕（fail-closed）** 的身份驗證機制。伺服器必須符合以下其一才能啟動：

1. 設定 API 金鑰：
   ```yaml
   environment:
     - API_KEY=your-secret-key-here
   ```
   然後在每次請求中附上金鑰：
   ```bash
   curl -X POST http://localhost:8000/api/v1/classify \
     -H "X-API-Key: your-secret-key-here" \
     -F "video=@video.mp4" \
     -F 'labels=["a","b","c"]'
   ```

2. 明確選擇關閉驗證（僅供本機開發使用）：
   ```yaml
   environment:
     - ALLOW_UNAUTHENTICATED=true
   ```

`/live` 端點永遠不需要驗證（用於容器健康檢查）。`/ready` 端點會遵循驗證設定。

---

## GPU 支援

GPU 加速可大幅提升推論速度（比 CPU 快 10-30 倍）。

### 需求

- 支援 CUDA 的 NVIDIA GPU
- 已安裝 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker 已設定 `nvidia` 執行環境

### 建置並執行 GPU 版本

```bash
# 建置 CUDA 版本（映像檔較大，約 5 GB）
docker compose --profile gpu build

# 以 GPU 啟動
docker compose --profile gpu up
```

GPU 設定檔：
- 使用 `cu121`（CUDA 12.1）PyTorch 套件
- 預設 `MAX_CONCURRENT_REQUESTS=1` 以避免顯存過度使用
- 自動保留所有可用的 GPU

### 確認是否正在使用 GPU

```bash
curl http://localhost:8000/ready
# {"status":"ready","model":"ViT-L-14",...,"device":"cuda"}
```

如果 `device` 顯示 `"cuda"`，表示 GPU 加速已啟用。

---

## 自訂模型建置

預設映像檔內建 `ViT-L-14` 搭配 `laion2b_s32b_b82k` 權重。如需使用不同的 OpenCLIP 模型，請傳入建置參數：

```bash
# 範例：使用較小、較快的 ViT-B-32
docker build \
  --build-arg MODEL_NAME=ViT-B-32 \
  --build-arg PRETRAINED=laion2b_s34b_b79k \
  --build-arg TORCH_VARIANT=cpu \
  -t clipcc-custom .
```

可用模型清單請參閱 [OpenCLIP 模型列表](https://github.com/mlfoundations/open_clip#pretrained-model-interface)。

模型選擇在建置時決定，執行時無法變更。這是刻意設計：避免意外的執行時下載，並確保映像檔完全自包含。

---

## 執行測試

### 不使用 Docker（本機開發）

```bash
# 安裝依賴
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx anyio trio

# 執行單元測試（不需要 ffmpeg 或模型）
python -m pytest tests/test_config.py tests/test_temp_store.py tests/test_scoring.py tests/test_resource_gates.py -v

# 執行所有測試（需要 PATH 中有 ffmpeg，首次執行會下載約 400 MB 的模型）
python -m pytest tests/ -v
```

### 測試摘要

| 測試檔案 | 測試數 | 涵蓋範圍 |
|---|---|---|
| `test_config.py` | 6 | 設定驗證、身份驗證組態 |
| `test_temp_store.py` | 5 | 檔案上傳、清理、定時清除 |
| `test_video.py` | 9 | ffprobe 驗證、影格擷取 |
| `test_clip_model.py` | 7 | 模型載入、編碼、分詞 |
| `test_scoring.py` | 7 | Mean/Max 聚合、logit 縮放 |
| `test_resource_gates.py` | 10 | 並行限制器 |
| `test_middleware.py` | 12 | 身份驗證、上傳閘門、請求大小 |
| `test_inference_runner.py` | 3 | 逾時處理、取消機制 |
| `test_api.py` | 8 | 完整整合測試（健康檢查、驗證、分類） |
| **總計** | **67** | |

---

## 專案結構

```
clipCC/
├── Dockerfile                  # 多階段建置，內建模型權重
├── docker-compose.yml          # CPU 和 GPU 設定檔
├── requirements.txt            # Python 依賴
├── .env.example                # 組態參考
├── app/
│   ├── main.py                 # FastAPI 應用程式工廠、路由、管線協調
│   ├── config.py               # 透過環境變數的 Pydantic 設定
│   ├── middleware.py            # ASGI 中介軟體：身份驗證、上傳並行、請求大小
│   ├── resource_gates.py       # anyio CapacityLimiter（上傳與推論）
│   ├── temp_store.py           # 暫存檔案生命週期與清理
│   ├── inference_runner.py     # 執行緒化管線執行器，支援協作式逾時
│   ├── models/
│   │   ├── clip_model.py       # OpenCLIP 模型載入與推論
│   │   └── model_spec.py       # 從 .baked_model 檔案讀取模型元資料
│   ├── services/
│   │   ├── video.py            # ffprobe 驗證與 ffmpeg 影格擷取
│   │   └── scoring.py          # 影格分數的 Mean/Max 聚合
│   ├── schemas/
│   │   └── response.py         # Pydantic 回應模型
│   └── errors/
│       └── handlers.py         # 自訂 HTTP 例外與友善錯誤訊息
└── tests/                      # 9 個測試檔案，共 67 個測試
```

---

## 運作原理

### 處理管線

```
用戶端上傳影片 + 標籤
        |
        v
[ASGI 中介軟體] ── 身份驗證檢查（X-API-Key 標頭）
        |              上傳並行限制
        |              請求大小限制（串流式檢查）
        v
[路由處理器] ── 輸入驗證（格式、fps、標籤、token）
        |
        v
[推論閘門] ── 取得推論槽位（或回傳 429）
        |
        v
[InferenceRunner 執行緒]
        |
        ├── 儲存上傳檔案至暫存目錄
        ├── ffprobe：驗證時長、解析度、串流數
        ├── ffmpeg：以指定 fps 擷取影格（縮放至最大 512px）
        ├── 對每批影格：
        │     ├── 以 PIL 載入影像
        │     ├── 透過 OpenCLIP 編碼影像與文字
        │     ├── 計算餘弦相似度
        │     ├── 套用 logit 縮放 + softmax
        │     └── 刪除已處理的影格檔案
        ├── 聚合分數（mean 或 max）
        └── 回傳結果
        |
        v
[清理] ── 刪除所有暫存檔案（透過 finally 保證執行）
```

### 關鍵設計決策

- **在執行緒中執行同步管線：** 推論管線（ffmpeg + PyTorch）是阻塞式的。透過 `anyio.to_thread.run_sync` 在工作執行緒中執行，使得非同步事件迴圈保持回應能力，可以處理健康檢查和並行閘門判斷。

- **兩層並行控制：** 上傳並行和推論並行由兩個獨立的 `anyio.CapacityLimiter` 實例控制。上傳有限制但額度較高；推論槽位較為稀缺（尤其在 GPU 上）。

- **協作式逾時：** `InferenceRunner` 使用 `threading.Event` 旗標，在每批次之間進行檢查。逾時時會設定旗標、終止執行中的 ffmpeg 子程序，並等待工作執行緒結束後才釋放推論槽位。不會有被遺棄的執行緒。

- **模型內建於 Docker 映像檔：** 模型權重在 `docker build` 期間下載並存入映像檔。不需要執行時下載，不需要掛載外部磁碟區。模型設定的唯一真實來源是 `/app/.baked_model` 檔案。

- **Logit 縮放：** CLIP 的學習參數 `logit_scale`（約 100）會在 softmax 之前套用。如果不進行縮放，對原始餘弦相似度（0.2-0.35 範圍）直接取 softmax 會產生近乎均勻、毫無意義的分佈。

---

## 常見問答

### 一般問題

**問：支援哪些影片格式？**
答：`.mp4`、`.avi`、`.mov` 和 `.mkv`。影片必須只有一個視訊串流，解析度不超過 3840x2160，時長不超過 5 分鐘（可透過組態調整）。

**問：分類需要多長時間？**
答：取決於影片長度、fps 和硬體。5 分鐘影片以 1fps 取樣 = 300 影格。在 CPU 上預計 1-5 分鐘，在 GPU 上預計 10-30 秒。回應中的 `processing_time_seconds` 欄位會告訴你確切的處理時間。

**問：「信心分數（confidence）」是什麼意思？**
答：信心分數是**相對的**，不是絕對的。它們表示每個標籤與影片的匹配程度，是相對於你提供的其他標籤而言。同一段影片對不同的標籤組合會產生不同的信心值。這些數值不是校準過的機率，不應用於安全關鍵的決策。

**問：`raw_similarity` 是什麼意思？**
答：這是影片影格嵌入與文字標籤嵌入之間未經縮放的餘弦相似度，在模型的學習溫度縮放和 softmax 之前的原始值。適合需要自行進行分數處理的進階使用者。

**問：可以用於即時影片嗎？**
答：不行。ClipCC 設計用於離線影片分類，處理的是上傳的影片檔案，不支援即時串流。

### 安裝與 Docker

**問：建置時間很長，這正常嗎？**
答：是的，首次建置會下載約 1.3 GB 的依賴，包括 ViT-L-14 模型權重。後續建置會因為 Docker 快取層機制而大幅加速。只有 `app/` 目錄中的程式碼變更才會觸發最後幾層的重新建置。

**問：可以不用 Docker 執行嗎？**
答：可以，但強烈建議使用 Docker 以確保環境一致性。本機開發方式：
```bash
pip install -r requirements.txt
ALLOW_UNAUTHENTICATED=true uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```
你需要 Python 3.11 以上版本、PATH 中有 ffmpeg，而且模型會在首次啟動時下載。

**問：Docker 映像檔有多大？**
答：CPU 版本約 2 GB，GPU（CUDA）版本約 5 GB。大部分空間是 PyTorch 和模型權重。

**問：出現「Cannot connect to the Docker daemon」錯誤怎麼辦？**
答：啟動 Docker Desktop（macOS/Windows）或 Docker 服務（Linux：`sudo systemctl start docker`）。等幾秒鐘讓它完成初始化。

**問：建置時出現 `torchvision` 錯誤怎麼辦？**
答：確認 Dockerfile 中的 `torch torchvision` 是從同一個 PyTorch wheel 索引安裝。專案中包含的 Dockerfile 已正確處理這個問題。

### GPU 相關

**問：怎麼確認是否正在使用 GPU？**
答：檢查 `/ready` 端點。如果 `"device": "cuda"`，表示 GPU 已啟用。如果 `"device": "cpu"`，即使用 GPU 設定檔建置，也表示正在使用 CPU。請檢查 NVIDIA Container Toolkit 是否已安裝，以及 Docker 是否能偵測到你的 GPU（`docker run --gpus all nvidia-smi`）。

**問：可以使用 AMD GPU 嗎？**
答：目前不直接支援。CUDA 建置設定檔針對 NVIDIA GPU。AMD GPU 透過 ROCm 的支援需要不同的 PyTorch 建置和 Dockerfile 修改。

**問：為什麼 GPU 的 `MAX_CONCURRENT_REQUESTS` 設為 1？**
答：兩個同時進行的 ViT-L-14 推論呼叫，每個處理 32 影格批次，可能會耗盡 16 GB 以下 GPU 的顯存。如果你的 GPU 有較大的顯存（24 GB 以上），可以安全地將此值增加到 2。

### API 使用

**問：如何選擇好的標籤？**
答：標籤應該要有區別性、具描述性，並涵蓋你預期的內容範圍。越具體的標籤效果越好。例如，`"person running on a track"`（在跑道上跑步的人）比 `"movement"`（動作）效果更好。

**問：`fps` 最佳值是多少？**
答：`1.0`（預設值）適用於大多數情境。對於短影片需要更細緻的時間解析度，可增加到 2-5。對於內容大多靜態的長影片，可降低到 0.5。較高的 fps = 更多影格 = 處理較慢但可能更準確。

**問：什麼時候該用 `max` 而不是 `mean`？**
答：當你想知道影片的主要內容時，使用 `mean`（「這段影片主要在講什麼？」）。當你想偵測某個內容是否在任何時間點出現過，使用 `max`（「影片中有沒有出現過這個？」）。例如，在一段長時間的行車記錄中偵測短暫的交通違規，使用 `max` 效果更好。

**問：可以送超過 10 個標籤嗎？**
答：不行。每次請求的限制是 3-10 個標籤。如果需要更多分類，請分多次請求，每次使用不同的標籤組合。

**問：標籤出現「identical token sequences」錯誤是為什麼？**
答：CLIP 的分詞器可能對文字上看起來不同的標籤產生相同的 token 序列（例如額外的空格、Unicode 變體）。API 會拒絕這些標籤，因為模型確實無法區分它們。請使用更明確不同的措辭。

**問：提示模板有什麼用？**
答：CLIP 是在圖像-文字配對上訓練的。將標籤包裝在自然語句中（如 "a video of {label}"）通常比直接使用裸標籤能獲得更好的準確度。預設的 `"a video of {}"` 適用於一般場景。對於特定領域的內容，可以嘗試如 `"a surveillance camera recording of {}"` 或 `"a dashcam video showing {}"` 等模板。

### 效能與限制

**問：收到 429「Too many uploads」該怎麼辦？**
答：伺服器限制了同時上傳數以防止磁碟空間耗盡。請稍候片刻後重試。如果經常發生，可增加 `MAX_UPLOAD_CONCURRENCY` 的值。

**問：收到 504 逾時該怎麼辦？**
答：推論管線超過了 `REQUEST_TIMEOUT_SECONDS`（預設 300 秒）的限制。解決方案：使用較短的影片、降低 `fps`，或增加逾時時間。在 CPU 上，300 影格可能需要數分鐘。

**問：影片檔案大小上限是多少？**
答：預設為 500 MB（可透過 `MAX_FILE_SIZE_MB` 調整）。此限制在 ASGI 中介軟體層就會執行，在檔案完全解析之前就會拒絕過大的上傳。

**問：多個使用者可以同時呼叫 API 嗎？**
答：可以，但有限制。上傳並行數預設為 `MAX_CONCURRENT_REQUESTS + 2`（允許少量排隊）。推論並行數在 CPU 上預設為 2，GPU 上預設為 1。超過限制的請求會收到 `429` 回應。如需更高的吞吐量，請在負載平衡器後方部署多個實例。

### 安全性

**問：直接部署到公網上安全嗎？**
答：對於正式環境部署，你應該：（1）設定 `API_KEY` 而非 `ALLOW_UNAUTHENTICATED`；（2）部署在反向代理（nginx、Caddy）後方，並啟用速率限制和 TLS；（3）限制網路存取。內建的並行限制提供基本的 DoS 防護，但不能替代完善的基礎設施。

**問：API 會儲存我的影片嗎？**
答：不會。上傳的影片會暫存到臨時目錄，處理完成後立即刪除。啟動時的清除程式也會清理先前異常終止所遺留的暫存檔案。請求完成後不會保留任何影片資料。
