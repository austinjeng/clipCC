# ClipCC

[English](README.md) | 繁體中文

跨平台、Docker 化的影片分類 API，使用 [SigLIP2](https://huggingface.co/docs/transformers/model_doc/siglip2) 模型對影片內容進行文字標籤比對。上傳影片檔案並提供一組描述性標籤，即可取得每個標籤與影片內容的匹配信心分數。

**應用範例：** 上傳行車記錄器畫面，搭配標籤如 `"drunk driving"`、`"normal driving"`、`"distracted driving"`，API 會回傳哪個標籤最符合影片內容。

---

## 功能特色

- **6 個 SigLIP2 模型** — 從 0.4B 到 1B 參數，256px 到 512px 解析度。可透過網頁 UI 或 API 熱切換模型，無需重啟。
- **內建網頁 UI** — 開啟 `http://localhost:8000/` 即可使用模型選擇器、影片上傳、標籤輸入和結果視覺化。
- **三種聚合模式** — `mean`（跨影格平均）、`max`（每標籤峰值含時間戳記）、`temporal`（逐影格時間軸搭配片段偵測）。
- **Sigmoid 評分** — 每個標籤獨立取得 0 到 1 之間的信心分數。多個標籤可同時獲得高分。
- **輕量化 Docker 映像** — 基礎映像約 1 GB。模型按需下載並快取於 Docker 磁碟區。
- **GPU 支援** — NVIDIA CUDA 加速（比 CPU 快 10-30 倍）。
- **預設拒絕的身份驗證** — 必須設定 API 金鑰或明確停用驗證。

---

## 目錄

- [功能特色](#功能特色)
- [可用模型](#可用模型)
- [快速開始](#快速開始)
  - [Docker 映像（推薦）](#docker-映像推薦)
  - [從原始碼建置](#從原始碼建置)
- [網頁 UI](#網頁-ui)
- [API 參考文件](#api-參考文件)
  - [POST /api/v1/classify](#post-apiv1classify)
  - [GET /api/v1/models](#get-apiv1models)
  - [POST /api/v1/models/load](#post-apiv1modelsload)
  - [GET /api/v1/models/active](#get-apiv1modelsactive)
  - [GET /live](#get-live)
  - [GET /ready](#get-ready)
- [組態設定](#組態設定)
- [身份驗證](#身份驗證)
- [GPU 支援](#gpu-支援)
- [執行測試](#執行測試)
- [專案結構](#專案結構)
- [運作原理](#運作原理)
- [常見問答](#常見問答)

---

## 可用模型

ClipCC 內建 6 個 SigLIP2 模型。預設模型會在啟動時自動載入，可透過 `DEFAULT_MODEL_ID` 環境變數變更。

| 模型 | 參數量 | 解析度 | 適用場景 |
|---|---|---|---|
| `siglip2-base-patch16-256` | 0.4B | 256px | 快速推論、低記憶體需求（CPU 預設） |
| `siglip2-base-patch16-384` | 0.4B | 384px | 更好的準確度，仍然輕量 |
| `siglip2-large-patch16-256` | 0.9B | 256px | 更高品質，中等速度 |
| `siglip2-large-patch16-384` | 0.9B | 384px | 高品質，中等記憶體 |
| `siglip2-so400m-patch14-384` | 1B | 384px | 最佳品質（GPU 預設） |
| `siglip2-so400m-patch16-512` | 1B | 512px | 最高解析度，最多記憶體 |

**切換模型：**
- **網頁 UI：** 從下拉選單選擇後點擊 **Load Model**
- **API：** `POST /api/v1/models/load`，傳入 `{"model_id": "siglip2-large-patch16-384"}`
- **啟動預設：** 設定 `DEFAULT_MODEL_ID` 環境變數

模型首次使用時從 HuggingFace 下載（基礎模型約 800 MB，大型/SO400M 模型約 2 GB），之後快取供未來使用。

---

## 快速開始

### Docker 映像（推薦）

拉取預建映像，無需複製或建置：

```bash
docker pull ghcr.io/austinjeng/clipcc:latest
docker run -p 8000:8000 -e ALLOW_UNAUTHENTICATED=true \
  -v clipcc-models:/app/models ghcr.io/austinjeng/clipcc:latest
```

開啟 `http://localhost:8000`。預設模型（約 800 MB）會在首次啟動時自動下載。

**正式部署**（使用 `--env-file` 避免密鑰留在 shell 歷史紀錄中）：

```bash
# 建立 .env 檔案
echo "API_KEY=your-secret-key" > .env

# 執行
docker run -p 8000:8000 --env-file .env \
  -v clipcc-models:/app/models ghcr.io/austinjeng/clipcc:latest
```

### 從原始碼建置

#### 1. 複製儲存庫

```bash
git clone https://github.com/austinjeng/clipCC.git
cd clipCC
```

### 2. 建置 Docker 映像檔（CPU 版本）

```bash
docker compose --profile cpu build
```

首次建置需要 **5-10 分鐘**，會下載 Python、ffmpeg、PyTorch 及相關函式庫。模型權重會在首次啟動時另外下載，不包含在建置中。

### 3. 啟動伺服器

```bash
docker compose --profile cpu up
```

首次啟動時，預設模型（`siglip2-base-patch16-256`，約 800 MB）會自動下載。此過程只需執行一次，Docker 磁碟區會快取模型以供後續使用。

等待出現以下日誌訊息：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Auto-loaded model: siglip2-base-patch16-256
```

### 4. 測試連線

開啟新的終端機視窗：

```bash
# 檢查伺服器是否存活
curl http://localhost:8000/live
# {"status":"ok"}

# 檢查模型是否已載入
curl http://localhost:8000/ready
# {"status":"ready","model":"siglip2-base-patch16-256","pretrained":"siglip2-base-patch16-256","device":"cpu"}
```

在瀏覽器中開啟 **http://localhost:8000/** 即可存取網頁 UI。

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
    "confidence": 0.92
  },
  "scores": [
    {
      "label": "test pattern",
      "confidence": 0.92,
      "raw_similarity": 0.35
    },
    {
      "label": "outdoor scene",
      "confidence": 0.12,
      "raw_similarity": 0.17
    },
    {
      "label": "person walking",
      "confidence": 0.08,
      "raw_similarity": 0.15
    }
  ],
  "metadata": {
    "frames_analyzed": 5,
    "video_duration_seconds": 5.0,
    "model": "siglip2-base-patch16-256",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 1.4,
    "disclaimer": "Scores are relative to the supplied labels, ...",
    "model_type": "siglip2",
    "score_semantics": "siglip2_pairwise_sigmoid"
  }
}
```

> **注意：** SigLIP2 使用 sigmoid 評分 — 每個標籤獨立取得 0 到 1 之間的信心分數。分數**不會**加總為 1.0。`"test pattern"` 標籤應該會得到最高分，因為測試影片就是 ffmpeg 的測試圖樣。

### 6. 停止伺服器

```bash
docker compose --profile cpu down
```

模型權重保存在 Docker 磁碟區中，下次啟動不需要重新下載。

> 更詳細的安裝指南（包括 Windows 原生、macOS 原生、Linux 原生等方式），請參閱 [English README](README.md)。

---

## 網頁 UI

ClipCC 包含內建網頁介面，位於 **http://localhost:8000/**。網頁 UI 為選用功能，所有功能均可透過 API 使用。

### 介面功能

- **模型選擇器** — 下拉選單列出所有 6 個 SigLIP2 模型，搭配狀態指示燈（綠色 = 已載入就緒）
- **影片上傳** — 拖放或選擇 `.mp4`、`.avi`、`.mov`、`.mkv` 檔案
- **標籤輸入** — 以逗號分隔輸入標籤（3-10 個）
- **聚合模式** — 選擇 mean、max 或 temporal
- **時序控制** — 選擇 temporal 模式時，會出現閾值、間隔容忍度和最短時長的滑桿

### 使用流程

1. 預設模型會在啟動時自動載入。等待綠色狀態燈亮起。
2. 上傳影片檔案。
3. 輸入以逗號分隔的標籤，例如：`driving, parking, reversing`
4. 選擇聚合模式。
5. 點擊 **Classify**。
6. 結果以水平信心長條圖顯示。在 temporal 模式中，還會顯示時間軸圖表和片段表格。

### 切換模型

1. 從下拉選單中選擇不同的模型。
2. 點擊 **Load Model**。
3. 等待載入完成（若未快取則先下載模型，再載入）。
4. 狀態燈變為綠色時表示就緒。
5. 再次分類 — 新模型已啟用。

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
| `prompt_template` | 字串 | 否 | `"This is a photo of {}."` | 文字提示模板。`{}` 會被替換為每個標籤。最長 500 字元。 |
| `fps` | 浮點數 | 否 | `1.0` | 影格取樣率。範圍：0.1-5.0。數值越高 = 取樣越多影格 = 處理較慢但可能更準確。 |
| `aggregation` | 字串 | 否 | `"mean"` | 分數聚合方式：`"mean"`、`"max"` 或 `"temporal"` |
| `threshold` | 浮點數 | 否 | 模型預設值 | 時序片段偵測的信心閾值。範圍：0.0-1.0。僅在 `aggregation=temporal` 時有效。 |
| `gap_tolerance` | 浮點數 | 否 | `2.0` | 合併為同一片段的最大間隔秒數。範圍：0.0-10.0。僅在 `aggregation=temporal` 時有效。 |
| `min_duration` | 浮點數 | 否 | `1.0` | 納入結果的最短片段時長（秒）。範圍：0.0-10.0。僅在 `aggregation=temporal` 時有效。 |

#### 聚合方式

**`mean`**（預設）— 對所有取樣影格的信心分數取平均值。適合回答「這段影片主要呈現什麼內容？」

**`max`** — 針對每個標籤，獨立回傳所有影格中的最高信心分數。適合回答「影片中是否在某個時間點出現過這個內容？」每個分數會附帶 `peak_frame_index` 和 `approx_timestamp_seconds`，標示產生最高分的影格。

**`temporal`** — 回傳逐影格的時間軸及每個標籤的信心分數，加上超過閾值的偵測片段。適合回答「每個事件何時發生、持續多久？」支援可設定的閾值、間隔容忍度（合併鄰近片段）和最短時長（過濾雜訊）。回傳片段統計資料，包括有效平均值、覆蓋率和時長加權信心分數。

#### 提示模板

模型會將影片影格與文字提示進行比對。預設情況下，每個標籤會被包裝成 `"This is a photo of {label}."`。你可以自訂模板：

```bash
# 預設："This is a photo of {label}."
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
    "confidence": 0.72
  },
  "scores": [
    {
      "label": "drunk driving",
      "confidence": 0.15,
      "raw_similarity": 0.27
    },
    {
      "label": "normal driving",
      "confidence": 0.72,
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
    "model": "siglip2-base-patch16-256",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 12.3,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Not suitable for safety-critical decisions.",
    "model_type": "siglip2",
    "score_semantics": "siglip2_pairwise_sigmoid"
  }
}
```

#### 回應 — Max 模式

```json
{
  "best_match": {
    "label": "normal driving",
    "confidence": 0.85
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
      "confidence": 0.85,
      "raw_similarity": 0.34,
      "peak_frame_index": 85,
      "approx_timestamp_seconds": 85.0
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "siglip2-base-patch16-256",
    "device": "cuda",
    "aggregation": "max",
    "processing_time_seconds": 2.1,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Max-mode scores are independent peaks per label and do not sum to 1. Not suitable for safety-critical decisions.",
    "model_type": "siglip2",
    "score_semantics": "siglip2_pairwise_sigmoid"
  }
}
```

#### 回應 — Temporal 模式

```json
{
  "best_match": {
    "label": "normal driving",
    "confidence": 0.85
  },
  "scores": [
    {
      "label": "drunk driving",
      "confidence": 0.38,
      "raw_similarity": 0.29
    },
    {
      "label": "normal driving",
      "confidence": 0.85,
      "raw_similarity": 0.34
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "siglip2-base-patch16-256",
    "device": "cpu",
    "aggregation": "temporal",
    "processing_time_seconds": 14.2,
    "disclaimer": "...",
    "model_type": "siglip2",
    "score_semantics": "siglip2_pairwise_sigmoid"
  },
  "temporal": {
    "timeline": [
      {"timestamp": 0.0, "frame_index": 0, "scores": {"drunk driving": 0.12, "normal driving": 0.78}},
      {"timestamp": 1.0, "frame_index": 1, "scores": {"drunk driving": 0.10, "normal driving": 0.81}}
    ],
    "segments": [
      {
        "label": "normal driving",
        "start_time": 0.0,
        "end_time": 120.5,
        "duration": 120.5,
        "stats": {
          "active_avg": 0.76,
          "interval_avg": 0.74,
          "coverage_ratio": 0.95,
          "active_duration": 114.5
        },
        "peak_confidence": 0.85,
        "peak_timestamp": 42.0
      }
    ],
    "label_summaries": [
      {
        "label": "normal driving",
        "segment_count": 2,
        "total_active_duration": 180.0,
        "total_segment_duration": 200.0,
        "peak_confidence": 0.85,
        "duration_weighted_confidence": 0.74
      }
    ],
    "best_segment": {
      "label": "normal driving",
      "start_time": 0.0,
      "end_time": 120.5,
      "duration": 120.5,
      "stats": {"active_avg": 0.76, "interval_avg": 0.74, "coverage_ratio": 0.95, "active_duration": 114.5},
      "peak_confidence": 0.85,
      "peak_timestamp": 42.0
    },
    "threshold_mode": "absolute",
    "effective_threshold": 0.5,
    "threshold_was_defaulted": true
  }
}
```

#### 回應欄位說明

| 欄位 | 說明 |
|---|---|
| `best_match.label` | 信心分數最高的標籤 |
| `best_match.confidence` | 最佳匹配的信心值 |
| `scores[].label` | 原始標籤文字 |
| `scores[].confidence` | 信心分數。SigLIP2：獨立 sigmoid（0-1），分數不會加總為 1。詳見 `score_semantics`。 |
| `scores[].raw_similarity` | 未縮放的餘弦相似度（影格嵌入與文字嵌入之間） |
| `scores[].peak_frame_index` | （僅 Max 模式）產生最高分的影格索引（從 0 開始） |
| `scores[].approx_timestamp_seconds` | （僅 Max 模式）最高分影格的近似時間戳記（`影格索引 / fps`） |
| `metadata.model` | 用於分類的模型 ID |
| `metadata.device` | 計算裝置（`cpu` 或 `cuda`） |
| `metadata.aggregation` | 使用的聚合方式 |
| `metadata.model_type` | 模型後端類型（`siglip2`） |
| `metadata.score_semantics` | 評分方法識別碼（`siglip2_pairwise_sigmoid`） |
| `metadata.processing_time_seconds` | 推論管線的實際執行時間 |
| `metadata.disclaimer` | 提醒分數為相對值，非絕對值 |
| `temporal.timeline` | （僅 Temporal 模式）每個標籤的逐影格分數 |
| `temporal.segments` | （僅 Temporal 模式）標籤超過閾值的偵測時間片段 |
| `temporal.segments[].stats` | 片段統計：`active_avg`、`interval_avg`、`coverage_ratio`、`active_duration` |
| `temporal.label_summaries` | （僅 Temporal 模式）每個標籤的跨片段彙總統計 |
| `temporal.best_segment` | （僅 Temporal 模式）峰值信心最高的片段，或 `null` |
| `temporal.threshold_mode` | （僅 Temporal 模式）`"absolute"`（SigLIP2） |
| `temporal.effective_threshold` | （僅 Temporal 模式）實際使用的閾值（明確指定或模型預設） |
| `temporal.threshold_was_defaulted` | （僅 Temporal 模式）`true` 表示使用了模型的預設閾值 |

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
| 422 | 提示超過模型 token 限制 | `"Prompt '...' has 83 tokens and will be truncated..."` |
| 422 | 無效的 fps | `"FPS value 10.0 is invalid. Must be between 0.1 and 5.0."` |
| 422 | 非 temporal 模式使用時序參數 | 時序參數僅在 `aggregation=temporal` 時有效 |
| 429 | 同時上傳數過多 | `"Too many uploads in progress. Please retry in a moment."` |
| 429 | 同時推論數過多 | `"Too many inference requests in progress. Please retry in a moment."` |
| 503 | 模型未載入 | `"Model not loaded"` |
| 504 | 處理逾時 | `"Inference timed out after 300.0s."` |

---

### GET /api/v1/models

列出所有可用模型及其快取與載入狀態。

```bash
curl http://localhost:8000/api/v1/models
```

```json
[
  {
    "model_id": "siglip2-base-patch16-256",
    "display_name": "SigLIP2 Base (256px)",
    "model_type": "siglip2",
    "params": "0.4B",
    "resolution": 256,
    "loaded": true,
    "cached": true
  },
  {
    "model_id": "siglip2-large-patch16-384",
    "display_name": "SigLIP2 Large (384px)",
    "model_type": "siglip2",
    "params": "0.9B",
    "resolution": 384,
    "loaded": false,
    "cached": false
  }
]
```

| 欄位 | 說明 |
|---|---|
| `loaded` | `true` 表示此模型目前為啟用中並正在處理請求 |
| `cached` | `true` 表示模型權重已下載至本機 |

---

### POST /api/v1/models/load

依模型 ID 載入模型。若未快取則先下載。若有其他模型正在使用中，會等待進行中的請求完成後再替換。

**Content-Type：** `application/json`

```bash
curl -X POST http://localhost:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "siglip2-large-patch16-384"}'
```

**成功回應：**
```json
{"status": "loaded", "model_id": "siglip2-large-patch16-384"}
```

**錯誤回應：**

| 狀態碼 | 條件 |
|---|---|
| 400 | 未知的 `model_id` |
| 500 | 模型載入失敗 |

---

### GET /api/v1/models/active

取得目前載入模型的元資料，包括時序模式預設值。

```bash
curl http://localhost:8000/api/v1/models/active
```

```json
{
  "model_id": "siglip2-base-patch16-256",
  "display_name": "SigLIP2 Base (256px)",
  "model_type": "siglip2",
  "params": "0.4B",
  "resolution": 256,
  "device": "cpu",
  "temporal_defaults": {
    "threshold": 0.5,
    "threshold_mode": "absolute",
    "gap_tolerance": 2.0,
    "min_duration": 1.0
  }
}
```

若無模型載入則回傳 `404`。

---

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
  "model": "siglip2-base-patch16-256",
  "pretrained": "siglip2-base-patch16-256",
  "device": "cpu"
}
```

---

## 組態設定

所有設定透過環境變數控制。可在 `docker-compose.yml`、`.env` 檔案中設定，或啟動時以行內方式傳入。

複製範例檔案：
```bash
cp .env.example .env
# 依需求編輯 .env
```

> **原生安裝使用者注意：** `.env.example` 預設使用 Docker 路徑（如 `CLIP_CACHE_DIR=/app/models`）。若以原生方式執行，**必須**改為本機路徑。請參閱 `.env.example` 中的註釋。

| 變數 | 預設值 | 說明 |
|---|---|---|
| `DEFAULT_MODEL_ID` | `siglip2-base-patch16-256` | 啟動時自動載入的模型。若有 GPU 或更多記憶體，可設為較大的模型（例如 `siglip2-so400m-patch14-384`）。 |
| `MAX_FILE_SIZE_MB` | `500` | 上傳檔案大小上限（MB） |
| `MAX_DURATION_SECONDS` | `300` | 影片時長上限（5 分鐘） |
| `MAX_FRAMES` | `300` | 每次請求可擷取的最大影格數 |
| `DEFAULT_FPS` | `1.0` | 請求未指定時的預設影格取樣率 |
| `BATCH_SIZE` | `32` | 每批推論處理的影格數 |
| `MAX_CONCURRENT_REQUESTS` | `2`（CPU）/ `1`（GPU） | 最大同時推論請求數。GPU 預設為 1 以避免顯存不足。 |
| `MAX_UPLOAD_CONCURRENCY` | `MAX_CONCURRENT_REQUESTS + 2` | 最大同時上傳解析數。限制暫存磁碟使用量。 |
| `API_KEY` | （未設定） | 若有設定，所有對 `/api/v1/classify` 和 `/ready` 的請求都必須在標頭中包含 `X-API-Key` |
| `ALLOW_UNAUTHENTICATED` | `false` | 若未設定 `API_KEY`，此值必須為 `true`。伺服器在沒有明確驗證設定時會拒絕啟動。 |
| `SKIP_MODEL_AUTOLOAD` | `false` | 啟動時跳過自動載入模型。伺服器啟動後 `/live` 正常運作，但 `/ready` 會回傳 503 直到透過 `/api/v1/models/load` 手動載入模型。適用於 CI 煙霧測試或延遲載入部署。 |
| `FFMPEG_TIMEOUT_SECONDS` | `120` | ffmpeg/ffprobe 子程序的逾時時間 |
| `REQUEST_TIMEOUT_SECONDS` | `300` | 整個推論管線的端對端逾時時間 |
| `CLIP_CACHE_DIR` | `/app/models` | 模型權重下載與快取目錄。 |
| `TEMP_DIR` | `/tmp/clipcc` | 暫存上傳和影格檔案的目錄。**Windows 原生使用者：** 須覆寫為 Windows 路徑如 `C:\temp\clipcc`。 |

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

- 支援 CUDA 的 **NVIDIA GPU**（建議 RTX 20xx 系列以上）
- 主機已安裝 **NVIDIA 驅動程式**
- 已安裝 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 建置並執行 GPU 版本

```bash
# 建置 CUDA 版本（映像檔較大，約 5 GB）
docker compose --profile gpu build

# 以 GPU 啟動
docker compose --profile gpu up
```

GPU 設定檔預設使用 `siglip2-so400m-patch14-384`（1B 參數）— 可在大多數 GPU 上舒適運行的最高品質模型。你可以隨時透過網頁 UI 或 API 切換到其他模型。

### 確認是否正在使用 GPU

```bash
curl http://localhost:8000/ready
# {"status":"ready","model":"siglip2-so400m-patch14-384",...,"device":"cuda"}
```

如果 `device` 顯示 `"cuda"`，表示 GPU 加速已啟用。如果顯示 `"cpu"`，請檢查 NVIDIA Container Toolkit 是否已安裝，以及 Docker 是否能偵測到 GPU：`docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`。

---

## 執行測試

```bash
# 安裝依賴
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx anyio trio

# 執行單元測試（不需要 ffmpeg 或模型）
python -m pytest tests/test_config.py tests/test_temp_store.py tests/test_scoring.py \
  tests/test_resource_gates.py tests/test_frame_timeline.py tests/test_temporal_policy.py -v

# 執行所有測試（需要 PATH 中有 ffmpeg，首次執行會下載預設模型約 800 MB）
python -m pytest tests/ -v
```

### 測試摘要

| 測試檔案 | 測試數 | 涵蓋範圍 |
|---|---|---|
| `test_api.py` | 28 | 完整整合測試（健康檢查、驗證、分類、對比） |
| `test_base_model.py` | 2 | BaseModel 抽象類別契約 |
| `test_config.py` | 14 | 設定驗證、身份驗證組態 |
| `test_download_script.py` | 7 | 模型下載腳本 |
| `test_frame_timeline.py` | 9 | 影格間隔、時間戳記、間隔/時長計算 |
| `test_inference_runner.py` | 5 | 逾時處理、取消機制、工作執行緒收尾 |
| `test_integration.py` | 1 | 端對端整合流程 |
| `test_middleware.py` | 12 | 身份驗證、上傳閘門、請求大小 |
| `test_model_manager.py` | 40 | 模型註冊、熱切換、租約並行 |
| `test_resource_gates.py` | 10 | 並行限制器 |
| `test_scoring.py` | 36 | Mean/Max/Temporal/對比聚合、評分上下文 |
| `test_siglip2_model.py` | 10 | SigLIP2 模型載入、sigmoid 評分 |
| `test_temp_store.py` | 5 | 檔案上傳、清理、定時清除 |
| `test_temporal_policy.py` | 6 | 評分策略、閾值模式 |
| `test_video.py` | 10 | ffprobe 驗證、影格擷取 |
| **總計** | **195** | |

---

## 專案結構

```
clipCC/
├── Dockerfile                  # 單階段建置（約 1 GB，不內建模型權重）
├── docker-compose.yml          # CPU 和 GPU 設定檔，含模型磁碟區
├── requirements.txt            # Python 依賴
├── .env.example                # 組態參考
├── pytest.ini                  # Pytest 設定
├── app/
│   ├── main.py                 # FastAPI 應用程式工廠、路由、管線協調
│   ├── config.py               # 透過環境變數的 Pydantic 設定
│   ├── middleware.py            # ASGI 中介軟體：身份驗證、上傳並行、請求大小
│   ├── resource_gates.py       # anyio CapacityLimiter（上傳與推論）
│   ├── temp_store.py           # 暫存檔案生命週期與清理
│   ├── inference_runner.py     # 執行緒化管線執行器，支援協作式逾時
│   ├── models/
│   │   ├── base_model.py       # 所有模型後端的抽象基礎類別
│   │   ├── siglip2_model.py    # SigLIP2 模型，使用 HuggingFace transformers（sigmoid 評分）
│   │   └── model_manager.py    # 模型註冊表、熱切換與租約式並行控制
│   ├── services/
│   │   ├── video.py            # ffprobe 驗證與 ffmpeg 影格擷取
│   │   ├── scoring.py          # Mean/Max/Temporal/對比聚合
│   │   ├── frame_timeline.py   # 時序模式的影格間隔計算
│   │   └── temporal_policy.py  # SigLIP2 評分策略（閾值行為）
│   ├── schemas/
│   │   └── response.py         # Pydantic 回應模型
│   ├── errors/
│   │   └── handlers.py         # 自訂 HTTP 例外
│   └── static/
│       ├── index.html          # 網頁 UI（模型選擇器、時序控制、Chart.js 視覺化）
│       └── vendor/
│           └── chart.min.js    # Chart.js 4.4.9，用於時序時間軸渲染
└── tests/                      # 15 個測試檔案，共 130 個測試
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
[ModelManager] ── 取得模型租約（或回傳 503）
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
        │     ├── model.score_batch() → ScoreBatch（信心、相似度、logits）
        │     └── 刪除已處理的影格檔案
        ├── 聚合分數：
        │     ├── mean：跨影格平均信心
        │     ├── max：每標籤峰值信心含時間戳記
        │     └── temporal：影格時間軸 → 閾值偵測
        │           → 片段合併 → 標籤彙總
        └── 回傳 ClassifyResponse
        |
        v
[清理] ── 刪除所有暫存檔案（透過 finally 保證執行）
```

### 關鍵設計決策

- **在執行緒中執行同步管線：** 推論管線（ffmpeg + PyTorch）是阻塞式的。透過 `anyio.to_thread.run_sync` 在工作執行緒中執行，使得非同步事件迴圈保持回應能力。

- **兩層並行控制：** 上傳並行和推論並行由兩個獨立的 `anyio.CapacityLimiter` 實例控制。上傳有限制但額度較高；推論槽位較為稀缺（尤其在 GPU 上）。

- **協作式逾時：** `InferenceRunner` 使用 `threading.Event` 旗標，在每批次之間進行檢查。逾時時會設定旗標、終止執行中的 ffmpeg 子程序，並等待工作執行緒結束後才釋放推論槽位。

- **隨需載入模型：** 模型首次使用時從 HuggingFace 下載，並快取於本機（Docker 磁碟區或本機目錄）。`ModelManager` 透過租約式並行控制協調熱切換 — 進行中的請求會在當前模型上完成，再載入新模型。

- **評分語意：** SigLIP2 模型使用成對 sigmoid 評分（每標籤獨立，分數介於 0-1，且不會加總為 1）。回應中的 `score_semantics` 欄位標示使用的方法，時序聚合管線會選擇對應的閾值策略。

---

## 常見問答

### 一般問題

**問：支援哪些影片格式？**
答：`.mp4`、`.avi`、`.mov` 和 `.mkv`。影片必須只有一個視訊串流，解析度不超過 3840x2160，時長不超過 5 分鐘（可透過組態調整）。

**問：分類需要多長時間？**
答：取決於影片長度、fps 和硬體。5 分鐘影片以 1fps 取樣 = 300 影格。在 CPU 上預計 1-5 分鐘，在 GPU 上預計 10-30 秒。回應中的 `processing_time_seconds` 欄位會告訴你確切的處理時間。

**問：「信心分數（confidence）」是什麼意思？**
答：信心分數是 **sigmoid** 分數，介於 0 到 1 之間 — 0.8 表示模型有 80% 的信心認為該標籤適用，與其他標籤無關。分數**不會**加總為 1。回應中的 `score_semantics` 欄位標示評分方法（`siglip2_pairwise_sigmoid`）。

**問：`raw_similarity` 是什麼意思？**
答：影片影格嵌入與文字標籤嵌入之間未經縮放的餘弦相似度，在模型特定的評分轉換之前的原始值。適合需要自行進行分數處理的進階使用者。

**問：可以用於即時影片嗎？**
答：不行。ClipCC 設計用於離線影片分類，處理的是上傳的影片檔案，不支援即時串流。

**問：如何切換模型？**
答：在網頁 UI 中，從下拉選單選擇模型後點擊 **Load Model**。透過 API，發送 `POST /api/v1/models/load` 搭配 `{"model_id": "siglip2-large-patch16-384"}`。新模型若未快取則先下載，再載入。進行中的請求會先在舊模型上完成。

**問：什麼是 temporal（時序）聚合？**
答：時序模式（`aggregation=temporal`）逐影格分析信心分數，偵測標籤超過閾值的時間片段，並回傳含片段統計的時間軸。用於回答「每個事件何時發生、持續多久？」— 例如在行車記錄器畫面中找出交通違規的確切時間戳記。

**問：模型在重啟後還會保留嗎？**
答：會。Docker 將下載的模型快取在具名磁碟區（`clipcc-models`）中。原生安裝則快取在 `CLIP_CACHE_DIR`。模型只需下載一次。

### 安裝與 Docker

**問：建置時間很長，這正常嗎？**
答：是的，首次建置會下載約 1 GB 的依賴，包括 Python、PyTorch 和 transformers。後續建置會因為 Docker 快取層機制而大幅加速。模型權重在首次啟動時另外下載，不在建置過程中。

**問：Docker 映像檔有多大？**
答：CPU 版本約 1 GB，GPU（CUDA）版本約 5 GB。模型權重另外儲存在 Docker 磁碟區中。

**問：可以不用 Docker 執行嗎？**
答：可以。詳細的原生安裝指南（Windows、macOS、Linux）請參閱 [English README](README.md)。

**問：出現「Cannot connect to the Docker daemon」錯誤怎麼辦？**
答：啟動 Docker Desktop（macOS/Windows）或 Docker 服務（Linux：`sudo systemctl start docker`）。等幾秒鐘讓它完成初始化。

### API 使用

**問：如何選擇好的標籤？**
答：標籤應該要有區別性、具描述性，並涵蓋你預期的內容範圍。越具體的標籤效果越好。例如，`"person running on a track"` 比 `"movement"` 效果更好。

**問：什麼時候該用 `mean`、`max` 還是 `temporal`？**
答：當你想知道影片的主要內容時，使用 `mean`（「這段影片主要在講什麼？」）。當你想偵測某個內容是否在任何時間點出現過，使用 `max`（「影片中有沒有出現過這個？」）。當你需要知道每個標籤何時出現、持續多久，使用 `temporal`，搭配片段偵測和時間軸視覺化。

**問：可以送超過 10 個標籤嗎？**
答：不行。每次請求的限制是 3-10 個標籤。如果需要更多分類，請分多次請求，每次使用不同的標籤組合。

**問：提示模板有什麼用？**
答：模型會將影片影格與文字提示進行比對。將標籤包裝在自然語句中通常能獲得更好的準確度。預設的 `"This is a photo of {}."` 適用於一般場景。對於特定領域的內容，可以嘗試如 `"a surveillance camera recording of {}"` 或 `"a dashcam video showing {}"` 等模板。

### 效能與限制

**問：收到 429「Too many uploads」該怎麼辦？**
答：伺服器限制了同時上傳數以防止磁碟空間耗盡。請稍候片刻後重試。如果經常發生，可增加 `MAX_UPLOAD_CONCURRENCY` 的值。

**問：收到 504 逾時該怎麼辦？**
答：推論管線超過了 `REQUEST_TIMEOUT_SECONDS`（預設 300 秒）的限制。解決方案：使用較短的影片、降低 `fps`，或增加逾時時間。

**問：多個使用者可以同時呼叫 API 嗎？**
答：可以，但有限制。上傳並行數預設為 `MAX_CONCURRENT_REQUESTS + 2`。推論並行數在 CPU 上預設為 2，GPU 上預設為 1。超過限制的請求會收到 `429` 回應。

### 安全性

**問：直接部署到公網上安全嗎？**
答：對於正式環境部署，你應該：（1）設定 `API_KEY` 而非 `ALLOW_UNAUTHENTICATED`；（2）部署在反向代理（nginx、Caddy）後方，並啟用速率限制和 TLS；（3）限制網路存取。

**問：API 會儲存我的影片嗎？**
答：不會。上傳的影片會暫存到臨時目錄，處理完成後立即刪除。啟動時的清除程式也會清理先前異常終止所遺留的暫存檔案。請求完成後不會保留任何影片資料。
