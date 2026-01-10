# Real AI Backend

基于 FastAPI 的 AI 对话后端服务，提供 AI 聊天、文件解析等 API 功能。

## 功能特性

- **AI 聊天接口** - 支持多种 AI 模型，支持流式响应
- **文件解析接口** - 支持 PDF、Word、文本文件解析
- **模型管理接口** - 查询可用模型列表
- **Swagger/ReDoc 文档** - 自动生成 API 文档

## 项目结构

```
real_ai_backend/
├── app/                          # 主应用目录
│   ├── __init__.py               # 包初始化
│   ├── main.py                   # FastAPI 应用入口
│   │
│   ├── api/                      # API 路由层
│   │   └── v1/
│   │       ├── __init__.py       # 路由汇总
│   │       └── endpoints/        # API 端点
│   │           ├── chat.py       # 聊天接口
│   │           ├── files.py      # 文件解析接口
│   │           └── models.py     # 模型管理接口
│   │
│   ├── core/                     # 核心配置
│   │   └── config.py             # 应用配置
│   │
│   ├── models/                   # 数据库模型 (ORM)
│   │
│   ├── schemas/                  # Pydantic 数据模型
│   │   ├── chat.py               # 聊天请求/响应模型
│   │   ├── file.py               # 文件解析模型
│   │   └── model.py              # AI 模型信息
│   │
│   ├── services/                 # 业务逻辑层
│   │   ├── chat_service.py       # 聊天服务
│   │   └── model_service.py      # 模型管理服务
│   │
│   └── utils/                    # 工具函数
│
├── .env.example                  # 环境变量示例
├── Dockerfile                    # Docker 构建文件
├── requirements.txt              # Python 依赖
└── run.py                        # 启动脚本
```

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量示例
cp .env.example .env

# 编辑 .env 文件，配置必要的环境变量
# 主要需要配置 OPENROUTER_API_KEY
```

### 3. 启动服务

```bash
# 开发模式 (热重载)
python run.py

# 或使用 uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问 API 文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

## API 接口

### 聊天接口

```bash
# 发送消息
POST /api/v1/chat/completions
{
  "messages": [{"role": "user", "content": "你好"}],
  "model": "google/gemini-2.5-flash-lite"
}

# 流式聊天
POST /api/v1/chat/stream
```

### 文件解析接口

```bash
# 解析单个文件
POST /api/v1/files/parse
Content-Type: multipart/form-data
file: <上传文件>

# 批量解析
POST /api/v1/files/parse-batch
```

### 模型管理接口

```bash
# 获取所有模型
GET /api/v1/models

# 按提供商筛选
GET /api/v1/models?provider=Google

# 按类型筛选
GET /api/v1/models?type=vision

# 获取模型详情
GET /api/v1/models/{model_id}
```

## 支持的模型

| 提供商 | 模型 | 类型 |
|--------|------|------|
| Google | Gemini 3 Pro | vision |
| Google | Gemini 3 Flash | vision |
| Google | Gemini 2.5 Flash Lite | vision |
| OpenAI | GPT-5.2 | chat |
| OpenAI | GPT-5 Mini | vision |
| OpenAI | GPT-5 Image Mini | image-gen |
| Anthropic | Claude Opus 4.5 | vision |
| Anthropic | Claude Haiku 4.5 | vision |
| Alibaba | Qwen3 235B | chat |
| DeepSeek | DeepSeek V3.2 | chat |

## 实时语音识别

### 麦克风监听脚本

`mic_listener.py` 用于从麦克风实时捕获音频并通过 WebSocket 发送到后端进行语音识别。

#### 安装依赖

```bash
# 系统依赖 (Ubuntu/Debian)
apt-get install -y portaudio19-dev alsa-utils

# Python 依赖
pip install pyaudio websockets
```

#### 使用方法

```bash
# 列出可用音频设备
python mic_listener.py -l

# 使用默认设备启动监听
python mic_listener.py

# 指定设备索引
python mic_listener.py -d 1

# 指定后端地址和端口
python mic_listener.py --host 192.168.1.100 -p 8000
```

#### 命令参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-l`, `--list-devices` | 列出可用音频设备 | - |
| `-d`, `--device` | 音频输入设备索引 | 自动检测 |
| `--host` | WebSocket 服务器地址 | localhost |
| `-p`, `--port` | WebSocket 服务器端口 | 8000 |

#### 音频配置

| 参数 | 值 |
|------|-----|
| 采样率 | 48000 Hz |
| 声道 | 单声道 |
| 格式 | 16-bit PCM |
| 缓冲区 | 1024 帧 |

#### Docker 音频设备映射

在 `docker-compose.yml` 中添加设备映射：

```yaml
services:
  backend:
    devices:
      - "/dev/snd/controlC1:/dev/snd/controlC1"
      - "/dev/snd/pcmC1D0c:/dev/snd/pcmC1D0c"
      - "/dev/snd/pcmC1D0p:/dev/snd/pcmC1D0p"
```

#### WebSocket 协议

1. 客户端连接: `ws://host:port/api/v1/speech/recognize/stream`
2. 发送启动命令: `{"action": "start", "format": "pcm", "sample_rate": 48000}`
3. 持续发送二进制音频数据
4. 接收识别结果: `{"text": "识别的文字", "is_final": false}`
5. 发送停止命令: `{"action": "stop"}`

## Docker 部署

```bash
# 构建镜像
docker build -t real-ai-backend .

# 运行容器
docker run -d -p 8000:8000 --env-file .env real-ai-backend

# 运行容器 (带音频设备)
docker run -d -p 8000:8000 --env-file .env \
  --device /dev/snd/controlC1 \
  --device /dev/snd/pcmC1D0c \
  --device /dev/snd/pcmC1D0p \
  real-ai-backend
```

## 开发

```bash
# 代码格式化
black app/
isort app/

# 代码检查
flake8 app/

# 运行测试
pytest
```

## 环境变量说明

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `APP_NAME` | 应用名称 | Real AI Backend |
| `DEBUG` | 调试模式 | false |
| `OPENROUTER_API_KEY` | OpenRouter API Key | (必填) |
| `CORS_ORIGINS` | 允许的跨域来源 | ["*"] |
| `SECRET_KEY` | JWT 密钥 | (生产环境必须修改) |

## License

MIT
