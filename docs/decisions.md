# 决策记录

为什么是现在这样。目的是不要在下一个 session 里把同样的路再走一遍。

时间：2026-09-08 ~ 09-09（scaffold 那次 session）

---

## 1. Python，不是 TypeScript

参照的 [pi](https://github.com/earendil-works/pi) 是 TypeScript monorepo，但这里用 Python 重写。

理由不是技术优劣，是目标生态：这个项目要融进的是 Python 那一侧（PyTorch、数据与量化工具链）。用 TS 等于把它从那个生态里拿出来。

**借 pi 的分层，不借它的语言。** 被问到就直说参照了哪个开源项目、借了什么、为什么换语言 —— 照抄不承认是最差的答案。

## 2. uv，不是 venv + pip

最初用 `python3 -m venv` + pip，连撞两个坑：

- `pip install -e .` 报 `File "setup.py" not found` —— Python 3.9.0 自带 pip 20.2.3，PEP 660 editable 要 pip ≥21.3
- `Multiple top-level packages discovered in a flat-layout: ['cici', 'evals']`

换 uv 之后两个都不存在，而且顺手解决了 **Python 3.9.0**（anthropic 1.x 要求 ≥3.10）—— `uv python install 3.12` 一条命令，没碰系统 Python。

约定：
- `uv sync` 装依赖 · `uv run cici` 跑 · `uv add <pkg>` 加依赖
- **不要用 `pip install`**，会绕过 `uv.lock`
- `uv run` 每次自己检查并同步，日常不需要手动 `uv sync`

## 3. src layout + hatchling，不是 flat layout + setuptools ⚠️

**这条最容易被改回去，别改。**

症状：`.venv/bin/cici` 报 `ModuleNotFoundError: No module named 'cici'`，但 `uv run python -c "import cici"` 却成功。

真相：**后者是假阳性**。它成功只是因为 cwd 里正好有个 `cici/` 目录，跟安装完全无关。实际情况是 setuptools 的 `__editable__` finder 在这台机器上反复失效 —— 每次 `uv run` 重新同步后 `.pth` 就不执行了（连 venv 自带的 `_virtualenv.pth` 都不执行）。`uv sync --reinstall-package cici` 能临时修好，但跑一次 `ruff` 又坏。

没有继续查根因，因为换结构比查便宜且更彻底：

- **src layout** —— cwd 永远不可能伪装成 import 成功，问题暴露在第一次而不是第十次
- **hatchling** —— `uv init --package` 的默认后端，跟 uv 配合更稳，也不需要 `[tool.setuptools.packages.find] include = ["cici*"]` 那个补丁

## 4. anthropic 1.x：cici_101 里已经失效的东西

`anthropic>=0.40` 被解析到 1.4.0，跨了 major。现在钉 `>=1,<2`。

搬 cici_101 代码时会踩到：

- **assistant prefill 被移除** —— 在 Opus 5 / Sonnet 5 / 4.6+ 上**直接返回 400**。`02_prompt_eval.py` 靠「prefill + stop sequence 强制 JSON」，搬进 `evals/` 时必须换成 structured outputs：`output_config={"format": {"type": "json_schema", "schema": ...}}` 或 `client.messages.parse(output_format=Model)`。
  cici_101 当时确实这么做、也确实有效，那是历史事实；但**今天不会再这么写** —— 换成 structured outputs。
- **`temperature` / `top_p` / `top_k` 从 `messages.create()/.stream()` 移除**（TypeError）。cici_101 的 `chat()` 传了 `temperature=1.0`。
- **`thinking` 用 `{"type": "adaptive"}`**，`budget_tokens` 是 400。
- SDK 1.x 要求 **Python ≥3.10**，HTTP 层 `httpx` → `httpx2`（只在把 httpx 对象递给 SDK 时才需要改，这里没有）。

参数选择：`claude-opus-5` + `max_tokens=64000`（流式没有 HTTP 超时压力，4096 对 coding agent 太小）+ `thinking={"type":"adaptive","display":"summarized"}`（默认 `omitted` 在终端上是一段死等）。

## 5. thinking 渲染：别同时处理两种事件

第一版渲染出来文字**重复交错**。原因是同时处理了 `chunk.type == "thinking"` 和 `content_block_delta` 的 `thinking_delta` —— stream helper 对**同一份内容两种事件都发**，两个都打印就是每个 delta 打两遍。

**只处理 helper 合成的 `thinking` 事件**（跟 `text` 对称）。那句"防御性地两个都处理"的注释直接制造了 bug。

## 6. `.env.local` 按 repo 根解析，不按 cwd

原来是 `load_dotenv(".env.local")` —— 相对 cwd。**coding agent 的用途就是在别人的项目目录里跑**，按 cwd 找 key 必然找不到。现在用 `Path(__file__).resolve().parents[2] / ".env.local"`。

同时去掉了对 `ANTHROPIC_API_KEY` 的硬性检查：SDK 还接受 `ANTHROPIC_AUTH_TOKEN` 和 `ant auth login` 的 OAuth profile，硬卡会无理由拒绝配置正确的用户。改成让 SDK 解析，只把它的报错翻译成一行。

## 7. cici_101 已经踩过的流式坑

- `server_tool_use` 在 `content_block_start` 时 `input` 还是 `{}`，要等 `content_block_stop` 才落地
- `web_search_20260209` 同时授予 `code_execution`，要按 `block.name` 分派，不能假定是 search
- spinner 停止时必须 `\r\033[K` 擦行，否则真实输出会叠在动画上

---

## 还没做的决定

- **推 GitHub** —— 现在是本地 repo，没有 remote。`gh repo create cici --private --source=. --push`，公开前先过一遍内容。
- **prompt_toolkit REPL** —— 现在是最小 `input()` 循环。`cici_101/cli_project/core/cli.py` 有现成的（`/命令` 补全、`@资源` mention、自定义 key bindings）。
