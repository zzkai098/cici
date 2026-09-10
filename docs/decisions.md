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

## 8. harness 改成 async（2026-09-10）

参照 `cici_101/cli_project/`，但那个 repo async 的**理由是 MCP**——`core/tools.py::ToolManager` 每个 `await` 都是 MCP 网络往返。cici 不接 MCP，所以借的是它的类/方法形状，不是它的分层。

不接 MCP 的话，async 买到的是三样，都对着已排好的 roadmap：

1. **bash 的真超时与可取消**（roadmap 1）。`subprocess.run(timeout=)` 期间整个进程干等；`asyncio.create_subprocess_exec` + `asyncio.timeout` 才有 kill 路径。
2. **trajectory eval 的并发**（roadmap 4）。一组任务 `asyncio.gather` + `Semaphore`，入口就是 `Agent.run(query)`。
3. **prompt_toolkit**（roadmap 3）。`PromptSession.prompt_async` 本来就要求 event loop。

**`llm.py` 几乎没动。** `AsyncAnthropic().messages.stream()` 是**普通 `def`**，返回 `AsyncMessageStreamManager`（不是 coroutine），所以 `stream()` 的签名和 params 组装一个字都没改，只是 `Anthropic` → `AsyncAnthropic` 加一个 `aclose()`（异步 client 不像同步版能被 GC 兜底关连接池）。

`AnthropicProvider` 改名 `Claude`，对齐 `cici_101/cli_project/core/claude.py`；`text_from_message` 也从那里搬过来。

cici_101 的 `add_user_message` / `add_assistant_message` 一个方法干两件事——把 `Message` 拆成 content，再 append 进 messages 数组。这两件事在 cici 里拆开：

- **拆包留在 `Claude.content_from_message`**——「`Message` 长什么样」是 provider 知识，放进 `Session` 就等于把 Anthropic 类型漏到 seam 后面。
- **append 归 `Session`**，它只收 content，两个方法完全对称。cici_101 把两件事塞在一起，只是因为那个 repo 没有会话对象可以交。

拆包用 `isinstance(message, Message)`，不用 `getattr(message, "content", message)`。鸭子类型版会拆**任何**带 `.content` 的对象——`httpx2.Response` 就有——那会把一坨 bytes 安静地塞进对话历史，而不是当场报错。（`ParsedMessage` 是 `Message` 的子类，`isinstance` 在 1.x 上照样成立。）

存 blocks 原样而不是抽文本，是硬要求不是风格：开了 `display: "summarized"` 之后 content 里会带 `ThinkingBlock`，同一模型上继续对话必须原样回传，否则带 `tool_use` 的那轮下次请求会 400。

**从 cli_project 明确不借的**：`ToolManager._find_client_with_tool` 每次工具调用都遍历 client 做一次 `list_tools()` 网络往返来路由工具名，字典查找严格更好；而且 `core/tools.py:100` 的 `except` 分支引用了可能未绑定的 `tool_output`（NameError）。

## 9. 工具契约定成 async——趁工具还没写

`Tool.run` 是 `async def`。纯本地阻塞 I/O 的工具继承 `SyncTool`，只写同步 `_run`，基类用 `asyncio.to_thread` 包一层。

时机是理由：改这条契约的时候 6 个工具全是 `NotImplementedError`，成本是零；等 roadmap 1 写完 4 个再改就是 4 处返工。

**`except Exception` 必须保持是 `Exception`，不许改成 `BaseException`。** `asyncio.CancelledError` 继承自 `BaseException`，所以取消会正确穿透出去，不会被包成一个假的 `is_error` tool_result；而 bash 超时抛的 `TimeoutError` 属于 `Exception`，会被正常包成 `is_error` 回传给模型。这个分工是免费拿到的，改成 `BaseException` 就把取消吞了。

顺带修了一个：`TimeoutError` 的 `str()` 是空的，原来的 `f"Error: {e}"` 会给模型回一句 `Error: `。改成 `f"Error: {str(e) or type(e).__name__}"`。

## 10. `repl.py` 用 `asyncio.Runner`，不是 `asyncio.run`

**两条看起来更自然的写法都实测会让 Ctrl-C 挂死。** 在真 pty 上对照过（fifo/管道测不出来，管道下连改动前的同步 `input()` 都挂）：

| 形态 | Ctrl-C 结果 |
|---|---|
| 改动前的同步 `input()` | 干净退出 |
| `asyncio.run(main())` + 循环内 `input()` | **挂死** |
| `asyncio.Runner` + 循环外 `input()` | 干净退出 |

原因在 CPython 的 `asyncio/runners.py`：`Runner.run()` 会安装自己的 SIGINT handler，**第一次** Ctrl-C 是 `main_task.cancel()` 而不是抛 `KeyboardInterrupt`。主任务卡在同步 `input()` 里，取消永远送不进去，要按第二次才出得来。

`asyncio.Runner` 只在每次 `run()` 期间接管 SIGINT，提示符那一刻 handler 已经还原成默认的，行为跟改动前完全一致；同时一个 loop 跨多轮复用，`AsyncAnthropic` 的连接池不会因为换 loop 而失效。

**`input()` 也不能包 `asyncio.to_thread`。** Ctrl-C 的信号处理器跑在主线程，worker 里的 `input()` 按 PEP 475 只会重试 read 而不返回；`asyncio.run` 收尾时 `shutdown_default_executor()` 会去 join 这个永远不返回的线程，一样挂死。

等 roadmap 3 接上 prompt_toolkit，`await session.prompt_async("> ")` 从 loop 内部接管 stdin 并自己处理 Ctrl-C，那时 `repl.run()` 才变回 `async def`，Runner 就可以去掉。

## 11. 这台机器上 editable 安装失效的真正原因：macOS `UF_HIDDEN`

`uv run cici` 报 `ModuleNotFoundError: No module named 'cici'`，`.pth` 内容和路径都对。

真实原因：uv 写进 `.venv/lib/python3.12/site-packages/` 的**每一个条目都带 macOS `UF_HIDDEN` 标志**（`ls -lO` 能看到 `hidden`），而 Python 3.12 的 `site.addpackage` 里有这么一段——

```python
if ((getattr(st, 'st_flags', 0) & stat.UF_HIDDEN) or ...):
    _trace(f"Skipping hidden .pth file: {fullname!r}")
    return
```

**hidden 的 `.pth` 被静默跳过**，所以 `src` 从来没进过 `sys.path`。这跟 setuptools / flat layout / `__editable__` finder 都没关系——之前归因错了。普通包不受影响（import 机制不看 flags），只有 `.pth` 这条路径检查。

**解法**（2026-09-10 验证）：

```
chflags -R nohidden .venv
```

关键是 **`-R` 作用在整个 `.venv`**，不能只清 `*.pth`。uv 在 macOS 上会把 `.venv` 目录本身标成 hidden，只清 `.pth` 的话下次 `uv run` 重新写文件时又是 hidden——实测只能撑一次运行。整棵树清掉之后就稳了：连跑三次 `uv run --no-sync cici`、一次带 sync 的 `uv run cici`、再加一次 `uv sync --reinstall-package cici`（会重写 `.pth`），标志都没有回来。

**如果哪天删掉 `.venv` 重建（`rm -rf .venv && uv sync`），要重新执行一次**——uv 建 venv 的时候就会打上这个标志。

不依赖 venv 状态的验证方式仍然是 `PYTHONPATH=src .venv/bin/python -m cici`。

---

## 还没做的决定

- **推 GitHub** —— 现在是本地 repo，没有 remote。`gh repo create cici --private --source=. --push`，公开前先过一遍内容。
- **prompt_toolkit REPL** —— 现在是最小 `input()` 循环。`cici_101/cli_project/core/cli.py` 有现成的（`/命令` 补全、`@资源` mention、自定义 key bindings）。接上之后 `repl.run()` 变回 `async def`，`asyncio.Runner` 可以撤掉。
- **要不要给 `UF_HIDDEN` 加个自动化兜底**（见 §11）—— `chflags -R nohidden .venv` 已经是稳的解法，但重建 venv 后要手动跑一次。可以塞进 README 的 setup 步骤，或者等升级 uv 看上游是否已修（现在是 0.10.7，Homebrew 2026-02-27 的构建）。
