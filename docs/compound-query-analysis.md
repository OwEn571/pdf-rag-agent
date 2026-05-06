# 复合查询任务分解机制 — 完整代码分析

> 写给外部专家评审。梳理从入口到执行完毕的完整链路，标注关键数据流、分支逻辑和已知问题。

---

## 1. 入口：chat_runtime.py

```
用户查询
  ↓
run_compound_turn_if_needed()   ← 入口在 loop.py:54
  ↓
run_compound_query_if_needed()  ← 实现在 compound.py:90
  ↓
  成功？→ 返回 compound_response，不执行标准流程
  失败？→ 返回 None，fallback 到 run_standard_turn()
```

**关键判定**（compound.py:137）：`if len(subcontracts) < 2: return None`——分解结果少于 2 个子任务则跳过，走标准单查询路径。

---

## 2. 分解阶段：llm_decompose_compound_query()

**文件**：`app/services/planning/compound_tasks.py:78`

核心逻辑：调 LLM（`invoke_json`，修复后）输出 `{is_compound, subtasks[]}`，然后通过 `compound_contracts_from_decomposer_payload()` 转换为 `list[QueryContract]`。

**LLM Prompt 关键指令**（compound_tasks.py:42-75）：
```
"如果同一问题包含多个独立实体/论文/方法需要分别查询和比较，必须拆分为多个 subtask。"
"比较/综合使用 answer_slots=[comparison]，并且应放在其依赖的检索子任务之后。"
```

**GPT-4o 的分解输出**（实测）：
```json
{
  "is_compound": true,
  "subtasks": [
    {"clean_query": "GRPO 的定义、特点和应用", "answer_slots": ["entity_definition"], "targets": ["GRPO"], "interaction_mode": "research"},
    {"clean_query": "PPO 的定义、特点和应用",  "answer_slots": ["entity_definition"], "targets": ["PPO"],  "interaction_mode": "research"},
    {"clean_query": "DPO 的定义、特点和应用",  "answer_slots": ["entity_definition"], "targets": ["DPO"],  "interaction_mode": "research"},
    {"clean_query": "GRPO、PPO 和 DPO 的区别和联系", "answer_slots": ["comparison"], "targets": ["GRPO","PPO","DPO"], "interaction_mode": "conversation"}
  ]
}
```

**之前失败的根因**：`llm_decompose_compound_query` 原本优先使用 `invoke_json_messages`（走 `_chat_messages` 包装），GPT-4o 在该路径下返回 `is_compound: false`，拒绝分解。改为优先 `invoke_json`（直接 SystemMessage + HumanMessage）后正常产出 4 个子任务。

---

## 3. 执行阶段：run_compound_query_if_needed()

**文件**：compound.py:90-319

### 3.1 顺序执行（compound.py:184-250）

```python
for index, sub_contract in enumerate(subcontracts, start=1):
    if sub_contract.relation == "comparison_synthesis":
        # 综合子任务：不调 Agent，直接调 compose_compound_comparison_answer()
        comparison = compose_compound_comparison_answer(
            query=clean_query,
            subtask_results=subtask_results,   # ← 依赖前 3 步的结果
            ...
        )
    elif sub_contract.interaction_mode == "conversation":
        # 对话子任务
        subtask_result = execute_compound_task_subagent(...)
    else:
        # 检索子任务（entity_definition, formula 等）
        subtask_result = execute_compound_task_subagent(...)
        # 检查是否需要澄清
        if verification.status == "clarify":
            return compound_clarification_response(...)  ← 中断，等用户澄清
```

### 3.2 子任务执行：execute_compound_task_subagent()

**文件**：compound.py:52-87 → task.py:25-70

```python
def run_task_subagent(agent, prompt, contract, ...):
    sub_contract = contract or extract_agent_query_contract(agent, query=prompt, ...)
    sub_plan = agent.planner.plan_actions(contract=sub_contract, ...)
    
    if sub_contract.interaction_mode == "conversation":
        sub_state = agent.runtime.execute_conversation_tools(...)
    else:
        sub_state = agent.runtime.run_research_agent_loop(...)  ← 走标准 Agent Loop
```

**注意**：子任务使用的是 **同一个 Agent 实例**（同一个 `agent.retriever`、`agent.clients`），理论上检索行为与单查询一致。

### 3.3 澄清中断机制（compound.py:231-250）

```python
if sub_verification.status == "clarify":
    return compound_clarification_response(
        blocked_contract=sub_result_contract,
        subcontracts=subcontracts,
        blocked_index=index,
    )
```

**返回给前端**：已完成的子任务答案 + "在继续前，我需要先确认 XXX 的含义..." + 选项列表。

**用户选择后**（compound.py:109-119）：
```python
resumed_plan = pending_compound_plan(session=session)  # 从 working_memory 恢复
if resumed_plan is not None:
    selected_contract = contract_from_pending_clarification(...)
    subcontracts = pending_compound_subcontracts(plan, selected_contract)
```

会话状态（`pending_compound_plan`）存储在 `SessionContext.working_memory` 中，由 `store_pending_compound_plan` 写入（compound.py:422-435），在 SQLite sessions 表中随 SessionContext 一起持久化。

---

## 4. 综合阶段：compose_compound_comparison_answer()

**文件**：compound_tasks.py:391

将前 N 个子任务的 answer 作为 context，调 LLM 生成对比总结。对比表格的 `特性 | GRPO | PPO | DPO` 结构由此函数生成。

---

## 5. 已知问题

### 5.1 体验问题：逐个澄清而非批量

GRPO/PPO/DPO 如果都需要消歧，当前是按顺序逐个阻塞——第 1 个 sub-task 遇到 `clarify` 就 return，第 2、3 个根本没执行。等用户澄清完第 1 个，恢复后第 2 个又卡。

**建议**：在执行前先预检所有 subtask 是否需要澄清，一次性列出所有需要澄清的选项。

### 5.2 正确性问题：GRPO subtask 答案错误

GRPO subtask 第一次回答输出了错误的 "Gradient-based Policy Optimization"。这与用户单独问 "GRPO是什么" 答对的情况矛盾。

**可能原因**：
- `run_task_subagent` 中调用了 `extract_agent_query_contract(agent, query=prompt, ...)`（task.py:38），这会重新调 Router。如果 subtask 的 `prompt` 是分解器生成的 "GRPO 的定义、特点和应用" 而非原始 "GRPO是什么"，GPT-4o Router 可能产生不同的路由/提取结果。
- subtask 的 `contract` 参数已经提供了 targets=["GRPO"]（来自分解器），但 task.py 优先用 `contract or extract_agent_query_contract(...)`——如果 `contract` 不为 None，直接使用，避免了 Router 干扰。实测中 `contract` 被传入，所以应使用分解器的 targets。需要进一步排查。

### 5.3 比较综合的 citation 丢失

`compose_compound_comparison_answer` 只拿到了前几步的 `answer` 文本，没有 citations。最终对比表格里的证据引用可能不完整。

---

## 6. 架构总结

```
入口 (chat_runtime)
  ↓
run_compound_turn_if_needed  ← 总尝试复合分解
  ↓
llm_decompose_compound_query ← LLM 分解为 subtask[]
  ↓ len < 2? → 放弃，走 run_standard_turn
  ↓
顺序执行每个 subtask：
  ├─ entity_definition → execute_compound_task_subagent → Agent Loop (search→solve→verify→compose)
  ├─ entity_definition → ...
  ├─ entity_definition → ...
  └─ comparison_synthesis → compose_compound_comparison_answer (纯 LLM 对比汇总)
  ↓ 中途遇到 clarify?
  ├─ 存储 pending_compound_plan → return clarification 给前端
  └─ 用户选择 → 下一个 turn 恢复 pending plan → 继续执行
  ↓ 全部完成
  ├─ remember_compound_outcome → working_memory
  └─ 返回 AssistantResponse
```
