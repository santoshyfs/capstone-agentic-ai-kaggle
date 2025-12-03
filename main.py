"""
Multi-Agent System Demo with:

- Agent powered by an LLM (stub LLMClient – easy to swap to OpenAI/Gemini/HF)
- Parallel agents
- Sequential agents
- Loop agents
- Tools:
    * MCP (stub)
    * custom tools (Math, Magento GraphQL)
    * built-in tools (Search stub, Code Execution)
    * OpenAPI tools
- Long-running operations (pause/resume agents)
- Sessions & Memory
- Context engineering (context compaction)
- Observability (logging, metrics)
- Agent evaluation
- A2A Protocol
- Agent deployment (FastAPI)
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# =======================
# Logging & Observability
# =======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [req_id=%(req_id)s] %(name)s - %(message)s",
)
logger = logging.getLogger("multi_agent_demo")

METRICS = {
    "requests_total": 0,
    "agent_runs_total": 0,
    "tool_calls_total": 0,
    "errors_total": 0,
    "avg_latency_ms": 0.0,
}

# =======================
# Session & Memory System
# =======================

class Message(BaseModel):
    role: Literal["user", "assistant", "system", "agent"]
    content: str
    agent_name: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: time.time())


class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, List[Message]] = {}

    def get_or_create_session(self, session_id: Optional[str]) -> str:
        if session_id and session_id in self.sessions:
            return session_id
        new_id = session_id or str(uuid.uuid4())
        self.sessions.setdefault(new_id, [])
        return new_id

    def add_message(self, session_id: str, message: Message):
        self.sessions.setdefault(session_id, []).append(message)

    def get_history(self, session_id: str) -> List[Message]:
        return self.sessions.get(session_id, [])


session_service = InMemorySessionService()


class MemoryBank:
    def __init__(self):
        self.storage: Dict[str, List[str]] = {}

    def add_fact(self, key: str, fact: str):
        self.storage.setdefault(key, []).append(fact)

    def get_facts(self, key: str) -> List[str]:
        return self.storage.get(key, [])


memory_bank = MemoryBank()

# =======================
# Context Engineering
# =======================

def compact_context(messages: List[Message], max_messages: int = 10) -> List[Message]:
    if len(messages) <= max_messages:
        return messages

    older = messages[:-max_messages]
    recent = messages[-max_messages:]

    summary_text = "Summary of earlier conversation: " + "; ".join(
        msg.content[:40].replace("\n", " ") + "..."
        for msg in older[:5]
    )

    summary_message = Message(
        role="system",
        content=summary_text,
        agent_name="context_compactor",
    )
    return [summary_message] + recent

# =======================
# LLM Client (stub)
# =======================

class LLMClient:
    """
    Stub LLM client – replace with OpenAI/Gemini/HF easily.
    For now it just echoes what the user said and lists available tools.
    """

    async def acomplete(
        self,
        system_prompt: str,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        user_text = last_user.content if last_user else ""
        tool_names = ", ".join(t["name"] for t in tools or [])
        return (
            f"[LLM Stub]\n"
            f"System: {system_prompt[:80]}...\n"
            f"User said: {user_text[:120]}\n"
            f"Available tools: {tool_names or 'none'}\n"
            "NOTE: In a real system, I would decide when to call tools like magento_product_tool."
        )


llm_client = LLMClient()

# =======================
# Tools
# =======================

class Tool:
    name: str
    description: str

    async def __call__(self, **kwargs) -> Any:
        raise NotImplementedError


class CustomMathTool(Tool):
    name = "math_tool"
    description = "Evaluate simple arithmetic expressions like '2+2*5'."

    async def __call__(self, expression: str) -> Any:
        import ast, operator as op

        allowed_ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.Mod: op.mod,
        }

        def eval_(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                if type(node.op) not in allowed_ops:
                    raise ValueError("Operator not allowed")
                return allowed_ops[type(node.op)](eval_(node.left), eval_(node.right))
            else:
                raise ValueError("Unsupported expression")

        try:
            node = ast.parse(expression, mode="eval").body
            return eval_(node)
        except Exception as e:
            return f"Error evaluating expression: {e}"


class SearchTool(Tool):
    name = "search_tool"
    description = "Search the web (stub)."

    async def __call__(self, query: str) -> Any:
        return {
            "query": query,
            "results": [
                {"title": "Fake result 1", "url": "https://example.com/1"},
                {"title": "Fake result 2", "url": "https://example.com/2"},
            ],
        }


class CodeExecTool(Tool):
    name = "code_exec_tool"
    description = "Safely execute small Python expressions (no imports)."

    async def __call__(self, code: str) -> Any:
        allowed_builtins = {"range": range, "len": len, "sum": sum}
        local_env = {}
        try:
            exec(
                code,
                {"__builtins__": allowed_builtins},
                local_env,
            )
            return {"output": local_env}
        except Exception as e:
            return {"error": str(e)}


class OpenAPITool(Tool):
    name = "openapi_tool"
    description = "Call an HTTP endpoint defined by OpenAPI schema (simplified)."

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def __call__(self, path: str, method: str = "GET", params: dict = None, body: dict = None) -> Any:
        async with httpx.AsyncClient(timeout=10) as client:
            url = self.base_url.rstrip("/") + "/" + path.lstrip("/")
            if method.upper() == "GET":
                resp = await client.get(url, params=params)
            else:
                resp = await client.request(method.upper(), url, json=body, params=params)
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return {"status_code": resp.status_code, "body": body}


class MCPTool(Tool):
    """
    MCP (Model Context Protocol) style tool stub.
    """

    name = "mcp_tool"
    description = "Call an MCP server tool (stub)."

    def __init__(self, mcp_server_url: str):
        self.mcp_server_url = mcp_server_url

    async def __call__(self, tool_name: str, args: dict) -> Any:
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.post(
                    self.mcp_server_url,
                    json={"tool": tool_name, "args": args},
                )
                return resp.json()
            except Exception as e:
                return {"error": f"MCP call failed: {e}"}


# ---------- Magento product cards helper ----------

def render_product_cards_html(items: List[Dict[str, Any]], search_text: str) -> str:
    """
    Returns an HTML snippet with pretty product cards for Magento items.
    Can be injected directly with innerHTML on the frontend.
    """
    if not items:
        return f"""
        <div class="product-cards">
          <p>No products found for <strong>{search_text}</strong>.</p>
        </div>
        """

    cards = []
    for item in items:
        name = item.get("name", "No name")
        sku = item.get("sku", "")
        url_key = item.get("url_key", "")
        price_info = (
            item.get("price_range", {})
            .get("minimum_price", {})
            .get("regular_price", {})
        )
        price_value = price_info.get("value")
        currency = price_info.get("currency", "")

        if price_value is not None:
            price_str = f"{price_value} {currency}"
        else:
            price_str = "Price not available"

        product_url = f"/{url_key}.html" if url_key else "#"

        card_html = f"""
        <div class="product-card">
          <div class="product-name">{name}</div>
          <div class="product-sku">SKU: {sku}</div>
          <div class="product-price">{price_str}</div>
          <a class="product-link" href="{product_url}" target="_blank">View Product</a>
        </div>
        """
        cards.append(card_html)

    return f"""
    <div class="product-cards">
      <h3>Magento Products for "{search_text}"</h3>
      <div class="product-card-grid">
        {''.join(cards)}
      </div>
    </div>
    """


class MagentoProductTool(Tool):
    """
    Real Magento GraphQL tool for jamna.clouddeploy.in.

    Calls GraphQL:
      query ($search: String!, $pageSize: Int) {
        products(search: $search, pageSize: $pageSize) {
          items {
            sku
            name
            url_key
            price_range {
              minimum_price {
                regular_price {
                  value
                  currency
                }
              }
            }
          }
        }
      }

    Returns BOTH structured data and HTML cards (html_cards) ready for UI.
    """

    name = "magento_product_tool"
    description = "Search Magento products on jamna.clouddeploy.in via GraphQL and return pretty HTML cards."

    def __init__(self, graphql_endpoint: str):
        self.graphql_endpoint = graphql_endpoint

    async def __call__(self, search_text: str, page_size: int = 10) -> Any:
        query = """
        query ($search: String!, $pageSize: Int) {
          products(search: $search, pageSize: $pageSize) {
            items {
              sku
              name
              url_key
              price_range {
                minimum_price {
                  regular_price {
                    value
                    currency
                  }
                }
              }
            }
          }
        }
        """

        variables = {
            "search": search_text,
            "pageSize": page_size,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.post(
                    self.graphql_endpoint,
                    json={"query": query, "variables": variables},
                    headers={"Content-Type": "application/json"},
                )
            except Exception as e:
                return {
                    "error": f"Magento GraphQL request failed: {e}",
                    "endpoint": self.graphql_endpoint,
                }

        try:
            data = resp.json()
        except Exception:
            return {
                "error": "Invalid JSON from Magento GraphQL",
                "status_code": resp.status_code,
                "text": resp.text,
            }

        if "errors" in data:
            return {
                "error": "Magento GraphQL returned errors",
                "status_code": resp.status_code,
                "errors": data["errors"],
            }

        items = (
            data.get("data", {})
            .get("products", {})
            .get("items", [])
        )

        html_cards = render_product_cards_html(items, search_text)

        return {
            "search_text": search_text,
            "page_size": page_size,
            "status_code": resp.status_code,
            "items": items,
            "html_cards": html_cards,
        }


tools_registry: Dict[str, Tool] = {
    "math_tool": CustomMathTool(),
    "search_tool": SearchTool(),
    "code_exec_tool": CodeExecTool(),
    "openapi_tool": OpenAPITool(base_url="https://jsonplaceholder.typicode.com"),
    "mcp_tool": MCPTool(mcp_server_url="http://localhost:9000/mcp"),
    "magento_product_tool": MagentoProductTool(
        graphql_endpoint="http://jamna.clouddeploy.in/graphql"
    ),
}

def tool_schemas():
    return [
        {"name": name, "description": tool.description}
        for name, tool in tools_registry.items()
    ]

# =======================
# Agent & A2A Protocol
# =======================

class AgentConfig(BaseModel):
    name: str
    system_prompt: str
    tools: List[str] = []
    loop: bool = False
    max_iterations: int = 3


class Agent:
    def __init__(self, config: AgentConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    async def run(
        self,
        session_id: str,
        user_message: str,
        history: List[Message],
        context: Dict[str, Any],
    ) -> Any:
        """
        Returns either:
        - plain string (LLM output), OR
        - dict with:
            {
              "agent_output": str,
              "tool_used": str,
              "tool_args": dict,
              "tool_result": Any
            }

        So orchestrator can expose tool_result separately (e.g., as HTML cards).
        """
        METRICS["agent_runs_total"] += 1

        compacted_history = compact_context(history)
        messages = compacted_history + [
            Message(role="user", content=user_message)
        ]

        available_tools = [
            {"name": t, "description": tools_registry[t].description}
            for t in self.config.tools
            if t in tools_registry
        ]

        llm_output = await self.llm.acomplete(
            system_prompt=self.config.system_prompt,
            messages=messages,
            tools=available_tools,
        )

        # Simple heuristic: if user mentions "product" or "tyre" and this agent has magento_product_tool,
        # we proactively call the tool.
        tool_result = None
        tool_name = None
        tool_args = None

        lower_msg = user_message.lower()
        for t in self.config.tools:
            if t == "magento_product_tool":
                tool_name = t
                tool_args = {"search_text": user_message, "page_size": 8}
                METRICS["tool_calls_total"] += 1
                tool = tools_registry[t]
                tool_result = await tool(**tool_args)

                # Save tool output into history (as text)
                session_service.add_message(
                    session_id,
                    Message(
                        role="agent",
                        content=f"TOOL_RESULT[{t}]: {str(tool_result)[:500]}",
                        agent_name=self.config.name,
                    ),
                )

                return {
                    "agent_output": llm_output,
                    "tool_used": tool_name,
                    "tool_args": tool_args,
                    "tool_result": tool_result,
                }

        # Loop simulation (still applies)
        if self.config.loop:
            for i in range(1, self.config.max_iterations):
                llm_output += f"\n[Loop iteration {i+1} simulated]"
                await asyncio.sleep(0)

        return llm_output


class AgentMessage(BaseModel):
    from_agent: str
    to_agent: str
    content: str
    session_id: str


class AgentRouter:
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents

    async def route(self, msg: AgentMessage) -> str:
        target = self.agents.get(msg.to_agent)
        if not target:
            return f"Agent '{msg.to_agent}' not found."

        history = session_service.get_history(msg.session_id)
        response = await target.run(
            session_id=msg.session_id,
            user_message=f"[From {msg.from_agent}] {msg.content}",
            history=history,
            context={},
        )
        session_service.add_message(
            msg.session_id,
            Message(role="agent", content=msg.content, agent_name=msg.from_agent),
        )
        session_service.add_message(
            msg.session_id,
            Message(role="agent", content=str(response)[:500], agent_name=msg.to_agent),
        )
        return response

# =======================
# Preconfigured Agents
# =======================

research_agent = Agent(
    AgentConfig(
        name="research_agent",
        system_prompt="You are a research assistant that uses search_tool.",
        tools=["search_tool"],
        loop=False,
    ),
    llm_client,
)

coding_agent = Agent(
    AgentConfig(
        name="coding_agent",
        system_prompt="You are a coding assistant. Use code_exec_tool for small snippets.",
        tools=["code_exec_tool", "math_tool"],
        loop=False,
    ),
    llm_client,
)

ecommerce_agent = Agent(
    AgentConfig(
        name="ecommerce_agent",
        system_prompt=(
            "You are an ecommerce/Magento agent. Use magento_product_tool and openapi_tool when helpful."
        ),
        tools=["magento_product_tool", "openapi_tool"],
        loop=True,
        max_iterations=2,
    ),
    llm_client,
)

agents_registry: Dict[str, Agent] = {
    "research_agent": research_agent,
    "coding_agent": coding_agent,
    "ecommerce_agent": ecommerce_agent,
}

router = AgentRouter(agents_registry)

# =======================
# Long-running Agent Jobs
# =======================

class AgentJob(BaseModel):
    id: str
    agent_name: str
    session_id: str
    user_message: str
    status: Literal["running", "paused", "completed"]
    progress: int = 0


jobs_store: Dict[str, AgentJob] = {}


async def _run_long_job(job_id: str):
    job = jobs_store[job_id]
    agent = agents_registry[job.agent_name]
    history = session_service.get_history(job.session_id)

    for step in range(1, 11):
        while jobs_store[job_id].status == "paused":
            await asyncio.sleep(0.5)

        if jobs_store[job_id].status != "running":
            return

        await asyncio.sleep(0.5)
        jobs_store[job_id].progress = step * 10

    response = await agent.run(
        session_id=job.session_id,
        user_message=job.user_message + " (long-running job finalization)",
        history=history,
        context={},
    )
    session_service.add_message(
        job.session_id,
        Message(
            role="agent",
            content=f"[Long Job {job.id}] {str(response)[:500]}",
            agent_name=job.agent_name,
        ),
    )
    jobs_store[job_id].status = "completed"

# =======================
# Agent Evaluation
# =======================

class EvaluationRequest(BaseModel):
    expected: str
    actual: str


class EvaluationResult(BaseModel):
    score: float
    feedback: str


def evaluate_response(expected: str, actual: str) -> EvaluationResult:
    import re

    def tokens(text: str):
        return {
            t.lower()
            for t in re.findall(r"\w+", text)
            if len(t) > 2
        }

    exp_tokens = tokens(expected)
    act_tokens = tokens(actual)
    if not exp_tokens:
        return EvaluationResult(score=0.0, feedback="No expectation provided.")

    overlap = len(exp_tokens & act_tokens) / len(exp_tokens)
    feedback = f"Token overlap: {overlap:.2f}. "
    if overlap > 0.7:
        feedback += "Response is close to expectation."
    elif overlap > 0.4:
        feedback += "Response partially matches expectation."
    else:
        feedback += "Response differs significantly from expectation."
    return EvaluationResult(score=overlap, feedback=feedback)

# =======================
# Orchestration Logic
# =======================

async def run_sequential_agents(
    agent_names: List[str],
    session_id: str,
    user_message: str,
) -> Dict[str, Any]:
    history = session_service.get_history(session_id)
    responses: Dict[str, Any] = {}
    current_input = user_message

    for name in agent_names:
        agent = agents_registry[name]
        response = await agent.run(
            session_id=session_id,
            user_message=current_input,
            history=history,
            context={},
        )

        # If agent returned tool info, split it out
        if isinstance(response, dict) and "tool_result" in response:
            responses[name] = response.get("agent_output", "")
            responses[name + "_tool"] = response["tool_result"]
        else:
            responses[name] = response

        current_input = str(response)
        session_service.add_message(
            session_id,
            Message(role="agent", content=str(response)[:500], agent_name=name),
        )

    return responses


async def run_parallel_agents(
    agent_names: List[str],
    session_id: str,
    user_message: str,
) -> Dict[str, Any]:
    history = session_service.get_history(session_id)

    async def run_for_agent(name: str):
        agent = agents_registry[name]
        resp = await agent.run(
            session_id=session_id,
            user_message=user_message,
            history=history,
            context={},
        )
        session_service.add_message(
            session_id,
            Message(role="agent", content=str(resp)[:500], agent_name=name),
        )

        if isinstance(resp, dict) and "tool_result" in resp:
            return {
                "name": name,
                "agent_output": resp.get("agent_output", ""),
                "tool_result": resp["tool_result"],
            }
        else:
            return {"name": name, "agent_output": resp, "tool_result": None}

    tasks = [run_for_agent(name) for name in agent_names]
    results = await asyncio.gather(*tasks)

    combined: Dict[str, Any] = {}
    for r in results:
        combined[r["name"]] = r["agent_output"]
        if r["tool_result"] is not None:
            combined[r["name"] + "_tool"] = r["tool_result"]
    return combined


async def run_loop_agent_orchestration(
    agent_name: str,
    session_id: str,
    user_message: str,
    max_loops: int = 3,
) -> Dict[str, Any]:
    history = session_service.get_history(session_id)
    agent = agents_registry[agent_name]
    responses: Dict[str, Any] = {}

    current_input = user_message
    for i in range(max_loops):
        resp = await agent.run(
            session_id=session_id,
            user_message=current_input,
            history=history,
            context={"loop_index": i},
        )
        tag = f"{agent_name}_loop_{i+1}"

        if isinstance(resp, dict) and "tool_result" in resp:
            responses[tag] = resp.get("agent_output", "")
            responses[tag + "_tool"] = resp["tool_result"]
        else:
            responses[tag] = resp

        session_service.add_message(
            session_id,
            Message(role="agent", content=str(resp)[:500], agent_name=agent_name),
        )
        current_input = str(resp)

    return responses

# =======================
# FastAPI App Setup
# =======================

app = FastAPI(title="Multi-Agent System Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI (index.html) from ./static
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.responses import JSONResponse

@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    req_id = str(uuid.uuid4())
    request.state.req_id = req_id
    logger_adapter = logging.LoggerAdapter(
        logger, extra={"req_id": req_id}
    )
    start = time.time()
    METRICS["requests_total"] += 1

    try:
        response = await call_next(request)
        delta = (time.time() - start) * 1000
        METRICS["avg_latency_ms"] = (
            METRICS["avg_latency_ms"] * 0.9 + delta * 0.1
        )
        logger_adapter.info(
            f"{request.method} {request.url.path} completed in {delta:.2f}ms"
        )
        return response
    except Exception as e:
        METRICS["errors_total"] += 1
        logger_adapter.exception(f"Error processing request: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# =======================
# API Schemas & Endpoints
# =======================

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: str
    mode: Literal["sequential", "parallel", "loop"] = "sequential"
    agents: Optional[List[str]] = None
    remember_fact: Optional[bool] = False
    memory_key: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    mode: str
    agent_responses: Dict[str, Any]
    history_size: int


@app.get("/metrics")
async def get_metrics():
    return METRICS


@app.get("/tools")
async def list_tools():
    return tool_schemas()


@app.get("/agents")
async def list_agents():
    return {
        name: {
            "system_prompt": agent.config.system_prompt,
            "tools": agent.config.tools,
            "loop": agent.config.loop,
            "max_iterations": agent.config.max_iterations,
        }
        for name, agent in agents_registry.items()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = session_service.get_or_create_session(req.session_id)

    session_service.add_message(
        session_id,
        Message(role="user", content=req.user_message),
    )

    if req.remember_fact and req.memory_key:
        memory_bank.add_fact(req.memory_key, req.user_message)

    agent_names = req.agents or list(agents_registry.keys())
    agent_names = [a for a in agent_names if a in agents_registry]
    if not agent_names:
        raise ValueError("No valid agents specified.")

    if req.mode == "sequential":
        agent_responses = await run_sequential_agents(
            agent_names, session_id, req.user_message
        )
    elif req.mode == "parallel":
        agent_responses = await run_parallel_agents(
            agent_names, session_id, req.user_message
        )
    else:
        agent_responses = await run_loop_agent_orchestration(
            agent_names[0], session_id, req.user_message
        )

    history = session_service.get_history(session_id)
    return ChatResponse(
        session_id=session_id,
        mode=req.mode,
        agent_responses=agent_responses,
        history_size=len(history),
    )


@app.post("/a2a")
async def agent_to_agent(msg: AgentMessage):
    result = await router.route(msg)
    return {"response": result}


class StartJobRequest(BaseModel):
    agent_name: str
    session_id: Optional[str] = None
    user_message: str


@app.post("/jobs/start")
async def start_job(req: StartJobRequest):
    session_id = session_service.get_or_create_session(req.session_id)
    job_id = str(uuid.uuid4())
    job = AgentJob(
        id=job_id,
        agent_name=req.agent_name,
        session_id=session_id,
        user_message=req.user_message,
        status="running",
    )
    jobs_store[job_id] = job
    asyncio.create_task(_run_long_job(job_id))
    return job


@app.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    job = jobs_store[job_id]
    job.status = "paused"
    return job


@app.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    job = jobs_store[job_id]
    job.status = "running"
    asyncio.create_task(_run_long_job(job_id))
    return job


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    return jobs_store.get(job_id)


@app.get("/memory/{key}")
async def get_memory(key: str):
    return {"key": key, "facts": memory_bank.get_facts(key)}


@app.post("/evaluate", response_model=EvaluationResult)
async def evaluate(req: EvaluationRequest):
    return evaluate_response(req.expected, req.actual)





@app.get("/")
async def root():
    return {"message": "Multi-Agent System Demo. Open /static/index.html for UI."}

# To run:
#   uvicorn main:app --reload --port 8000
