"""
HyTE Observability Module using Arize Phoenix.

Provides clean, labeled tracing for the Deep Agent workflow.
- trace_node: Agent-level spans (selective state capture, no full dumps)
- trace_tool: Tool-level spans (named params, full output)
- trace_llm_call: LLM-level spans (prompt/response)
- log_decision: Explicit decision marker spans
"""

import functools
import inspect
import json
import os

try:
    import phoenix as px  # type: ignore[reportMissingImports]
except Exception:
    px = None

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor  # type: ignore[reportMissingImports]
except Exception:
    LangChainInstrumentor = None

try:
    from opentelemetry import trace as trace_api  # type: ignore[reportMissingImports]
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore[reportMissingImports]
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore[reportMissingImports]
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # type: ignore[reportMissingImports]
except Exception:
    trace_api = None
    OTLPSpanExporter = None
    TracerProvider = None
    SimpleSpanProcessor = None

# Global Tracer
tracer = None

# Keys to extract from GraphState for node-level input (everything else is ignored)
_STATE_KEYS_FOR_INPUT = [
    "hypothesis", "current_step", "latest_feedback", "current_kpi",
    "current_kpi_index", "kpi_list"
]

# Keys to extract from node output
_STATE_KEYS_FOR_OUTPUT = [
    "current_step", "kpi_list", "initial_strategy", "methodology",
    "metadata_context", "current_kpi"
]


def setup_observability():
    """Initializes Arize Phoenix and OpenTelemetry tracing."""
    global tracer

    if os.getenv("ENABLE_PHOENIX", "false").lower() != "true":
        tracer = None
        return None

    if px is None:
        print("Phoenix package is not installed. Observability is disabled.")
        tracer = None
        return None

    if trace_api is None or OTLPSpanExporter is None or TracerProvider is None or SimpleSpanProcessor is None:
        print("OpenTelemetry dependencies are not installed. Observability is disabled.")
        tracer = None
        return None

    local_trace_api = trace_api
    local_otlp_exporter = OTLPSpanExporter
    local_tracer_provider_cls = TracerProvider
    local_span_processor_cls = SimpleSpanProcessor
    
    session = px.launch_app(host="0.0.0.0", port=6006, run_in_thread=True)
    print(f"Phoenix Observability UI running at: {session.url}")
    
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    tracer_provider = local_tracer_provider_cls()
    tracer_provider.add_span_processor(local_span_processor_cls(local_otlp_exporter(endpoint)))
    local_trace_api.set_tracer_provider(tracer_provider)
    
    tracer = local_trace_api.get_tracer(__name__)
    
    if LangChainInstrumentor is not None:
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    
    return session.url


def _safe_json(obj, max_depth=3):
    """Safely serialize an object to a JSON string for span attributes."""
    def _simplify(o, depth=0):
        if depth > max_depth:
            return "..."
        if isinstance(o, dict):
            return {str(k): _simplify(v, depth + 1) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [_simplify(i, depth + 1) for i in o]
        elif isinstance(o, (str, int, float, bool)) or o is None:
            return o
        else:
            return str(o)
    try:
        return json.dumps(_simplify(obj), ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _extract_state_summary(state, keys):
    """Extract only the specified keys from the state dict."""
    if not isinstance(state, dict):
        return str(state)
    summary = {}
    for k in keys:
        val = state.get(k)
        if val is not None and val != "" and val != []:
            if isinstance(val, str) and len(val) > 200:
                summary[k] = val[:200] + "..."
            else:
                summary[k] = val
    return summary


def log_decision(decision_maker, decision_summary, details=None):
    """
    Log an explicit decision span. Use this after a router makes a choice.
    
    Args:
        decision_maker: Who made the decision (e.g., "Orchestrator", "RAG Router")
        decision_summary: Short label (e.g., "trigger_initial_strategy")
        details: Optional dict with reasoning or extra context
    """
    if not tracer:
        return
    
    with tracer.start_as_current_span(
        name=f"Decision: {decision_summary}",
        attributes={
            "openinference.span.kind": "CHAIN",
            "decision.maker": decision_maker,
            "decision.action": decision_summary,
            "input.value": _safe_json(details) if details else decision_summary,
            "output.value": decision_summary
        }
    ) as span:
        pass  # Decision spans are markers, no work happens inside


def _record_span_exception(span, error):
    span.record_exception(error)
    if trace_api is not None and hasattr(trace_api, "Status") and hasattr(trace_api, "StatusCode"):
        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))


def trace_node(node_name):
    """
    Decorator for LangGraph nodes (Orchestrator, Methodology, etc.)
    Creates a CHAIN span with selective state capture.
    
    Input shows: hypothesis, current_step, feedback, current_kpi
    Output shows: updated current_step, key artifacts produced
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find the state dict from args (handles both func(state) and self.func(state))
            state = None
            for arg in args:
                if isinstance(arg, dict) and "current_step" in arg:
                    state = arg
                    break
            
            if not tracer:
                return func(*args, **kwargs)
            
            # Build clean input summary
            input_summary = _extract_state_summary(state, _STATE_KEYS_FOR_INPUT) if state else {}
            
            with tracer.start_as_current_span(
                name=node_name,
                attributes={
                    "openinference.span.kind": "CHAIN",
                    "input.value": _safe_json(input_summary),
                }
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    
                    # Build clean output summary
                    output_summary = _extract_state_summary(result, _STATE_KEYS_FOR_OUTPUT) if isinstance(result, dict) else {}
                    if not isinstance(output_summary, dict):
                        output_summary = {"summary": str(output_summary)}
                    
                    # Add the message content if present
                    if isinstance(result, dict) and "messages" in result:
                        msgs = result["messages"]
                        if msgs and isinstance(msgs, list):
                            last_msg = msgs[-1].get("content", "") if isinstance(msgs[-1], dict) else str(msgs[-1])
                            output_summary["message"] = last_msg
                    
                    span.set_attribute("output.value", _safe_json(output_summary))
                    return result
                except Exception as e:
                    _record_span_exception(span, e)
                    raise e
        return wrapper
    return decorator


def trace_tool(tool_name):
    """
    Decorator for tool functions (RAG queries, code generation, routers).
    Creates a TOOL span with named parameter capture.
    
    Input: named parameters (skip self), full values
    Output: full result
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not tracer:
                return func(*args, **kwargs)
            
            # Build clean input from function signature
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            input_dict = {}
            for i, arg in enumerate(args):
                if i < len(param_names):
                    name = param_names[i]
                    if name == "self":
                        continue
                    input_dict[name] = arg
            input_dict.update(kwargs)
            
            with tracer.start_as_current_span(
                name=tool_name,
                attributes={
                    "openinference.span.kind": "TOOL",
                    "tool.name": tool_name,
                    "input.value": _safe_json(input_dict),
                }
            ) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("output.value", _safe_json(result))
                    return result
                except Exception as e:
                    _record_span_exception(span, e)
                    raise e
        return wrapper
    return decorator


def trace_llm_call(model_name):
    """
    Decorator for the low-level Gemini client call.
    Creates an LLM span with prompt/response capture.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(prompt, *args, **kwargs):
            if not tracer:
                return func(prompt, *args, **kwargs)
            
            with tracer.start_as_current_span(
                name=f"LLM: {model_name}",
                attributes={
                    "openinference.span.kind": "LLM",
                    "llm.model_name": model_name,
                    "input.value": str(prompt),
                }
            ) as span:
                try:
                    span.set_attribute("llm.input_messages", [json.dumps({"role": "user", "content": str(prompt)})])
                    
                    result = func(prompt, *args, **kwargs)
                    
                    span.set_attribute("output.value", str(result))
                    span.set_attribute("llm.output_messages", [json.dumps({"role": "assistant", "content": str(result)})])
                    
                    return result
                except Exception as e:
                    _record_span_exception(span, e)
                    raise e
        return wrapper
    return decorator
