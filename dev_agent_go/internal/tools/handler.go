package tools

import (
    "encoding/json"
    "fmt"
    "strings"
    "time"
)

import (
    "dev_agent_go/internal/logx"
)

type ToolExecutionError struct{ Msg string }

func (e ToolExecutionError) Error() string { return e.Msg }

type BranchTracker struct {
    start  string
    latest string
}

func NewBranchTracker(start string) *BranchTracker {
    return &BranchTracker{start: start, latest: start}
}

func (t *BranchTracker) Record(id string) {
    if id == "" {
        return
    }
    if t.start == "" {
        t.start = id
    }
    t.latest = id
}

func (t *BranchTracker) Range() map[string]string {
    return map[string]string{"start_branch_id": t.start, "latest_branch_id": t.latest}
}

type ToolHandler struct {
    client        *MCPClient
    defaultProj   string
    maxBranches   int
    branchTracker *BranchTracker
}

func NewToolHandler(client *MCPClient, defaultProject string, startBranch string) *ToolHandler {
    return &ToolHandler{
        client:        client,
        defaultProj:   defaultProject,
        maxBranches:   4,
        branchTracker: NewBranchTracker(startBranch),
    }
}

func (h *ToolHandler) BranchRange() map[string]string { return h.branchTracker.Range() }

// ToolCall mirrors brain.ToolCall, but we keep it generic here if needed.
type ToolCall struct {
    ID       string `json:"id"`
    Type     string `json:"type"`
    Function struct {
        Name      string `json:"name"`
        Arguments string `json:"arguments"`
    } `json:"function"`
}

func (h *ToolHandler) Handle(call ToolCall) map[string]any {
    name := call.Function.Name
    if name == "" {
        return h.errorPayload("Missing tool name in call.")
    }
    var args map[string]any
    if call.Function.Arguments != "" {
        if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
            return h.errorPayload(fmt.Sprintf("Invalid JSON arguments: %v", err))
        }
    } else {
        args = map[string]any{}
    }

    var res map[string]any
    var err error
    switch name {
    case "execute_agent":
        res, err = h.executeAgent(args)
    case "check_status":
        res, err = h.checkStatus(args)
    case "read_artifact":
        res, err = h.readArtifact(args)
    default:
        err = ToolExecutionError{Msg: fmt.Sprintf("Unsupported tool: %s", name)}
    }
    if err != nil {
        return h.errorPayload(err.Error())
    }
    return map[string]any{"status": "success", "data": res}
}

func (h *ToolHandler) executeAgent(arguments map[string]any) (map[string]any, error) {
    agent, _ := arguments["agent"].(string)
    prompt, _ := arguments["prompt"].(string)
    project := h.defaultProj
    if v, ok := arguments["project_name"].(string); ok && v != "" {
        project = v
    }
    parent, _ := arguments["parent_branch_id"].(string)
    nb, nbOK := arguments["num_branches"].(float64)
    numBranches := 1
    if nbOK {
        numBranches = int(nb)
    }

    if agent == "" || prompt == "" || parent == "" || project == "" {
        return nil, ToolExecutionError{Msg: "missing required arguments"}
    }
    if numBranches < 1 || numBranches > h.maxBranches {
        return nil, ToolExecutionError{Msg: fmt.Sprintf("num_branches must be 1..%d", h.maxBranches)}
    }

    logx.Infof("Executing agent %s on project %s from parent %s", agent, project, parent)
    resp, err := h.client.ParallelExplore(project, parent, []string{prompt}, agent, numBranches)
    if err != nil {
        return nil, err
    }
    if isErr, ok := resp["isError"].(bool); ok && isErr {
        return nil, ToolExecutionError{Msg: fmt.Sprintf("%v", resp["error"]) }
    }
    var branchID string
    if branches, ok := resp["branches"].([]any); ok && len(branches) > 0 {
        if m, ok := branches[0].(map[string]any); ok {
            branchID = extractBranchID(m)
        }
    }
    if branchID == "" {
        return nil, ToolExecutionError{Msg: "Missing branch id in parallel_explore response."}
    }
    h.branchTracker.Record(branchID)
    return map[string]any{"parallel_explore": resp, "branch_id": branchID}, nil
}

func (h *ToolHandler) checkStatus(arguments map[string]any) (map[string]any, error) {
    branchID, _ := arguments["branch_id"].(string)
    if branchID == "" {
        return nil, ToolExecutionError{Msg: "`branch_id` is required"}
    }
    timeout := 1800.0
    if v, ok := arguments["timeout_seconds"].(float64); ok && v > 0 {
        timeout = v
    }
    poll := 3.0
    if v, ok := arguments["poll_interval_seconds"].(float64); ok && v > 0 {
        poll = v
    }
    maxPoll := 30.0
    if v, ok := arguments["max_poll_interval_seconds"].(float64); ok && v >= poll {
        maxPoll = v
    }
    deadline := time.Now().Add(time.Duration(timeout) * time.Second)
    sleep := time.Duration(poll * float64(time.Second))

    logx.Infof("Checking status for branch %s (timeout=%ss)", branchID, int(timeout))
    for attempt := 1; ; attempt++ {
        resp, err := h.client.GetBranch(branchID)
        if err != nil {
            return nil, err
        }
        // Record/validate branch id
        if id := extractBranchID(resp); id != "" {
            h.branchTracker.Record(id)
        } else {
            return nil, ToolExecutionError{Msg: "Branch status response missing branch identifier."}
        }

        status := stringsLower(resp["status"]) 
        logx.Infof("Branch %s response (attempt %d): %s", branchID, attempt, toJSON(resp))
        if status == "succeed" || status == "failed" {
            return resp, nil
        }
        if time.Now().After(deadline) {
            return nil, ToolExecutionError{Msg: fmt.Sprintf("Timed out waiting for branch %s (last status=%s)", branchID, status)}
        }
        logx.Infof("Branch %s still active (status=%s). Sleeping %.1fs.", branchID, status, sleep.Seconds())
        time.Sleep(sleep)
        // exponential-ish backoff
        sleep = time.Duration(minFloat(float64(sleep/time.Second)*1.5, maxPoll)) * time.Second
    }
}

func (h *ToolHandler) readArtifact(arguments map[string]any) (map[string]any, error) {
    branchID, _ := arguments["branch_id"].(string)
    path, _ := arguments["path"].(string)
    if branchID == "" || path == "" {
        return nil, ToolExecutionError{Msg: "`branch_id` and `path` are required"}
    }
    logx.Infof("Reading artifact %s from branch %s", path, branchID)
    return h.client.BranchReadFile(branchID, path)
}

func extractBranchID(m map[string]any) string {
    for _, k := range []string{"branch_id", "id"} {
        if v, ok := m[k].(string); ok && v != "" {
            return v
        }
    }
    if b, ok := m["branch"].(map[string]any); ok {
        for _, k := range []string{"branch_id", "id"} {
            if v, ok := b[k].(string); ok && v != "" {
                return v
            }
        }
    }
    return ""
}

func (h *ToolHandler) errorPayload(msg string) map[string]any { return map[string]any{"status": "error", "error": msg} }

func stringsLower(v any) string {
    if v == nil { return "" }
    s, _ := v.(string)
    return stringsTrimLower(s)
}

func stringsTrimLower(s string) string {
    return strings.ToLower(strings.TrimSpace(s))
}

func minFloat(a, b float64) float64 {
    if a < b { return a }
    return b
}

// Tool schema to feed the LLM
func GetToolDefinitions() []map[string]any {
    return []map[string]any{
        {
            "type": "function",
            "function": map[string]any{
                "name":        "execute_agent",
                "description": "Launch an MCP parallel_explore job for a specialist agent.",
                "parameters": map[string]any{
                    "type": "object",
                    "properties": map[string]any{
                        "agent":            map[string]any{"type": "string", "description": "Target specialist agent name."},
                        "prompt":           map[string]any{"type": "string", "description": "Prompt for the agent."},
                        "project_name":     map[string]any{"type": "string", "description": "Pantheon project name."},
                        "parent_branch_id": map[string]any{"type": "string", "description": "Branch UUID to branch from."},
                        "num_branches":     map[string]any{"type": "integer", "description": "Optional number of sibling branches.", "default": 1, "minimum": 1, "maximum": 4},
                    },
                    "required": []any{"agent", "prompt", "project_name", "parent_branch_id"},
                },
            },
        },
        {
            "type": "function",
            "function": map[string]any{
                "name":        "check_status",
                "description": "Fetch status information for an MCP branch id.",
                "parameters": map[string]any{
                    "type": "object",
                    "properties": map[string]any{
                        "branch_id":                 map[string]any{"type": "string", "description": "Branch UUID returned from execute_agent."},
                        "timeout_seconds":           map[string]any{"type": "number", "default": 1800},
                        "poll_interval_seconds":     map[string]any{"type": "number", "default": 3},
                        "max_poll_interval_seconds": map[string]any{"type": "number", "default": 30},
                    },
                    "required": []any{"branch_id"},
                },
            },
        },
        {
            "type": "function",
            "function": map[string]any{
                "name":        "read_artifact",
                "description": "Read a text artifact produced by a branch.",
                "parameters": map[string]any{
                    "type": "object",
                    "properties": map[string]any{
                        "branch_id": map[string]any{"type": "string", "description": "Branch that produced the artifact."},
                        "path":      map[string]any{"type": "string", "description": "Artifact path or filename."},
                    },
                    "required": []any{"branch_id", "path"},
                },
            },
        },
    }
}

func toJSON(v any) string { b, _ := json.Marshal(v); return string(b) }
