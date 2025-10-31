package tools

import (
    "bufio"
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "strings"
    "time"
)

import "dev_agent_go/internal/logx"
type MCPError struct{ Msg string }

func (e MCPError) Error() string { return e.Msg }

type MCPClient struct {
    rpcURL     string
    timeout    time.Duration
    maxRetries int
    sessionID  string
    client     *http.Client
    requestID  int
}

func NewMCPClient(baseURL string) *MCPClient {
    base := strings.TrimRight(baseURL, "/")
    if base == "" {
        base = "http://localhost:8000/mcp/sse"
    }
    return &MCPClient{
        rpcURL:     base,
        timeout:    30 * time.Second,
        maxRetries: 3,
        sessionID:  fmt.Sprintf("%d", time.Now().UnixNano()),
        client:     &http.Client{Timeout: 60 * time.Second},
    }
}

func (c *MCPClient) rpcPost(url string, body map[string]any, timeout time.Duration) (*http.Response, error) {
    payload, _ := json.Marshal(body)
    req, _ := http.NewRequest("POST", url, bytes.NewReader(payload))
    req.Header.Set("Accept", "application/json, text/event-stream")
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Mcp-Session-Id", c.sessionID)
    return c.client.Do(req)
}

func (c *MCPClient) call(method string, params map[string]any, timeout time.Duration) (map[string]any, error) {
    c.requestID++
    payload := map[string]any{"jsonrpc": "2.0", "id": c.requestID, "method": method, "params": params}
    var lastErr error

    for attempt := 0; attempt < c.maxRetries; attempt++ {
        logx.Debugf("MCP POST %s attempt %d to %s", method, attempt+1, c.rpcURL)
        resp, err := c.rpcPost(c.rpcURL, payload, timeout)
        if err != nil {
            lastErr = err
        } else {
            defer resp.Body.Close()
            ct := resp.Header.Get("Content-Type")
            data, _ := io.ReadAll(resp.Body)
            if resp.StatusCode < 200 || resp.StatusCode >= 300 {
                logx.Errorf("MCP HTTP error %d for %s (CT=%s): %.500s", resp.StatusCode, method, ct, string(data))
                lastErr = fmt.Errorf("MCP HTTP %d: %s", resp.StatusCode, string(data))
            } else if strings.Contains(ct, "text/event-stream") {
                ssePreview := string(data)
                if len(ssePreview) > 1000 { ssePreview = ssePreview[:1000] }
                logx.Debugf("MCP SSE preview: %q", ssePreview)
                obj, err := parseSSEFirstJSON(string(data))
                if err != nil {
                    logx.Errorf("Failed to parse SSE JSON for %s. Content-Type: %s, Status: %d", method, ct, resp.StatusCode)
                    logx.Errorf("Full SSE response:\n%s", string(data))
                    lastErr = err
                } else {
                    return normalizeRPC(obj), nil
                }
            } else {
                var obj map[string]any
                if err := json.Unmarshal(data, &obj); err != nil {
                    logx.Errorf("MCP response not JSON (status %d, CT=%s). First 1000 bytes: %q", resp.StatusCode, ct, string(data[:min(1000,len(data))]))
                    lastErr = err
                } else {
                    return normalizeRPC(obj), nil
                }
            }
        }
        if attempt < c.maxRetries-1 {
            wait := time.Duration(1<<attempt) * time.Second
            logx.Warningf("MCP call %s failed (attempt %d/%d): %v. Retrying in %ds...", method, attempt+1, c.maxRetries, lastErr, int(wait.Seconds()))
            time.Sleep(wait)
        }
    }
    if lastErr == nil {
        lastErr = MCPError{"Unknown MCP error"}
    }
    return nil, lastErr
}

func normalizeRPC(obj map[string]any) map[string]any {
    if errVal, ok := obj["error"]; ok {
        _ = errVal
        return obj
    }
    if res, ok := obj["result"].(map[string]any); ok {
        if sc, ok := res["structuredContent"].(map[string]any); ok {
            return sc
        }
        return res
    }
    return obj
}

func parseSSEFirstJSON(text string) (map[string]any, error) {
    // Split into events and try parsing each data line as JSON
    var events []map[string]any
    current := map[string]any{}
    scanner := bufio.NewScanner(strings.NewReader(text))
    for scanner.Scan() {
        line := strings.TrimRight(scanner.Text(), "\r")
        if line == "" {
            if len(current) > 0 {
                events = append(events, current)
                current = map[string]any{}
            }
            continue
        }
        if strings.HasPrefix(line, ":") {
            continue
        }
        if i := strings.Index(line, ":"); i >= 0 {
            field := strings.TrimSpace(line[:i])
            value := strings.TrimSpace(line[i+1:])
            if field == "event" {
                current["event"] = value
            } else if field == "data" {
                if _, ok := current["data"]; !ok {
                    current["data"] = []string{}
                }
                arr := current["data"].([]string)
                arr = append(arr, value)
                current["data"] = arr
            }
        }
    }
    if len(current) > 0 {
        events = append(events, current)
    }
    for _, ev := range events {
        dl, _ := ev["data"].([]string)
        for _, chunk := range dl {
            cand := strings.TrimSpace(chunk)
            if cand == "" || cand == "[DONE]" || cand == "DONE" {
                continue
            }
            if cand[0] == '{' || cand[0] == '[' {
                var obj map[string]any
                if json.Unmarshal([]byte(cand), &obj) == nil {
                    return obj, nil
                }
            }
        }
        joined := strings.TrimSpace(strings.Join(dl, "\n"))
        if joined != "" && (joined[0] == '{' || joined[0] == '[') {
            var obj map[string]any
            if json.Unmarshal([]byte(joined), &obj) == nil {
                return obj, nil
            }
        }
    }

    // Fallback: scan raw for first JSON object/array
    idx := strings.Index(text, "{")
    if idx >= 0 {
        dec := json.NewDecoder(bytes.NewReader([]byte(text[idx:])))
        var m map[string]any
        if dec.Decode(&m) == nil {
            return m, nil
        }
    }
    return nil, fmt.Errorf("no JSON data event in SSE response")
}

func (c *MCPClient) CallTool(name string, arguments map[string]any) (map[string]any, error) {
    return c.call("tools/call", map[string]any{"name": name, "arguments": arguments}, c.timeout)
}

func (c *MCPClient) ParallelExplore(projectName, parentBranchID string, prompts []string, agent string, numBranches int) (map[string]any, error) {
    return c.CallTool("parallel_explore", map[string]any{
        "project_name":          projectName,
        "parent_branch_id":      parentBranchID,
        "shared_prompt_sequence": prompts,
        "num_branches":          numBranches,
        "agent":                 agent,
    })
}

func (c *MCPClient) GetBranch(branchID string) (map[string]any, error) {
    // extend timeout for branch status
    return c.call("tools/call", map[string]any{
        "name":      "get_branch",
        "arguments": map[string]any{"branch_id": branchID},
    }, 300*time.Second)
}

func (c *MCPClient) BranchReadFile(branchID, filePath string) (map[string]any, error) {
    return c.CallTool("branch_read_file", map[string]any{"branch_id": branchID, "file_path": filePath})
}

func min(a, b int) int { if a < b { return a }; return b }
