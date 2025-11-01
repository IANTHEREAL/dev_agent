package brain

import (
	"bytes"
	"dev_agent/internal/logx"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type LLMBrain struct {
	apiKey     string
	endpoint   string
	deployment string
	apiVersion string
	maxRetries int
	client     *http.Client
}

func NewLLMBrain(apiKey, endpoint, deployment, apiVersion string, maxRetries int) *LLMBrain {
	if maxRetries <= 0 {
		maxRetries = 3
	}
	return &LLMBrain{
		apiKey:     apiKey,
		endpoint:   endpoint,
		deployment: deployment,
		apiVersion: apiVersion,
		maxRetries: maxRetries,
		client:     &http.Client{Timeout: 60 * time.Second},
	}
}

type chatCompletionRequest struct {
	Model               string           `json:"model"`
	Messages            []ChatMessage    `json:"messages"`
	MaxCompletionTokens int              `json:"max_completion_tokens,omitempty"`
	Tools               []map[string]any `json:"tools,omitempty"`
	ToolChoice          any              `json:"tool_choice,omitempty"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
}

func (b *LLMBrain) Complete(messages []ChatMessage, tools []map[string]any) (*chatCompletionResponse, error) {
	var lastErr error
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s", b.endpoint, b.deployment, b.apiVersion)

	body := chatCompletionRequest{
		Model:               b.deployment,
		Messages:            messages,
		MaxCompletionTokens: 4000,
	}
	if len(tools) > 0 {
		body.Tools = tools
		body.ToolChoice = "auto"
	}
	payload, _ := json.Marshal(body)

	for attempt := 0; attempt < b.maxRetries; attempt++ {
		req, _ := http.NewRequest("POST", url, bytes.NewReader(payload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("api-key", b.apiKey)

		resp, err := b.client.Do(req)
		if err != nil {
			lastErr = err
		} else {
			defer resp.Body.Close()
			data, _ := io.ReadAll(resp.Body)
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				var out chatCompletionResponse
				if err := json.Unmarshal(data, &out); err != nil {
					lastErr = err
				} else {
					return &out, nil
				}
			} else {
				lastErr = fmt.Errorf("azure openai error %d: %s", resp.StatusCode, string(data))
			}
		}

		if attempt < b.maxRetries-1 {
			wait := time.Duration(1<<attempt) * time.Second
			logx.Warningf("Azure OpenAI call failed (attempt %d/%d): %v. Retrying in %ds...", attempt+1, b.maxRetries, lastErr, int(wait.Seconds()))
			time.Sleep(wait)
		}
	}
	if lastErr == nil {
		lastErr = errors.New("unknown Azure OpenAI API error")
	}
	logx.Errorf("Azure OpenAI call failed after retries: %v", lastErr)
	return nil, lastErr
}
