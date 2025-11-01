package config

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type AgentConfig struct {
	AzureAPIKey       string
	AzureEndpoint     string
	AzureDeployment   string
	AzureAPIVersion   string
	MCPBaseURL        string
	PollInitial       time.Duration
	PollMax           time.Duration
	PollTimeout       time.Duration
	PollBackoffFactor float64
	WorklogFilename   string
	ProjectName       string
	WorkspaceDir      string
	GitHubToken       string
}

func FromEnv() (AgentConfig, error) {
	// Load .env if present (non-destructive)
	_ = loadDotenv(".env")

	apiKey := os.Getenv("AZURE_OPENAI_API_KEY")
	if apiKey == "" {
		return AgentConfig{}, errors.New("AZURE_OPENAI_API_KEY must be set")
	}

	endpoint := os.Getenv("AZURE_OPENAI_ENDPOINT")
	if endpoint == "" {
		return AgentConfig{}, errors.New("AZURE_OPENAI_ENDPOINT must be set")
	}
	if !strings.HasPrefix(endpoint, "https://") {
		return AgentConfig{}, errors.New("AZURE_OPENAI_ENDPOINT must start with 'https://'")
	}
	endpoint = strings.TrimRight(endpoint, "/")

	deployment := os.Getenv("AZURE_OPENAI_DEPLOYMENT")
	if deployment == "" {
		return AgentConfig{}, errors.New("AZURE_OPENAI_DEPLOYMENT must be set")
	}

	apiVersion := os.Getenv("AZURE_OPENAI_API_VERSION")
	if apiVersion == "" {
		apiVersion = "2024-12-01-preview"
	}

	baseURL := os.Getenv("MCP_BASE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8000/mcp/sse"
	}
	if !(strings.HasPrefix(baseURL, "http://") || strings.HasPrefix(baseURL, "https://")) {
		return AgentConfig{}, errors.New("MCP_BASE_URL must be a valid HTTP/HTTPS URL")
	}

	pollInitial := envSeconds("MCP_POLL_INITIAL_SECONDS", 2)
	pollMax := envSeconds("MCP_POLL_MAX_SECONDS", 30)
	pollTimeout := envSeconds("MCP_POLL_TIMEOUT_SECONDS", 600)
	if pollInitial >= pollMax {
		return AgentConfig{}, errors.New("MCP_POLL_INITIAL_SECONDS must be less than MCP_POLL_MAX_SECONDS")
	}
	if pollTimeout <= pollMax {
		return AgentConfig{}, errors.New("MCP_POLL_TIMEOUT_SECONDS must be greater than MCP_POLL_MAX_SECONDS")
	}

	project := os.Getenv("PROJECT_NAME")
	workspace := os.Getenv("WORKSPACE_DIR")
	if workspace == "" {
		workspace = "/home/pan/workspace"
	}

	backoff := 2.0
	if v := os.Getenv("MCP_POLL_BACKOFF_FACTOR"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f <= 1.0 {
			return AgentConfig{}, errors.New("MCP_POLL_BACKOFF_FACTOR must be a float greater than 1.0")
		}
		backoff = f
	}

	githubToken := os.Getenv("GITHUB_ACCESS_TOKEN")
	if githubToken == "" {
		return AgentConfig{}, errors.New("GITHUB_ACCESS_TOKEN must be set")
	}

	return AgentConfig{
		AzureAPIKey:       apiKey,
		AzureEndpoint:     endpoint,
		AzureDeployment:   deployment,
		AzureAPIVersion:   apiVersion,
		MCPBaseURL:        baseURL,
		PollInitial:       pollInitial,
		PollMax:           pollMax,
		PollTimeout:       pollTimeout,
		PollBackoffFactor: backoff,
		WorklogFilename:   "worklog.md",
		ProjectName:       project,
		WorkspaceDir:      workspace,
		GitHubToken:       githubToken,
	}, nil
}

func envSeconds(name string, def int) time.Duration {
	v := os.Getenv(name)
	if v == "" {
		return time.Duration(def) * time.Second
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		panic(fmt.Errorf("invalid integer for %s: %s", name, v))
	}
	return time.Duration(n) * time.Second
}

// loadDotenv loads key=value pairs into env if not already set.
func loadDotenv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Support simple KEY=VALUE lines
		if i := strings.Index(line, "="); i >= 0 {
			key := strings.TrimSpace(line[:i])
			val := strings.TrimSpace(line[i+1:])
			val = trimQuotes(val)
			if os.Getenv(key) == "" {
				_ = os.Setenv(key, val)
			}
		}
	}
	return scanner.Err()
}

func trimQuotes(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 {
		if (s[0] == '\'' && s[len(s)-1] == '\'') || (s[0] == '"' && s[len(s)-1] == '"') {
			return s[1 : len(s)-1]
		}
	}
	// Do not path-clean arbitrary values like URLs; return as-is
	return s
}
