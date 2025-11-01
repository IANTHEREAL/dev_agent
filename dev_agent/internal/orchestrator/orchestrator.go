package orchestrator

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"

	b "dev_agent/internal/brain"
	"dev_agent/internal/logx"

	t "dev_agent/internal/tools"
)

const systemPrompt = `You are a TDD (Test-Drive Development) workflow orchestrator.

### Agents
* **claude_code**: Implements solutions and tests. Summarizes work in '/home/pan/workspace/worklog.md'.
* **codex**: Reviews code for P0/P1 issues. Records findings in '/home/pan/workspace/worklog.md' and '/home/pan/workspace/codex_review.log'.

### Workflow
1.  **Implement (claude_code)**: Implement the solution and matching tests for the user's task.
2.  **Review (codex)**: Review the implementation for P0/P1 issues.
3.  **Fix (claude_code)**: If issues are found, fix all P0/P1 issues and ensure tests pass.
4.  Repeat **Review** and **Fix** until 'codex' reports no P0/P1 issues.

### Your Orchestration Rules
1.  **Call Agents**: For each workflow step, call 'execute_agent'.
2.  **Maintain State**: Track branch lineage ('parent_branch_id') and report any tool errors immediately.
3.  **Handle Review Data**: Before launching a **Fix** run, you **must** use 'read_artifact' to get the issues from 'codex_review.log'.

### Agent Prompt Templates

Don't go into too much detail. You're just a TDD manager, clearly explain the tasks and let the agent analyze and execute them. So please Use the following prompt, Fill in the correct task and issues.

#### Implement (claude_code)

You are a expert engineer, please Analyze user task or issue, then design, implement and test.

**User Task/Issue**: [The user's original task description - must be passed on exactly as is]

**Instructions**:
1.  **Analyze**: Analyze user intents and understand the existing codebase in the current directory in relation to the user task.
2.  **Design**: Before Implement, Must design a clear solution approach.
3.  **Implement & Test**: Write the implementation code and comprehensive tests following TDD principles.
    * Tests must validate the core logic of your implementation.
    * Cover critical paths and important edge cases.
    * Ensure all new and existing tests pass successfully.

Remeber you are linus, hate over engineering.

**Final Step**: After completing all work, append a summary for your changes and test result to '/home/pan/workspace/worklog.md'.

Ultrathink! Please give your best efforts!
---

#### Review (codex)

You are a expert engineer, perform a comprehensive code review to find P0 and P1 issues.

**User Task**: [The user's original task description - must be passed on exactly as is]

**Instructions**:
1.  **Read Context**: First, read '/home/pan/workspace/worklog.md' to understand the recent changes made by the developer.
2.  **Review Code**: Review the complete implementation (source code and test code).
3.  **Identify Issues**: Report only P0 (Critical) and P1 (Major) issues. Provide clear evidence for each issue found.
4.  **Validate Tests**:
	- Analyze and list the tests involved in the code modifications. We need to use them to prove correctness and prevent regression issues. If there are suspected P0/P1 issues and there are no corresponding tests, you need to add the corresponding tests to find the P1/P0 issues.
	- Before running test, critically assess if the tests genuinely prove the code works as intended; We reject any fabrication or hacking attempts to bypass the test.

**Issue Definitions**:
* **P0 (Critical - Must Fix)**
* **P1 (Major - Should Fix)**
* **DO NOT Report**: Style preferences, naming conventions, minor optimizations, or subjective "could be better" suggestions.
---

####  Fix (claude_code)

Ultrathink! Fix all P0/P1 issues reported in the review.

**Issues to Fix**:
[List of P0/P1 issues from '/home/pan/workspace/codex_review.log']

**Original User Task**: [The user's original task description - must be passed on exactly as is]

**Final Step**: After fixing all issues, append a summary of the fixes to '/home/pan/workspace/worklog.md'.

### Completion
* Stop Condition: Stop when a codex Review run reports no P0/P1 issues.
* Final Output: Reply with JSON only (no other text): {"is_finished": true, "task":"<original user task description>","summary":"<Concise outcome, e.g., 'Implementation and review complete. No P0/P1 issues found.'>"}

Ultrathink! Please give your best efforts!
`

const maxIterations = 8

type publishHandler interface {
	BranchRange() map[string]string
	Handle(t.ToolCall) map[string]any
}

type PublishOptions struct {
	GitHubToken    string
	WorkspaceDir   string
	ParentBranchID string
	ProjectName    string
	Task           string
}

func finalizeBranchPush(handler publishHandler, opts PublishOptions, report map[string]any, success bool) (string, error) {
	if opts.GitHubToken == "" {
		return "", errors.New("missing GitHub token for publish step")
	}
	lineage := handler.BranchRange()
	parent := lineage["latest_branch_id"]
	if parent == "" {
		parent = opts.ParentBranchID
	}
	if parent == "" {
		return "", errors.New("unable to determine parent branch id for publish step")
	}

	outcome := "Reached iteration limit before clean review sign-off."
	if success {
		summary := ""
		if report != nil {
			if s, ok := report["summary"].(string); ok && s != "" {
				summary = s
			}
		}
		if summary != "" {
			outcome = summary
		} else {
			outcome = "Workflow completed successfully."
		}
	} else {
		outcome = "Reached iteration limit before clean review sign-off."
	}

	meta := fmt.Sprintf("commit-meta: start_branch=%s latest_branch=%s", lineage["start_branch_id"], lineage["latest_branch_id"])
	tokenLiteral := strconv.Quote(opts.GitHubToken)
	prompt := fmt.Sprintf(`Finalize the task by committing and pushing the current workspace state.

Task: %s
Outcome: %s
GitHub access token (export for git auth and unset afterwards): %s
Meta (include in the commit message if helpful): %s

The worklog is located into '/home/pan/workspace/worklog.md'.

Choose an appropriate git branch name for this task, commit the related file changes (only files related to user task, don't commit intermediate files, like worklog, review log, temporary tests or scripts), and reply with the branch name and commit hash. Do not print the raw token anywhere except when configuring git.`, opts.Task, outcome, tokenLiteral, meta)

	logx.Infof("Finalizing workflow by asking claude_code to push from branch %s lineage.", parent)
	execArgs := map[string]any{
		"agent":            "claude_code",
		"prompt":           prompt,
		"parent_branch_id": parent,
	}
	if opts.ProjectName != "" {
		execArgs["project_name"] = opts.ProjectName
	}
	argsBytes, _ := json.Marshal(execArgs)
	execCall := t.ToolCall{Type: "function"}
	execCall.Function.Name = "execute_agent"
	execCall.Function.Arguments = string(argsBytes)

	execResp := handler.Handle(execCall)
	if status, _ := execResp["status"].(string); status != "success" {
		return "", fmt.Errorf("publish execute_agent failed: %v", execResp)
	}
	data, _ := execResp["data"].(map[string]any)
	branchID := extractBranchIDFromData(data)
	if branchID == "" {
		return "", errors.New("publish execute_agent missing branch id")
	}

	checkArgs := map[string]any{"branch_id": branchID}
	checkBytes, _ := json.Marshal(checkArgs)
	checkCall := t.ToolCall{Type: "function"}
	checkCall.Function.Name = "check_status"
	checkCall.Function.Arguments = string(checkBytes)

	checkResp := handler.Handle(checkCall)
	if status, _ := checkResp["status"].(string); status != "success" {
		return "", fmt.Errorf("publish check_status failed: %v", checkResp)
	}
	return branchID, nil
}

func extractBranchIDFromData(data map[string]any) string {
	if data == nil {
		return ""
	}
	if id, _ := data["branch_id"].(string); id != "" {
		return id
	}
	if id, _ := data["id"].(string); id != "" {
		return id
	}
	if branch, _ := data["branch"].(map[string]any); branch != nil {
		if id := extractBranchIDFromData(branch); id != "" {
			return id
		}
	}
	if pe, _ := data["parallel_explore"].(map[string]any); pe != nil {
		if branches, _ := pe["branches"].([]any); len(branches) > 0 {
			if m, _ := branches[0].(map[string]any); m != nil {
				return extractBranchIDFromData(m)
			}
		}
	}
	return ""
}

func BuildInitialMessages(task, projectName, workspaceDir, parentBranchID string) []b.ChatMessage {
	userPayload := map[string]any{
		"task":             task,
		"parent_branch_id": parentBranchID,
		"project_name":     projectName,
		"workspace_dir":    workspaceDir,
		"notes":            "For every phase: craft an execute_agent prompt covering task, phase goal, context. Track branch lineage and stop when codex reports no P0/P1 issues.",
	}
	content, _ := json.MarshalIndent(userPayload, "", "  ")
	return []b.ChatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: string(content)},
	}
}

func assistantMessageToDict(msg b.ChatMessage) b.ChatMessage {
	// Already in the correct structure
	return msg
}

func ParseFinalReport(msg b.ChatMessage) (map[string]any, bool) {
	if msg.Content == "" {
		return nil, false
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(msg.Content), &m); err != nil {
		return nil, false
	}
	if m["is_finished"] == true {
		return m, true
	}
	return nil, false
}

func Orchestrate(brain *b.LLMBrain, handler *t.ToolHandler, messages []b.ChatMessage, publishOpts PublishOptions) (map[string]any, error) {
	tools := t.GetToolDefinitions()
	var (
		finalReport map[string]any
		finished    bool
		reviewCount int
	)

	for i := 1; ; i++ {
		logx.Infof("LLM iteration %d", i)
		resp, err := brain.Complete(messages, tools)
		if err != nil {
			return nil, err
		}
		choice := resp.Choices[0].Message
		messages = append(messages, assistantMessageToDict(choice))

		if len(choice.ToolCalls) > 0 {
			reviewCompleted := false
			for _, tc := range choice.ToolCalls {
				var args map[string]any
				if tc.Function.Arguments != "" {
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				}
				htc := t.ToolCall{ID: tc.ID, Type: tc.Type}
				htc.Function.Name = tc.Function.Name
				htc.Function.Arguments = tc.Function.Arguments
				result := handler.Handle(htc)
				toolMsg := b.ChatMessage{Role: "tool", ToolCallID: tc.ID, Content: toJSON(result)}
				messages = append(messages, toolMsg)

				if tc.Function.Name == "execute_agent" {
					if agent, _ := args["agent"].(string); agent == "codex" {
						if status, _ := result["status"].(string); status == "success" {
							reviewCompleted = true
						}
					}
				}
			}
			if reviewCompleted {
				reviewCount++
				logx.Infof("Completed review iteration %d/%d", reviewCount, maxIterations)
				if reviewCount >= maxIterations {
					logx.Errorf("Reached review iteration limit without final report.")
					break
				}
			}
			continue
		}

		if fr, ok := ParseFinalReport(choice); ok {
			finalReport = fr
			finished = true
			break
		}
		logx.Infof("Assistant response was not a final report; continuing.")
	}

	if finished {
		_, err := finalizeBranchPush(handler, publishOpts, finalReport, true)
		if err != nil {
			return nil, err
		}
		return finalReport, nil
	}

	branchID, err := finalizeBranchPush(handler, publishOpts, nil, false)
	if err != nil {
		return nil, err
	}
	if branchID != "" {
		logx.Infof("Workspace published to branch (branch_id=%s) after iteration limit.", branchID)
	}
	return nil, errors.New("reached maximum iterations without final report")
}

func ChatLoop(brain *b.LLMBrain, handler *t.ToolHandler, messages []b.ChatMessage, maxIters int, publishOpts PublishOptions) (map[string]any, error) {
	if maxIters <= 0 {
		maxIters = maxIterations
	}
	tools := t.GetToolDefinitions()
	var (
		finalReport map[string]any
		finished    bool
		reviewCount int
	)

	for i := 1; ; i++ {
		fmt.Printf("[iter %d] requesting completion...\n", i)
		resp, err := brain.Complete(messages, tools)
		if err != nil {
			return nil, err
		}
		choice := resp.Choices[0].Message
		if choice.Content != "" {
			fmt.Printf("assistant> %s\n", choice.Content)
		}
		messages = append(messages, assistantMessageToDict(choice))

		if len(choice.ToolCalls) > 0 {
			reviewCompleted := false
			for _, tc := range choice.ToolCalls {
				fmt.Printf("tool> %s %s\n", tc.Function.Name, tc.Function.Arguments)
				var args map[string]any
				if tc.Function.Arguments != "" {
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				}
				htc := t.ToolCall{ID: tc.ID, Type: tc.Type}
				htc.Function.Name = tc.Function.Name
				htc.Function.Arguments = tc.Function.Arguments
				result := handler.Handle(htc)
				js := toJSON(result)
				if len(js) > 2000 {
					js = js[:2000]
				}
				fmt.Printf("tool< %s\n", js)
				messages = append(messages, b.ChatMessage{Role: "tool", ToolCallID: tc.ID, Content: toJSON(result)})

				if tc.Function.Name == "execute_agent" {
					if agent, _ := args["agent"].(string); agent == "codex" {
						if status, _ := result["status"].(string); status == "success" {
							reviewCompleted = true
						}
					}
				}
			}
			if reviewCompleted {
				reviewCount++
				fmt.Printf("note: completed review iteration %d/%d\n", reviewCount, maxIters)
				if reviewCount >= maxIters {
					logx.Errorf("Reached review iteration limit without final report.")
					break
				}
			}
			continue
		}
		if fr, ok := ParseFinalReport(choice); ok {
			finalReport = fr
			finished = true
			fmt.Println("assistant< final_report")
			break
		}
		fmt.Println("assistant< not final yet, continuing...")
	}

	if finished {
		_, err := finalizeBranchPush(handler, publishOpts, finalReport, true)
		if err != nil {
			return nil, err
		}
		return finalReport, nil
	}

	branchID, err := finalizeBranchPush(handler, publishOpts, nil, false)
	if err != nil {
		return nil, err
	}
	if branchID != "" {
		fmt.Fprintf(os.Stderr, "info: workspace pushed (branch_id=%s)\n", branchID)
	}
	return nil, errors.New("reached iteration limit without final report")
}

func toJSON(v any) string { b, _ := json.Marshal(v); return string(b) }
