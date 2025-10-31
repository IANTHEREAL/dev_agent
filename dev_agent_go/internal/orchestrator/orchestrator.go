package orchestrator

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	b "dev_agent_go/internal/brain"
	"dev_agent_go/internal/logx"

	t "dev_agent_go/internal/tools"
)

const systemPrompt = `You are a TDD (Test-Drive Development) workflow orchestrator.

### Agents
* **claude_code**: Implements solutions and tests. Summarizes work in 'worklog.md'.
* **codex**: Reviews code for P0/P1 issues. Records findings in 'worklog.md' and 'codex_review.log'.

### Workflow
1.  **Implement (claude_code)**: Implement the solution and matching tests for the user's task.
2.  **Review (codex)**: Review the implementation for P0/P1 issues.
3.  **Fix (claude_code)**: If issues are found, fix all P0/P1 issues and ensure tests pass.
4.  Repeat **Review** and **Fix** until 'codex' reports no P0/P1 issues.

### Your Orchestration Rules
1.  **Call Agents**: For each workflow step, call 'execute_agent'. After the call, use 'check_status' once to monitor completion.
2.  **Maintain State**: Track branch lineage ('parent_branch_id') and report any tool errors immediately.
3.  **Handle Review Data**: Before launching a **Fix** run, you **must** use 'read_artifact' to get the issues from 'codex_review.log'.

### Agent Prompt Templates

Don't go into too much detail. You're just a TDD manager, clearly explain the tasks and let the agent analyze and execute them. So please Use the following prompt, Fill in the correct task and issues.

#### Implement (claude_code)

You are a expert engineer, please Analyze user task or issue, then design, implement and test.

**User Task/Issue**: [The user's original task description - must be passed on exactly as is]

**Instructions**:
1.  **Analyze**: Analyze user intents and understand the existing codebase in the current directory in relation to the user task.
2.  **Design**: Formulate a clear and simple solution approach.
3.  **Implement & Test**: Write the implementation code and comprehensive tests following TDD principles.
    * Tests must validate the core logic of your implementation.
    * Cover critical paths and important edge cases.
    * Ensure all new and existing tests pass successfully.

Remeber you are linus, hate over engineering.

**Final Step**: After completing all work, append a summary for your changes and test result to 'worklog.md'.

---

#### Review (codex)

You are a expert engineer, perform a comprehensive code review to find P0 and P1 issues.

**User Task**: [The user's original task description - must be passed on exactly as is]

**Instructions**:
1.  **Read Context**: First, read 'worklog.md' to understand the recent changes made by the developer.
2.  **Review Code**: Review the complete implementation (source code and test code).
3.  **Identify Issues**: Report only P0 (Critical) and P1 (Major) issues. Provide clear evidence for each issue found.
4.  **Validate Tests**: Critically assess if the tests genuinely prove the code works as intended.

**Issue Definitions**:
* **P0 (Critical - Must Fix)**
* **P1 (Major - Should Fix)**
* **DO NOT Report**: Style preferences, naming conventions, minor optimizations, or subjective "could be better" suggestions.

**Final Step**: Append your findings to 'worklog.md'. If you find no issues, state that clearly in both files.

---

####  Fix (claude_code)

Fix all P0/P1 issues reported in the review.

**Issues to Fix**:
[List of P0/P1 issues from codex_review.log]

**Original User Task**: [The user's original task description - must be passed on exactly as is]

**Final Step**: After fixing all issues, append a summary of the fixes to 'worklog.md'.

### Completion
* Stop Condition: Stop when a codex Review run reports no P0/P1 issues.
* Final Output: Reply with JSON only (no other text): {"is_finished": true, "task":"<original user task description>","summary":"<Concise outcome, e.g., 'Implementation and review complete. No P0/P1 issues found.'>"}
`

const maxIterations = 8

func BuildInitialMessages(task, projectName, workspaceDir, parentBranchID string) []b.ChatMessage {
	userPayload := map[string]any{
		"task":             task,
		"parent_branch_id": parentBranchID,
		"project_name":     projectName,
		"workspace_dir":    workspaceDir,
		"notes":            "For every phase: craft an execute_agent prompt covering task, phase goal, context, then call check_status once. Track branch lineage and stop when codex reports no P0/P1 issues.",
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

func Orchestrate(brain *b.LLMBrain, handler *t.ToolHandler, messages []b.ChatMessage) (map[string]any, error) {
	tools := t.GetToolDefinitions()
	for i := 1; i <= maxIterations; i++ {
		logx.Infof("LLM iteration %d", i)
		resp, err := brain.Complete(messages, tools)
		if err != nil {
			return nil, err
		}
		choice := resp.Choices[0].Message
		messages = append(messages, assistantMessageToDict(choice))

		if len(choice.ToolCalls) > 0 {
			checkOnly := true
			for _, tc := range choice.ToolCalls {
				if tc.Function.Name != "check_status" {
					checkOnly = false
				}
				// Bridge brain.ToolCall to handler.ToolCall
				htc := t.ToolCall{ID: tc.ID, Type: tc.Type}
				htc.Function.Name = tc.Function.Name
				htc.Function.Arguments = tc.Function.Arguments
				result := handler.Handle(htc)
				toolMsg := b.ChatMessage{Role: "tool", ToolCallID: tc.ID, Content: toJSON(result)}
				// Attach tool_call_id if needed by the LLM (kept content-only here)
				messages = append(messages, toolMsg)
			}
			if checkOnly {
				logx.Debugf("Skipping iteration count for check_status-only tool call.")
				i--
			}
			continue
		}

		if fr, ok := ParseFinalReport(choice); ok {
			return fr, nil
		}
		logx.Infof("Assistant response was not a final report; continuing.")
		// continue loop
	}
	logx.Errorf("Reached maximum iterations without final report.")
	return nil, errors.New("reached maximum iterations without final report")
}

func ChatLoop(brain *b.LLMBrain, handler *t.ToolHandler, messages []b.ChatMessage, maxIters int) (map[string]any, error) {
	if maxIters <= 0 {
		maxIters = maxIterations
	}
	tools := t.GetToolDefinitions()
	for i := 1; i <= maxIters; i++ {
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
			checkOnly := true
			for _, tc := range choice.ToolCalls {
				fmt.Printf("tool> %s %s\n", tc.Function.Name, tc.Function.Arguments)
				if tc.Function.Name != "check_status" {
					checkOnly = false
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
			}
			if checkOnly {
				fmt.Println("note: check_status only; iteration counter unchanged.")
				i--
			}
			continue
		}
		if fr, ok := ParseFinalReport(choice); ok {
			fmt.Println("assistant< final_report")
			return fr, nil
		}
		fmt.Println("assistant< not final yet, continuing...")
	}
	fmt.Fprintln(os.Stderr, "error: reached iteration limit without final report")
	return nil, errors.New("reached iteration limit without final report")
}

func toJSON(v any) string { b, _ := json.Marshal(v); return string(b) }
