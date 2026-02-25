package orchestrator

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	b "dev_agent/internal/brain"
	"dev_agent/internal/logx"
	"dev_agent/internal/streaming"

	t "dev_agent/internal/tools"
)

const systemPromptTemplate = `You are a expert software engineer, and a TDD (Test-Drive Development) workflow orchestrator.

### Agents
- **codex**: Analyze the requirement, Design and Implements solutions and tests. Summarizes work in '%[1]s/worklog.md'.
- **review_code**: Reviews code for P0/P1 issues. Records findings in '%[1]s/code_review.log'.

### Workflow
1.  **Implement (codex)**: Implement the solution and matching tests for the user's task.
2.  **Review (review_code)**: Review the implementation for P0/P1 issues.
3.  **Fix (codex)**: If issues are found, fix all P0/P1 issues and ensure tests pass.
4.  Repeat **Review** and **Fix** until 'review_code' agent reports no P0/P1 issues.

### Your Orchestration Rules
1.  **Single Call Per Turn**: Issue exactly one agent/tool call per assistant response; do not batch tool calls because each subsequent agent needs the prior branch's id to extend the branch lineage correctly.
2.  **Call Agents**: For each workflow step, the agent is invoked through the 'execute_agent'.
3.  **Maintain State**: Track branch lineage ('parent_branch_id') and report any tool errors immediately.
4.  **Local-Only Before Publish**: Implement/Review/Fix phases are strictly local development. You may create/checkout branches and stage/commit locally, but you must **NOT** run 'git push' or create PRs (e.g., via 'gh pr create') in these phases.

### Agent Prompt Templates

Don't go into too much detail. You're just a TDD manager, clearly explain the tasks and let the agent analyze and execute them. So please Use the following prompt, Fill in the correct task and issues.
Never hard-code absolute filesystem paths; derive locations relative to the repository or the configured workspace root (%[1]s).

---

#### Implement (codex)

You are an expert engineer. Your goal is to produce high-quality, verified code based on deep analysis.
Before you start coding: Read as much as you can, you have unlimited read quotas and available contexts. When you are not sure about something, you must study the code until you figure out.

**User Task**: [The user's original task description - must be passed on exactly as is]

**Instructions**:

1.  **Phase 0: Context Verification (CRITICAL)**
    * Identify the issue or requirement metioned in the User Task (e.g., GitHub Issue IDs, specific requirement/error messages, requirement doc).
    * **Abort Condition**: If you cannot verify or locate the specific references (e.g., an Issue ID returns 404, or a mentioned file doesn't exist), you must **STOP IMMEDIATELY**.
        * Do not proceed to design or code.
        * Write a "Context Failure Report" to '%[1]s/worklog.md' explaining what was missing.
        * Inform the user that the task cannot be processed due to missing context.

	Hints: if needed, Use the 'gh' CLI to inspect GitHub issues/PRs just like 'git'; if either tool lacks auth, run '~/.setup-git.sh' to configure both before proceeding.


2.  **Phase 1: Analysis & Design** (Only if Phase 0 passes)
	* Read as much as you can, you have unlimited read quotas and available contexts. When you are not sure about something, you must study the code until you figure out.
    
    **1.1 Root Cause Analysis**
    * **Bugs**: Locate exact issue location (file/function/line). Trace symptoms→root cause via error stacks/logs/data flows. Identify fundamental reason (logic/race/assumption/validation issues), not symptoms.
    * **Features**: Identify all code paths/files needing modification. Understand architecture and feature integration.
    
    **1.2 Dependency Analysis** (Find ALL affected code)
    * **Callers**: Find ALL functions calling problematic code (direct + indirect via call chains). Use grep/go doc/go list.
    * **Callees**: Find ALL functions called by problematic code (direct + indirect dependencies).
    * **Data Flow**: Trace data paths in/out. Identify shared state (globals, package vars, shared structs). Find config dependencies.
    * **Control Flow**: Analyze conditional branches, exception paths, concurrent/async paths if applicable.
    * **Modules**: Identify affected packages/modules, package dependencies/interfaces, public API impact.
    
    **1.3 Configuration & Parameters** (Find system settings)
    * **Config Files**: Scan config.yaml/json/toml/.env. Find related params (timeouts, retries, limits). Document current/default values.
    * **Runtime Params**: Find timeouts ("timeout", "deadline", "30 minutes"), retry limits ("maxRetries", "maxIterations"), concurrency ("maxConcurrency", "poolSize"), buffers ("bufferSize", "cacheSize").
    * **Env Vars**: Check os.Getenv/os.LookupEnv usage. Document relevant vars.
    * **Constants**: Find const/var declarations, magic numbers, defaults/limits, config-related constants.
    * **Impact**: Understand how each parameter affects the issue. Note which may need adjustment and validation rules.
    
    **1.4 Impact Assessment**
    * **Code**: Direct (must modify) + indirect (may be affected: callers/callees/shared state). List: [File] → [Function] → [Reason].
    * **Functional**: Affected modules/components, user-visible changes, feature dependencies.
    * **Test**: Existing test files, tests needing updates, coverage gaps, regression tests for bugs.
    * **Performance** (if applicable): Performance-critical paths, potential performance implications.
    * **Integration**: Integration points with other modules/services, external dependencies.
    
    **1.5 Design Strategy & Documentation**
    * Outline solution: fix approach, modification plan (files/functions), config changes, test strategy, risk mitigation.
    * **Write to '%[1]s/worklog.md'**: Root cause, affected locations (direct+indirect), config/parameter findings, impact assessment, design strategy.

3.  **Phase 2: TDD Implementation**
	* **CRITICAL: Use Phase 1 Analysis from '%[1]s/worklog.md'**
		* Read worklog.md before starting. It contains: root cause, all affected locations (direct+indirect), config/parameter findings, impact assessment, design strategy.
		* **MUST address ALL identified code locations** (not just direct problem).
		* **MUST consider ALL identified configuration parameters**.
		* **MUST follow Phase 1's test strategy**.
	
	* **Test First**: Write tests before code. For bugs, add regression tests.
		* Use Phase 1's test impact analysis. Add tests for identified coverage gaps. Cover direct + indirect impact areas.
	
	* **Implement**: Fix root cause. Modify ALL files/functions from Phase 1 impact assessment (direct + indirect). Adjust config parameters (config files/constants/env vars). Handle all dependencies (callers/callees/shared state). Follow Phase 1 design strategy.
	
	* **Verify**: Run all relevant tests (direct/indirect affected code, new regression tests, updated existing tests).
	
    * **Git Discipline**: Work locally only. You may create/checkout branches and stage/commit locally, but do **NOT** push, and do **NOT** create PRs (e.g., via 'gh pr create') during this phase.

4.  **Final Step**: Update '%[1]s/worklog.md' with: files actually modified, deviations from Phase 1 plan (if any) and reasons, test results, confirmation that all identified locations were addressed.

Ultrathink! Analyze first, then code. Avoid over-engineering.
---

#### Review (review_code)

**User Task**: [The user's original task description]

**Phase 1 Analysis Reference** (Optional but recommended):
* If '%[1]s/worklog.md' contains Phase 1 analysis, review it to understand:
	* The intended scope of changes (all affected code locations)
	* Configuration parameters that should have been adjusted
	* Test strategy that should have been followed
* Use this context to verify that the implementation addressed all identified areas.

**Instructions**:
1.  **Review Code Changes**: Review the recent modifications and tests to determine if they satisfy the User Task.
	* Check if the implementation addressed all code locations identified in Phase 1 analysis (if available).
	* Verify that configuration parameters identified in Phase 1 were properly handled.
	* Confirm that the test strategy from Phase 1 was followed.
2.  **Check Merge Conflicts** (Pre-PR): Fetch latest remote (git fetch origin), find base branch (main/master), test merge (git merge --no-commit --no-ff origin/main then abort). If conflicts found, report as **P0**. If same files modified but no conflict, report as **P1** risk.
3.  **Scope**: 
    * **Review Focus**: Focus on reviewing the changed code and its direct impact. Do NOT review unrelated legacy code or pre-existing issues unless they are made worse by this change.
    * **Phase 1 Verification**: Verify that all code locations identified in Phase 1 analysis were properly addressed (including both direct and indirect impact areas). If Phase 1 identified indirect impact areas that should have been addressed but weren't, flag this as a P1 issue.
4.  **Report**: Identify and log **P0 (Critical)** or **P1 (Major)** issues to '%[1]s/code_review.log', including merge conflict check results.
    * If the code meets the requirements and has no critical/major issues, report "No P0/P1 issues found".
    * If Phase 1 analysis identified areas that weren't addressed, mention this in the review.

Hints: if needed, Use the 'gh' CLI to inspect GitHub issues/PRs just like 'git'; if either tool lacks auth, run '~/.setup-git.sh' to configure both before proceeding.

Think it hard and

---

####  Fix (codex)

Ultrathink! Fix all P0/P1 issues reported in the review.

**Issues to Fix**:
[List of P0/P1 issues from '%[1]s/code_review.log']

**Original User Task**: [The user's original task description]

**Instructions**:
1.  **Address Issues**: Systematically fix every P0 and P1 issue listed.
2.  **Verify**: Ensure existing tests pass and add new tests if the review indicated missing coverage.
3.  **Update Log**: Append a "Fix Summary" to '%[1]s/worklog.md' explaining what was changed.
4.  **Git Discipline**: Work locally only. You may create/checkout branches and stage/commit locally, but do **NOT** push, and do **NOT** create PRs (e.g., via 'gh pr create') during this phase.

Hints: if needed, Use the 'gh' CLI to inspect GitHub issues/PRs just like 'git'; if either tool lacks auth, run '~/.setup-git.sh' to configure both before proceeding.

Ultrathink! Analyze first, then code. Avoid over-engineering.

### Completion
* Stop Condition: Stop when a review_code run reports no P0/P1 issues.
* Final Output: Reply with JSON only: {"is_finished": true, "task":"<original task>","summary":"<Concise outcome>"}
`

const (
	statusCompleted         = "completed"
	statusIterationLimit    = "iteration_limit"
	statusFinishedWithError = "FINISHED_WITH_ERROR"

	iterationLimitSummary = "Reached iteration limit before clean review sign-off."
	defaultSuccessSummary = "Workflow completed successfully."
)

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
	GitUserName    string
	GitUserEmail   string
}

type RunOptions struct {
	Publish  PublishOptions
	Streamer *streaming.JSONStreamer
}

func finalizeBranchPush(handler publishHandler, opts PublishOptions, report map[string]any, success bool, emitter *eventEmitter) (string, error) {
	lineage := handler.BranchRange()
	parent := lineage["latest_branch_id"]
	if parent == "" {
		parent = opts.ParentBranchID
	}
	if parent == "" {
		return "", errors.New("unable to determine parent branch id for publish step")
	}

	outcome := iterationLimitSummary
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
			outcome = defaultSuccessSummary
		}
	}

	meta := fmt.Sprintf("commit-meta: start_branch=%s latest_branch=%s", lineage["start_branch_id"], lineage["latest_branch_id"])
	prompt := fmt.Sprintf(`Finalize the task by committing and pushing the current workspace state.

Task: %[1]s
Outcome: %[2]s
Meta (include in the commit message if helpful): %[3]s

The worklog is located into '%[4]s/worklog.md'.

Choose an appropriate git branch name for this task, commit the related file changes, and reply with a concise publish report that MUST include: repository URL, pushed Git branch name, commit hash, and pointers to the latest implementation summary/tests (e.g., '%[4]s/worklog.md' and any test artifact).

Publishing rules:
- Use existing git identity and credentials. If you hit permission/auth issues, run '~/.setup-git.sh' once to configure git and retry. If it still fails, stop and report the failure.
- Use the original user task and the latest entries in '%[4]s/worklog.md' to determine the target repository; confirm the repository root with 'git rev-parse --show-toplevel' and verify the remote via 'git remote -v'. Do not operate on an unrelated repo.
- If you cannot confirm a valid git repository (rev-parse/root or remotes are missing), stop immediately, summarize the delivered work (reference '%[4]s/worklog.md' and tests), and exit instead of attempting any git commands.
- Stage and commit only the files required for this task; exclude logs, review artifacts, and temporary scratch files.
- Keep branch names kebab-case and describe the task scope.
- Keep the commit subject <= 72 characters and meaningful.
- Git push must be fully non-interactive. Rely on existing credentials or the setup script; do not reveal secrets in logs.
- Do not stage or commit '%[4]s/worklog.md' or '%[4]s/code_review.log'.

Include a short publish report that states the repository URL, branch name, and a concise PR-style summary.`, opts.Task, outcome, meta, opts.WorkspaceDir, opts.WorkspaceDir)

	logx.Infof("Finalizing workflow by asking codex to push from branch %s lineage.", parent)
	execArgs := map[string]any{
		"agent":            "codex",
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

	var (
		itemID   string
		start    time.Time
		duration time.Duration
	)
	if emitter != nil {
		args := map[string]any{
			"agent":            "codex",
			"parent_branch_id": parent,
		}
		if opts.ProjectName != "" {
			args["project_name"] = opts.ProjectName
		}
		itemID = emitter.ItemStarted("publish", "publish", args)
		start = time.Now()
	}

	execResp := handler.Handle(execCall)
	if !start.IsZero() {
		duration = time.Since(start)
	}

	data, _ := execResp["data"].(map[string]any)
	branchID := t.ExtractBranchID(data)
	if branchID == "" {
		branchID = t.ExtractBranchID(execResp)
	}
	publishSummary := extractBranchOutput(data)
	if publishSummary == "" {
		publishSummary = summarizeToolResult(execResp)
	}

	status := resultStatus(execResp)
	if emitter != nil {
		emitter.ItemCompleted(itemID, status, duration, branchID, publishSummary)
	}

	if status != "success" {
		return "", fmt.Errorf("publish execute_agent failed: %v", execResp)
	}
	if branchID == "" {
		return "", errors.New("publish execute_agent missing branch id")
	}
	if publishSummary == "" {
		logx.Warningf("Publish response missing required report (repo/branch/commit/tests); continuing without it (branch_id=%s)", branchID)
		publishSummary = fmt.Sprintf("Publish report unavailable; inspect Pantheon branch %s for push details.", branchID)
	}
	if report != nil {
		report["publish_report"] = publishSummary
	}
	if branchStatus := strings.TrimSpace(fmt.Sprintf("%v", data["status"])); branchStatus != "" {
		switch strings.ToLower(branchStatus) {
		case "failed":
			return "", fmt.Errorf("publish branch %s completed with failure status", branchID)
		}
	}

	return branchID, nil
}

func BuildInitialMessages(task, projectName, workspaceDir, parentBranchID string) []b.ChatMessage {
	systemPrompt := fmt.Sprintf(systemPromptTemplate, workspaceDir)
	userPayload := map[string]any{
		"task":             task,
		"parent_branch_id": parentBranchID,
		"project_name":     projectName,
		"workspace_dir":    workspaceDir,
		"notes":            "For every phase: craft a single execute_agent prompt covering task, phase goal, context. Do not batch tool calls. Track branch lineage and stop when review_code reports no P0/P1 issues.",
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

func Orchestrate(brain *b.LLMBrain, handler *t.ToolHandler, messages []b.ChatMessage, opts RunOptions) (map[string]any, error) {
	tools := t.GetToolDefinitions()
	emitter := newEventEmitter(opts.Streamer)
	var (
		finalReport    map[string]any
		finished       bool
		errorState     bool
		reviewCount    int
		totalToolCalls int
		lastTurn       int
	)

	for i := 1; ; i++ {
		lastTurn = i
		logx.Infof("LLM iteration %d", i)
		turnID := fmt.Sprintf("turn_%d", i)
		if emitter != nil {
			emitter.TurnStarted(turnID, i, len(messages), totalToolCalls)
		}
		resp, err := brain.Complete(messages, tools)
		if err != nil {
			if emitter != nil {
				emitter.EmitError("llm.complete", err.Error(), map[string]any{"iteration": i, "turn_id": turnID})
			}
			return nil, err
		}
		choice := resp.Choices[0].Message
		messages = append(messages, assistantMessageToDict(choice))
		if emitter != nil {
			emitter.AssistantMessage(turnID, choice.Content, len(choice.ToolCalls))
		}

		if len(choice.ToolCalls) > 0 {
			turnToolCount := 0
			reviewCompleted := false
			stopDueToInstruction := false
			for _, tc := range choice.ToolCalls {
				turnToolCount++
				totalToolCalls++
				args := parseToolArgs(tc.Function.Arguments)
				var itemArgs map[string]any
				if emitter != nil {
					itemArgs = sanitizeToolArgs(tc.Function.Name, args)
				}
				itemID := ""
				if emitter != nil {
					itemID = emitter.ItemStarted("tool_call", tc.Function.Name, itemArgs)
				}
				htc := t.ToolCall{ID: tc.ID, Type: tc.Type}
				htc.Function.Name = tc.Function.Name
				htc.Function.Arguments = tc.Function.Arguments
				var start time.Time
				if emitter != nil {
					start = time.Now()
				}
				result := handler.Handle(htc)
				var duration time.Duration
				if emitter != nil {
					duration = time.Since(start)
				}
				toolMsg := b.ChatMessage{Role: "tool", ToolCallID: tc.ID, Content: toJSON(result)}
				messages = append(messages, toolMsg)
				if emitter != nil {
					emitter.ItemCompleted(itemID, resultStatus(result), duration, eventBranchID(result), summarizeToolResult(result))
				}

				if instr, summaryMsg, details := toolInstruction(result); instr != "" {
					if emitter != nil {
						emitter.EmitError("tool_instruction", summaryMsg, map[string]any{"instruction": instr})
					}
					finalReport = buildErrorFinalReport(opts.Publish.Task, summaryMsg, instr, details)
					finished = true
					errorState = true
					stopDueToInstruction = true
					break
				}

				if tc.Function.Name == "execute_agent" {
					if agent, _ := args["agent"].(string); agent == "review_code" {
						if status, _ := result["status"].(string); status == "success" {
							reviewCompleted = true
						}
					}
				}
			}
			if emitter != nil {
				emitter.TurnCompleted(turnID, i, turnToolCount, false)
			}
			if stopDueToInstruction {
				break
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

		hasFinal := false
		if fr, ok := ParseFinalReport(choice); ok {
			finalReport = fr
			finished = true
			hasFinal = true
		} else {
			logx.Infof("Assistant response was not a final report; continuing.")
		}
		if emitter != nil {
			emitter.TurnCompleted(turnID, i, 0, hasFinal)
		}
		if finished {
			break
		}
	}

	runPublish := func(report map[string]any, success bool) (string, error) {
		turnNum := lastTurn + 1
		var turnID string
		if emitter != nil {
			turnID = fmt.Sprintf("turn_%d", turnNum)
			emitter.TurnStarted(turnID, turnNum, len(messages), totalToolCalls)
		}
		branchID, err := finalizeBranchPush(handler, opts.Publish, report, success, emitter)
		totalToolCalls++
		lastTurn = turnNum
		if emitter != nil {
			emitter.TurnCompleted(turnID, turnNum, 1, false)
		}
		return branchID, err
	}

	if finished {
		if errorState {
			ensureReportDefaults(finalReport, opts.Publish.Task, statusFinishedWithError, true)
			return finalReport, nil
		}
		ensureReportDefaults(finalReport, opts.Publish.Task, statusCompleted, true)
		_, err := runPublish(finalReport, true)
		if err != nil {
			if emitter != nil {
				emitter.EmitError("publish", err.Error(), nil)
			}
			return nil, err
		}
		return finalReport, nil
	}

	finalReport = map[string]any{
		"is_finished": false,
		"status":      statusIterationLimit,
		"task":        opts.Publish.Task,
		"summary":     iterationLimitSummary,
	}
	branchID, err := runPublish(finalReport, false)
	if err != nil {
		if emitter != nil {
			emitter.EmitError("publish", err.Error(), nil)
		}
		return nil, err
	}
	if branchID != "" {
		logx.Infof("Workspace published to branch (branch_id=%s) after iteration limit.", branchID)
	}
	return finalReport, nil
}

func ChatLoop(brain *b.LLMBrain, handler *t.ToolHandler, messages []b.ChatMessage, maxIters int, opts RunOptions) (map[string]any, error) {
	if maxIters <= 0 {
		maxIters = maxIterations
	}
	tools := t.GetToolDefinitions()
	var (
		finalReport map[string]any
		finished    bool
		errorState  bool
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
			stopDueToInstruction := false
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

				if instr, summaryMsg, details := toolInstruction(result); instr != "" {
					finalReport = buildErrorFinalReport(opts.Publish.Task, summaryMsg, instr, details)
					finished = true
					errorState = true
					stopDueToInstruction = true
					break
				}

				if tc.Function.Name == "execute_agent" {
					if agent, _ := args["agent"].(string); agent == "review_code" {
						if status, _ := result["status"].(string); status == "success" {
							reviewCompleted = true
						}
					}
				}
			}
			if stopDueToInstruction {
				break
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
		if errorState {
			ensureReportDefaults(finalReport, opts.Publish.Task, statusFinishedWithError, true)
			return finalReport, nil
		}
		ensureReportDefaults(finalReport, opts.Publish.Task, statusCompleted, true)
		_, err := finalizeBranchPush(handler, opts.Publish, finalReport, true, nil)
		if err != nil {
			return nil, err
		}
		return finalReport, nil
	}

	finalReport = map[string]any{
		"is_finished": false,
		"status":      statusIterationLimit,
		"task":        opts.Publish.Task,
		"summary":     iterationLimitSummary,
	}
	branchID, err := finalizeBranchPush(handler, opts.Publish, finalReport, false, nil)
	if err != nil {
		return nil, err
	}
	if branchID != "" {
		fmt.Fprintf(os.Stderr, "info: workspace pushed (branch_id=%s)\n", branchID)
	}
	return finalReport, nil
}

func toJSON(v any) string { b, _ := json.Marshal(v); return string(b) }

func ensureReportDefaults(report map[string]any, task, status string, finished bool) {
	if report == nil {
		return
	}
	if _, ok := report["task"]; !ok && task != "" {
		report["task"] = task
	}
	if _, ok := report["status"]; !ok && status != "" {
		report["status"] = status
	}
	if _, ok := report["is_finished"]; !ok {
		report["is_finished"] = finished
	}
}

func BuildInstructions(report map[string]any) string {
	if report == nil {
		return ""
	}
	start := reportString(report, "start_branch_id")
	latest := reportString(report, "latest_branch_id")
	status := reportString(report, "status")
	publishReport := reportString(report, "publish_report")

	var parts []string

	switch {
	case start != "" && latest != "":
		if start == latest {
			parts = append(parts, fmt.Sprintf("Branch lineage: start=%s, latest=%s. Inspect manifest %s in Pantheon to review artifacts.", start, latest, latest))
		} else {
			parts = append(parts, fmt.Sprintf("Branch lineage: start=%s → latest=%s. Inspect manifest %s in Pantheon to review artifacts.", start, latest, latest))
		}
	case latest != "":
		parts = append(parts, fmt.Sprintf("Inspect manifest %s in Pantheon to review artifacts.", latest))
	case start != "":
		parts = append(parts, fmt.Sprintf("Branch lineage started from %s; inspect it in Pantheon to review artifacts.", start))
	}

	if publishReport != "" {
		parts = append(parts, fmt.Sprintf("Publish report describes the GitHub push target: %s", publishReport))
	}

	switch status {
	case statusIterationLimit:
		target := latest
		if target == "" {
			target = start
		}
		if target != "" {
			parts = append(parts, fmt.Sprintf("Next (if your are allowed or instructed), you can rerun dev-agent with --parent-branch-id %s to continue automated iterations;", target))
		}
	case statusFinishedWithError:
		parts = append(parts, "Workflow stopped in FINISHED_WITH_ERROR. See error.message for details.")
	default:
		if publishReport != "" {
			parts = append(parts, "Next step: review the pushed GitHub branch and, based on your process, proceed with the normal PR/merge workflow.")
		} else if latest != "" {
			parts = append(parts, "Next step: review the manifest and test results above, then proceed with whichever merge/publish flow fits your process.")
		}
	}

	return strings.TrimSpace(strings.Join(parts, " "))
}

func reportString(report map[string]any, key string) string {
	if report == nil {
		return ""
	}
	if v, ok := report[key]; ok && v != nil {
		return strings.TrimSpace(fmt.Sprintf("%v", v))
	}
	return ""
}

func extractBranchOutput(data map[string]any) string {
	if data == nil {
		return ""
	}
	branch, _ := data["branch"].(map[string]any)
	if branch == nil {
		return ""
	}
	if out, _ := branch["output"].(string); strings.TrimSpace(out) != "" {
		return strings.TrimSpace(out)
	}
	if snap, _ := branch["latest_snap"].(map[string]any); snap != nil {
		if out, _ := snap["output"].(string); strings.TrimSpace(out) != "" {
			return strings.TrimSpace(out)
		}
	}
	if manifest, _ := branch["manifest"].(map[string]any); manifest != nil {
		if summary, _ := manifest["summary"].(string); strings.TrimSpace(summary) != "" {
			return strings.TrimSpace(summary)
		}
	}
	return ""
}

type eventEmitter struct {
	streamer *streaming.JSONStreamer
	nextItem int
}

func newEventEmitter(streamer *streaming.JSONStreamer) *eventEmitter {
	if streamer == nil || !streamer.Enabled() {
		return nil
	}
	return &eventEmitter{streamer: streamer}
}

func (e *eventEmitter) TurnStarted(turnID string, iteration, messageCount, toolCount int) {
	if e == nil {
		return
	}
	e.streamer.EmitTurnStarted(turnID, iteration, messageCount, toolCount)
}

func (e *eventEmitter) AssistantMessage(turnID, preview string, toolCalls int) {
	if e == nil {
		return
	}
	e.streamer.EmitAssistantMessage(turnID, preview, toolCalls)
}

func (e *eventEmitter) TurnCompleted(turnID string, iteration, toolCalls int, hasFinal bool) {
	if e == nil {
		return
	}
	e.streamer.EmitTurnCompleted(turnID, iteration, toolCalls, hasFinal)
}

func (e *eventEmitter) ItemStarted(kind, name string, args map[string]any) string {
	if e == nil {
		return ""
	}
	e.nextItem++
	itemID := fmt.Sprintf("item_%d", e.nextItem)
	e.streamer.EmitItemStarted(itemID, kind, name, args)
	return itemID
}

func (e *eventEmitter) ItemCompleted(itemID, status string, duration time.Duration, branchID, summary string) {
	if e == nil || itemID == "" {
		return
	}
	e.streamer.EmitItemCompleted(itemID, status, duration, branchID, summary)
}

func (e *eventEmitter) EmitError(scope, message string, extra map[string]any) {
	if e == nil {
		return
	}
	e.streamer.EmitError(scope, message, extra)
}

func parseToolArgs(raw string) map[string]any {
	if strings.TrimSpace(raw) == "" {
		return map[string]any{}
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return map[string]any{}
	}
	return args
}

func sanitizeToolArgs(name string, args map[string]any) map[string]any {
	if len(args) == 0 {
		return map[string]any{}
	}
	out := map[string]any{}
	switch name {
	case "execute_agent":
		copyStringField(out, args, "agent")
		copyStringField(out, args, "project_name")
		copyStringField(out, args, "parent_branch_id")
		if prompt, _ := args["prompt"].(string); prompt != "" {
			preview := streaming.PromptPreview(prompt)
			out["prompt_preview"] = preview
			if strings.TrimSpace(prompt) != preview {
				out["prompt_truncated"] = true
			}
		}
	case "read_artifact":
		copyStringField(out, args, "branch_id")
		copyStringField(out, args, "path")
	case "branch_output":
		copyStringField(out, args, "branch_id")
		copyBoolField(out, args, "full_output")
	default:
		for k, v := range args {
			switch val := v.(type) {
			case string:
				out[k] = streaming.PromptPreview(val)
			case float64, bool:
				out[k] = val
			}
		}
	}
	return out
}

func copyStringField(dst, src map[string]any, key string) {
	if val, ok := src[key].(string); ok && strings.TrimSpace(val) != "" {
		dst[key] = strings.TrimSpace(val)
	}
}

func copyFloatField(dst, src map[string]any, key string) {
	if val, ok := src[key].(float64); ok {
		dst[key] = val
	}
}

func copyBoolField(dst, src map[string]any, key string) {
	if val, ok := src[key].(bool); ok {
		dst[key] = val
	}
}

func resultStatus(resp map[string]any) string {
	if resp == nil {
		return "error"
	}
	if status, ok := resp["status"].(string); ok && strings.TrimSpace(status) != "" {
		return strings.ToLower(strings.TrimSpace(status))
	}
	return "error"
}

func eventBranchID(resp map[string]any) string {
	if resp == nil {
		return ""
	}
	if data, _ := resp["data"].(map[string]any); data != nil {
		if id := t.ExtractBranchID(data); id != "" {
			return id
		}
	}
	return t.ExtractBranchID(resp)
}

func summarizeToolResult(resp map[string]any) string {
	if resp == nil {
		return ""
	}
	if errMsg, _ := resp["error"].(string); strings.TrimSpace(errMsg) != "" {
		return streaming.PromptPreview(errMsg)
	}
	if errObj, _ := resp["error"].(map[string]any); errObj != nil {
		if msg, _ := errObj["message"].(string); strings.TrimSpace(msg) != "" {
			return streaming.PromptPreview(msg)
		}
		if instr, _ := errObj["instruction"].(string); strings.TrimSpace(instr) != "" {
			return fmt.Sprintf("instruction=%s", strings.TrimSpace(instr))
		}
	}
	if data, _ := resp["data"].(map[string]any); data != nil {
		if out, _ := data["response"].(string); strings.TrimSpace(out) != "" {
			return streaming.PromptPreview(out)
		}
		if status, _ := data["status"].(string); strings.TrimSpace(status) != "" {
			return fmt.Sprintf("status=%s", strings.TrimSpace(status))
		}
	}
	if status, _ := resp["status"].(string); strings.TrimSpace(status) != "" {
		return fmt.Sprintf("status=%s", strings.TrimSpace(status))
	}
	return ""
}

func toolInstruction(resp map[string]any) (string, string, map[string]any) {
	if resp == nil {
		return "", "", nil
	}
	if errObj, ok := resp["error"].(map[string]any); ok {
		instr := strings.TrimSpace(reportString(errObj, "instruction"))
		message := strings.TrimSpace(reportString(errObj, "message"))
		var details map[string]any
		if det, ok := errObj["details"].(map[string]any); ok && len(det) > 0 {
			details = det
		}
		return instr, message, details
	}
	if msg, ok := resp["error"].(string); ok {
		return "", strings.TrimSpace(msg), nil
	}
	return "", "", nil
}

func buildErrorFinalReport(task, summary, instruction string, details map[string]any) map[string]any {
	out := map[string]any{
		"is_finished": true,
		"status":      statusFinishedWithError,
	}
	if strings.TrimSpace(task) != "" {
		out["task"] = strings.TrimSpace(task)
	}
	summary = strings.TrimSpace(summary)
	if summary == "" {
		summary = "Workflow halted due to a tool execution error."
	}
	out["summary"] = summary
	if instruction != "" {
		out["instruction"] = instruction
	}
	errPayload := map[string]any{}
	if summary != "" {
		errPayload["message"] = summary
	}
	if instruction != "" {
		errPayload["instruction"] = instruction
	}
	if len(details) > 0 {
		errPayload["details"] = details
	}
	if len(errPayload) > 0 {
		out["error"] = errPayload
	}
	return out
}
