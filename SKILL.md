# TDD Agent

## Input
- Accept user tasks (feature requests, bug fixes, etc.) 
- `parent_branch_id`

## Funtional
- Delegate work to specialist agents (`claude_code`, `codex`) via pantheon mcp tool `parallel_explore`.
- Enforce the TDD workflow: implement → review → fix.
- Provide a final detailed report for implementation.

## Usage Notes
1. Set environment variables:
   ```bash
   export OPENAI_API_KEY=...
   export PROJECT_NAME=my-project
   export MCP_BASE_URL=http://localhost:8082/api/jsonrpc
   ```
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Execute:
   ```bash
   dev-agent --task "Add pagination to orders API" \
     --parent-branch-id 123e4567-e89b-12d3-a456-426614174000 \
     --workspace-dir /home/pan/workspace
   ```
4. CLI prints JSON summary with branch IDs and final status.
