# TDD Agent

## Input
- Accept user tasks (feature requests, bug fixes, etc.) 
- `parent_branch_id`

## Funtional
- Delegate work to specialist agents (`claude_code`, `codex`) via pantheon mcp tool `parallel_explore`.
- Enforce the TDD workflow: implement → review → fix.
- Provide a final detailed report for implementation.

## Usage Notes
1. Configure environment (either .env or exports):
   ```bash
   # Option A: create a .env file at repo root
   cat > .env << 'EOF'
   OPENAI_API_KEY=sk-...
   PROJECT_NAME=my-project
   MCP_BASE_URL=http://localhost:8000/mcp/sse
   EOF

   # Option B: export vars in your shell
   # export OPENAI_API_KEY=...
   # export PROJECT_NAME=my-project
   # export MCP_BASE_URL=http://localhost:8000/mcp/sse
   ```
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Execute:
   ```bash
   dev-agent --task "Add pagination to orders API" \
     --parent-branch-id 123e4567-e89b-12d3-a456-426614174000
   ```
   Or use interactive chat:
   ```bash
   dev-agent-chat --parent-branch-id 123e4567-e89b-12d3-a456-426614174000 --task "Add pagination to orders API"
   ```
4. CLI prints JSON summary with branch IDs and final status.
